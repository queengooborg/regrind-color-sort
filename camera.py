# Detect plastic regrind against background and classify color
# Mostly written by ChatGPT

import cv2
import numpy as np
import time
import json
import os
from collections import deque

# ========================= Defaults (changed via Settings panel) =========================
SETTINGS = {
    "d_ab": 12.0,
    "dL": 18.0,
    "use_edges": True,
    "update_bg": False,
    "min_area": 1500,
    "deltaE_thresh": 18.0,
    "hide_unlabeled": False,
    "palette_path": "palette.json",
    "camera_index": 0,
    "width": 1280,
    "height": 720
}

# ========================= Per-pixel Lab BG model =========================
class PixelBG:
    def __init__(self, H, W):
        self.mu_lab = np.zeros((H, W, 3), np.float32)
        self.var_ab = np.ones((H, W, 2), np.float32) * 9.0
        self.ready = False

    def init_from(self, lab):
        self.mu_lab[:] = lab
        self.var_ab[:] = 9.0
        self.ready = True

    def update(self, lab, fg_mask, alpha=0.02):
        if not self.ready:
            return
        bg = fg_mask == 0
        if not np.any(bg):
            return
        bg3 = bg[:, :, None]
        self.mu_lab[bg3] = (1 - alpha) * self.mu_lab[bg3] + alpha * lab[bg3]
        delta_ab = lab[:, :, [1, 2]] - self.mu_lab[:, :, [1, 2]]
        self.var_ab[bg] = (1 - alpha) * self.var_ab[bg] + alpha * (delta_ab[bg] ** 2)

# ========================= Segmentation =========================
def segment(frame_bgr, bg):
    blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    if not bg.ready:
        return np.zeros(frame_bgr.shape[:2], np.uint8), lab

    ab = lab[:, :, 1:3]
    ab0 = bg.mu_lab[:, :, 1:3]
    dab = np.linalg.norm(ab - ab0, axis=2)

    dL = lab[:, :, 0] - bg.mu_lab[:, :, 0]

    mask_chroma = (dab >= SETTINGS["d_ab"]).astype(np.uint8) * 255
    mask_L = ((np.abs(dL) >= SETTINGS["dL"]) & (dab < SETTINGS["d_ab"] * 0.7)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask_chroma, mask_L)

    if SETTINGS["use_edges"]:
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
        mask = cv2.bitwise_or(mask, edges)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)

    return mask, lab

# ========================= Palette manager (empty by default) =========================
KEY_POOL = list("1234567890-=[];',.")

class Palette:
    def __init__(self, path=None):
        self.path = path
        self.meta = {"deltaE_thresh": SETTINGS["deltaE_thresh"]}
        self.colors = []
        if path and os.path.exists(path):
            self.load(path)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.meta = data.get("meta", self.meta)
        self.colors = data.get("colors", [])
        self.path = path
        SETTINGS["deltaE_thresh"] = float(self.meta.get("deltaE_thresh", SETTINGS["deltaE_thresh"]))

    def save(self, path=None):
        if path is None:
            path = self.path or SETTINGS["palette_path"]
        self.meta["deltaE_thresh"] = SETTINGS["deltaE_thresh"]
        with open(path, "w") as f:
            json.dump({"version": 1, "meta": self.meta, "colors": self.colors}, f, indent=2)
        self.path = path

    def legend(self):
        if not self.colors:
            return "(empty)"
        parts = []
        for c in self.colors:
            if c.get("key"):
                parts.append(f"{c.get('key')}:{c['name']}")
            else:
                parts.append(f"•:{c['name']}")
        return ", ".join(parts)

    def key_to_index(self, k):
        for i, c in enumerate(self.colors):
            if c.get("key") == k:
                return i
        return None

    def auto_key(self):
        used = {c.get("key") for c in self.colors if c.get("key")}
        for k in KEY_POOL:
            if k not in used:
                return k
        return None

    @staticmethod
    def _deltaE76(c1, c2):
        c1 = np.array(c1, np.float32)
        c2 = np.array(c2, np.float32)
        return float(np.linalg.norm(c1 - c2))

    def classify_lab(self, lab_pixels):
        if len(self.colors) == 0:
            return None
        if len(lab_pixels) == 0:
            return None

        L = float(np.median(lab_pixels[:, 0]))
        a = float(np.median(lab_pixels[:, 1]))
        b = float(np.median(lab_pixels[:, 2]))

        best_name = None
        best_d = 1e9
        best_idx = None

        for i, c in enumerate(self.colors):
            cent = c.get("centroid")
            if not cent:
                continue
            d = self._deltaE76([L, a, b], cent)
            if d < best_d:
                best_d = d
                best_name = c["name"]
                best_idx = i

        if best_idx is None:
            return None

        if best_d <= float(SETTINGS["deltaE_thresh"]):
            return best_name, best_d, best_idx

        return None

    def add_class_from_pixels(self, name, lab_pixels, key=None):
        if len(lab_pixels) == 0:
            return None

        L = float(np.median(lab_pixels[:, 0]))
        a = float(np.median(lab_pixels[:, 1]))
        b = float(np.median(lab_pixels[:, 2]))

        if key is None or len(key) == 0:
            key = self.auto_key()

        entry = {
            "name": name,
            "key": key if key else None,
            "centroid": [L, a, b],
            "samples": 1
        }

        self.colors.append(entry)
        return entry

    def add_sample(self, idx, lab_pixels):
        if idx is None:
            return
        if idx < 0:
            return
        if idx >= len(self.colors):
            return
        if len(lab_pixels) == 0:
            return

        L = float(np.median(lab_pixels[:, 0]))
        a = float(np.median(lab_pixels[:, 1]))
        b = float(np.median(lab_pixels[:, 2]))

        c = self.colors[idx]
        if not c.get("centroid"):
            c["centroid"] = [L, a, b]
            c["samples"] = 1
            return

        n = int(c.get("samples", 1))
        alpha = 1.0 / (n + 1)

        c["centroid"][0] = (1 - alpha) * c["centroid"][0] + alpha * L
        c["centroid"][1] = (1 - alpha) * c["centroid"][1] + alpha * a
        c["centroid"][2] = (1 - alpha) * c["centroid"][2] + alpha * b
        c["samples"] = n + 1

# ========================= UI helpers =========================
def put_panel(img, lines, top_left=(10, 10), pad=8, alpha=0.6, color=(0, 255, 0)):
    x, y = top_left
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    line_gap = 6

    text_sizes = [cv2.getTextSize(ln, font, scale, thick)[0] for ln in lines]
    if len(text_sizes) == 0:
        return

    line_h = max(sz[1] for sz in text_sizes) + line_gap
    total_h = line_h * len(lines)
    panel_w = max(sz[0] for sz in text_sizes)

    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - pad), (x + panel_w + pad, y + total_h + pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    y_text = y + line_h - line_gap // 2
    for ln in lines:
        cv2.putText(img, ln, (x, y_text), font, scale, color, 1, cv2.LINE_AA)
        y_text += line_h

def put_banner(img, text, color=(0, 255, 255)):
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 10
    x = (W - tw) // 2
    y = 15
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y), (x + tw + pad, y + th + pad * 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x, y + th + pad // 2), font, scale, color, thick, cv2.LINE_AA)

def draw_label(img, text, org, color=(0, 255, 0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# ========================= Settings panel (in-window) =========================
class SettingsUI:
    def __init__(self):
        self.fields = [
            ("d_ab", "float", 1.0),
            ("dL", "float", 1.0),
            ("use_edges", "bool", None),
            ("update_bg", "bool", None),
            ("min_area", "int", 100),
            ("deltaE_thresh", "float", 1.0),
            ("hide_unlabeled", "bool", None),
            ("palette_path", "str", None),
            ("camera_index", "int", 1),
            ("width", "int", 16),
            ("height", "int", 16)
        ]
        self.idx = 0
        self.editing_text = False
        self.text_buf = ""

    def show(self, img):
        lines = ["SETTINGS (o to close, arrows navigate, ←/→ change, Enter edit/apply)"]
        for i, (k, typ, step) in enumerate(self.fields):
            val = SETTINGS[k]
            prefix = "→ " if i == self.idx else "  "
            if self.editing_text and i == self.idx and typ == "str":
                val_str = f"{self.text_buf}▌"
            else:
                val_str = str(val)
            lines.append(f"{prefix}{k}: {val_str}")
        put_panel(img, lines, top_left=(10, 110))

    def handle_key(self, k, cap_ref):
        kchar = chr(k) if 32 <= k < 127 else None
        name, typ, step = self.fields[self.idx]

        if self.editing_text:
            if k == 27:
                self.editing_text = False
                self.text_buf = ""
                return
            if k in (13, 10):
                if self.text_buf != "":
                    SETTINGS[name] = self.text_buf
                self.editing_text = False
                self.text_buf = ""
                return
            if k in (8, 127):
                self.text_buf = self.text_buf[:-1]
                return
            if kchar:
                self.text_buf += kchar
            return

        if k == ord('o'):
            return

        if k in (82, ord('k')):
            self.idx = (self.idx - 1) % len(self.fields)
            return

        if k in (84, ord('j')):
            self.idx = (self.idx + 1) % len(self.fields)
            return

        if k in (81, ord('h')):
            self._bump(-1, cap_ref)
            return

        if k in (83, ord('l')):
            self._bump(+1, cap_ref)
            return

        if k in (13, 10):
            if typ == "bool":
                SETTINGS[name] = not SETTINGS[name]
                return
            if typ == "str":
                self.editing_text = True
                self.text_buf = str(SETTINGS[name])
                return

    def _bump(self, direction, cap_ref):
        name, typ, step = self.fields[self.idx]
        if typ == "bool":
            SETTINGS[name] = not SETTINGS[name]
            return
        if typ in ("int", "float"):
            delta = (step or 1) * direction
            if typ == "int":
                SETTINGS[name] = max(0, int(SETTINGS[name] + delta))
            else:
                SETTINGS[name] = float(SETTINGS[name] + delta)
            return
        # 'str' is edited via Enter

# ========================= Main =========================
def main():
    cap = cv2.VideoCapture(SETTINGS["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SETTINGS["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SETTINGS["height"])

    if not cap.isOpened():
        raise SystemExit("Could not open video source (change camera_index in Settings).")

    pal = Palette(SETTINGS["palette_path"] if os.path.exists(SETTINGS["palette_path"]) else None)
    if pal.path is None:
        pal.path = SETTINGS["palette_path"]

    bg = None
    t_last = time.time()
    fps_hist = deque(maxlen=30)

    ui_mode = "normal"
    input_name = ""
    input_key = ""
    settings_ui = SettingsUI()

    cv2.namedWindow("regrind", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if bg is None:
            H, W = frame.shape[:2]
            bg = PixelBG(H, W)

        mask, lab = segment(frame, bg)

        if bg.ready and SETTINGS["update_bg"]:
            bg.update(lab, mask, alpha=0.02)

        vis = frame.copy()

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        largest_i = None
        largest_area = 0
        labeled = 0
        unlabeled = 0

        for i in range(1, num):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < SETTINGS["min_area"]:
                continue

            if area > largest_area:
                largest_area = area
                largest_i = i

            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            cnt = max(contours, key=cv2.contourArea)

            lab_pixels = lab[labels == i].reshape(-1, 3)
            match = pal.classify_lab(lab_pixels)

            if match:
                name, dE, idx = match
                labeled += 1
                cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
                draw_label(vis, f"{name} (dE {dE:.1f})", (x, max(20, y - 8)), (0, 255, 0))
            else:
                unlabeled += 1
                if not SETTINGS["hide_unlabeled"]:
                    cv2.drawContours(vis, [cnt], -1, (0, 0, 255), 2)
                    draw_label(vis, "unlabeled", (x, max(20, y - 8)), (0, 0, 255))

        t_now = time.time()
        dt = max(1e-6, t_now - t_last)
        fps_hist.append(1.0 / dt)
        t_last = t_now
        fps = sum(fps_hist) / len(fps_hist)

        if ui_mode == "normal":
            lines = [
                f"FPS {fps:.1f}   Labeled:{labeled}  Unlabeled:{unlabeled}",
                "[b] capture BG   [u] adapt BG   [w] save palette   [o] settings   [q] quit",
                "[n] new class from largest   [key] add sample to that class",
                "Palette: " + pal.legend()
            ]
            put_panel(vis, lines, top_left=(10, 10))
        else:
            put_panel(vis, [f"FPS {fps:.1f}   Labeled:{labeled}  Unlabeled:{unlabeled}"], top_left=(10, 10))

        if not bg.ready:
            put_banner(vis, "BACKGROUND NOT SET — press [b] on a clean background", (0, 255, 255))

        if ui_mode == "name":
            put_panel(
                vis,
                [
                    "NEW CLASS: type name, Enter=confirm, Esc=cancel, Tab=also set key",
                    f"Name: {input_name}"
                ],
                top_left=(10, 110)
            )

        if ui_mode == "key":
            put_panel(
                vis,
                [
                    f"NEW CLASS '{input_name}': press ONE key to assign, Enter=skip, Esc=cancel",
                    f"Key: {input_key}"
                ],
                top_left=(10, 110)
            )

        if ui_mode == "settings":
            settings_ui.show(vis)
            n_lines = 1 + len(settings_ui.fields)
            put_panel(vis, ["Note: camera/size changes apply on restart."], top_left=(10, 110 + 24 * n_lines))

        cv2.imshow("regrind", vis)

        k = cv2.waitKey(1) & 0xFF

        if ui_mode == "normal":
            if k in (27, ord('q')):
                break

            if k == ord('b'):
                blur = cv2.GaussianBlur(frame, (5, 5), 0)
                lab0 = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
                bg.init_from(lab0)

            if k == ord('u'):
                SETTINGS["update_bg"] = not SETTINGS["update_bg"]

            if k == ord('w'):
                pal.save()
                print(f"Palette saved to {pal.path}")

            if k == ord('o'):
                ui_mode = "settings"

            if k == ord('n') and largest_i is not None:
                ui_mode = "name"
                input_name = ""
                input_key = ""

            if k != 255 and largest_i is not None:
                if 32 <= k < 127:
                    ch = chr(k)
                else:
                    ch = None
                if ch is not None:
                    idx = pal.key_to_index(ch)
                    if idx is not None:
                        pal.add_sample(idx, lab[labels == largest_i].reshape(-1, 3))

        elif ui_mode == "name":
            if k == 27:
                ui_mode = "normal"
                input_name = ""
                input_key = ""

            elif k in (13, 10):
                name_final = input_name if input_name != "" else f"class_{len(pal.colors) + 1}"
                pal.add_class_from_pixels(name_final, lab[labels == largest_i].reshape(-1, 3))
                ui_mode = "normal"
                input_name = ""
                input_key = ""

            elif k == 9:
                ui_mode = "key"

            elif k in (8, 127):
                input_name = input_name[:-1]

            elif 32 <= k < 127:
                input_name += chr(k)

        elif ui_mode == "key":
            if k == 27:
                ui_mode = "normal"
                input_name = ""
                input_key = ""

            elif k in (13, 10):
                name_final = input_name if input_name != "" else f"class_{len(pal.colors) + 1}"
                pal.add_class_from_pixels(name_final, lab[labels == largest_i].reshape(-1, 3), key=None)
                ui_mode = "normal"
                input_name = ""
                input_key = ""

            elif 32 <= k < 127:
                input_key = chr(k)
                name_final = input_name if input_name != "" else f"class_{len(pal.colors) + 1}"
                pal.add_class_from_pixels(name_final, lab[labels == largest_i].reshape(-1, 3), key=input_key[0])
                ui_mode = "normal"
                input_name = ""
                input_key = ""

        elif ui_mode == "settings":
            if k in (27, ord('o')):
                ui_mode = "normal"
            else:
                settings_ui.handle_key(k, cap)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()