# Detect plastic regrind against background and classify color
# Mostly written by ChatGPT

import cv2, numpy as np, argparse, time, json, os
from collections import deque

# ---------------- Per-pixel Lab BG model ----------------
class PixelBG:
    def __init__(self, H, W):
        self.mu_lab = np.zeros((H,W,3), np.float32)
        self.var_ab = np.ones((H,W,2), np.float32) * 9.0
        self.ready = False
    def init_from(self, lab):
        self.mu_lab[:] = lab; self.var_ab[:] = 9.0; self.ready = True
    def update(self, lab, fg_mask, alpha=0.02):
        if not self.ready: return
        bg = (fg_mask == 0)
        if not np.any(bg): return
        bg3 = bg[:,:,None]
        self.mu_lab[bg3] = (1-alpha)*self.mu_lab[bg3] + alpha*lab[bg3]
        delta_ab = lab[:,:,[1,2]] - self.mu_lab[:,:,[1,2]]
        self.var_ab[bg] = (1-alpha)*self.var_ab[bg] + alpha*(delta_ab[bg]**2)

# ---------------- Segmentation (shadow-robust) ----------------
def segment(frame_bgr, bg: PixelBG, d_ab_thresh=12.0, dL_thresh=18.0, use_edges=True):
    blur = cv2.GaussianBlur(frame_bgr, (5,5), 0)
    lab  = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    if not bg.ready:
        return np.zeros(frame_bgr.shape[:2], np.uint8), lab
    ab   = lab[:,:,1:3]; ab0 = bg.mu_lab[:,:,1:3]
    dab  = np.linalg.norm(ab - ab0, axis=2)
    dL   = lab[:,:,0] - bg.mu_lab[:,:,0]
    mask_chroma = (dab >= d_ab_thresh).astype(np.uint8)*255
    mask_L = ((np.abs(dL) >= dL_thresh) & (dab < d_ab_thresh*0.7)).astype(np.uint8)*255
    mask = cv2.bitwise_or(mask_chroma, mask_L)
    if use_edges:
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)
        mask = cv2.bitwise_or(mask, edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), 1)
    return mask, lab

# ---------------- Palette manager (empty by default) ----------------
KEY_POOL = list("1234567890-=[];',.")

class Palette:
    def __init__(self, path=None):
        self.path = path
        self.meta = {"deltaE_thresh": 18}
        self.colors = []  # {"name":str, "key":str|None, "centroid":[L,a,b], "samples":int}
        if path and os.path.exists(path):
            self.load(path)
    def load(self, path):
        with open(path,"r") as f: data = json.load(f)
        self.meta = data.get("meta", self.meta)
        self.colors = data.get("colors", [])
        self.path = path
    def save(self, path=None):
        if path is None: path = self.path or "palette.json"
        with open(path,"w") as f:
            json.dump({"version":1,"meta":self.meta,"colors":self.colors}, f, indent=2)
        self.path = path
    def legend(self):
        return "(empty)" if not self.colors else ", ".join([f"{c.get('key','•')}:{c['name']}" if c.get('key') else f"•:{c['name']}" for c in self.colors])
    def key_to_index(self, k):
        for i,c in enumerate(self.colors):
            if c.get("key") == k: return i
        return None
    def auto_key(self):
        used = {c.get("key") for c in self.colors if c.get("key")}
        for k in KEY_POOL:
            if k not in used: return k
        return None
    @staticmethod
    def _deltaE76(c1, c2):
        c1 = np.array(c1, np.float32); c2 = np.array(c2, np.float32)
        return float(np.linalg.norm(c1 - c2))
    def classify_lab(self, lab_pixels):
        if len(self.colors) == 0 or len(lab_pixels) == 0: return None
        L = float(np.median(lab_pixels[:,0])); a = float(np.median(lab_pixels[:,1])); b = float(np.median(lab_pixels[:,2]))
        best = (None, 1e9, None)
        for i,c in enumerate(self.colors):
            cent = c.get("centroid")
            if not cent: continue
            d = self._deltaE76([L,a,b], cent)
            if d < best[1]: best = (c["name"], d, i)
        if best[2] is None: return None
        if best[1] <= float(self.meta.get("deltaE_thresh", 18)): return best
        return None
    def add_class_from_pixels(self, name, lab_pixels, key=None):
        if len(lab_pixels)==0: return None
        L = float(np.median(lab_pixels[:,0])); a = float(np.median(lab_pixels[:,1])); b = float(np.median(lab_pixels[:,2]))
        if key is None or len(key)==0: key = self.auto_key()
        entry = {"name":name, "key": (key if key else None), "centroid":[L,a,b], "samples":1}
        self.colors.append(entry)
        return entry
    def add_sample(self, idx, lab_pixels):
        if idx is None or idx < 0 or idx >= len(self.colors) or len(lab_pixels)==0: return
        L = float(np.median(lab_pixels[:,0])); a = float(np.median(lab_pixels[:,1])); b = float(np.median(lab_pixels[:,2]))
        c = self.colors[idx]
        if not c.get("centroid"):
            c["centroid"] = [L,a,b]; c["samples"] = 1; return
        n = int(c.get("samples",1))
        alpha = 1.0 / (n + 1)  # EMA
        c["centroid"][0] = (1-alpha)*c["centroid"][0] + alpha*L
        c["centroid"][1] = (1-alpha)*c["centroid"][1] + alpha*a
        c["centroid"][2] = (1-alpha)*c["centroid"][2] + alpha*b
        c["samples"] = n + 1

# ---------------- Overlay UI helpers ----------------
def put_panel(img, lines, top_left=(10,10), pad=8, alpha=0.6):
    """Draw a semi-transparent text panel with correctly centered text."""
    x, y = top_left
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    line_gap = 6  # space between lines

    # Compute panel size using text ascent + descent
    text_sizes = [cv2.getTextSize(ln, font, scale, thick)[0] for ln in lines]
    line_height = max(sz[1] for sz in text_sizes) + line_gap
    total_height = line_height * len(lines)
    panel_width = max(sz[0] for sz in text_sizes)

    # Draw semi-transparent background
    panel = img.copy()
    cv2.rectangle(panel,
                  (x - pad, y - pad),
                  (x + panel_width + pad, y + total_height + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(panel, alpha, img, 1 - alpha, 0, img)

    # Render each line with corrected baseline offset
    y_text = y + line_height - line_gap // 2
    for ln in lines:
        (tw, th), baseline = cv2.getTextSize(ln, font, scale, thick)
        cv2.putText(img, ln, (x, y_text), font, scale, (0,255,0), thick, cv2.LINE_AA)
        y_text += line_height

def draw_label(img, text, org, color=(0,255,0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--d_ab", type=float, default=12.0)
    ap.add_argument("--dL", type=float, default=18.0)
    ap.add_argument("--update_bg", action="store_true")
    ap.add_argument("--min_area", type=int, default=1500)
    ap.add_argument("--no_edge", action="store_true")
    ap.add_argument("--palette", type=str, default="palette.json")
    ap.add_argument("--hide_unlabeled", action="store_true")
    args = ap.parse_args()

    # Camera
    src = int(args.src) if args.src.isdigit() else args.src
    cap = cv2.VideoCapture(src)
    if args.width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened(): raise SystemExit("Could not open video source.")

    pal = Palette(args.palette if os.path.exists(args.palette) else None)
    if pal.path is None: pal.path = args.palette

    bg = None
    t_last = time.time(); fps_hist = deque(maxlen=30)

    # Simple state machine for in-window text input
    ui_mode = "normal"   # "normal" | "name" | "key"
    input_name = ""
    input_key = ""

    cv2.namedWindow("regrind", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok: break
        if bg is None:
            H,W = frame.shape[:2]; bg = PixelBG(H,W)

        # Segmentation
        mask, lab = segment(frame, bg, args.d_ab, args.dL, use_edges=not args.no_edge)
        if bg.ready and args.update_bg: bg.update(lab, mask, alpha=0.02)

        vis = frame.copy()

        # Components
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        largest_i, largest_area = None, 0
        labeled = 0; unlabeled = 0

        # Draw contours for each kept component
        for i in range(1, num):
            x,y,w,h,area = stats[i]
            if area < args.min_area: continue
            if area > largest_area:
                largest_area = area; largest_i = i

            comp_mask = (labels == i).astype(np.uint8) * 255
            # Contour: get the external contour of this component
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            # Merge/choose largest contour
            cnt = max(cnts, key=cv2.contourArea)

            # Palette-only classify
            lab_pixels = lab[labels == i].reshape(-1,3)
            match = pal.classify_lab(lab_pixels)

            if match:
                name, dE, idx = match
                labeled += 1
                cv2.drawContours(vis, [cnt], -1, (0,255,0), 2)
                # label near contour bbox top-left
                draw_label(vis, f"{name} (dE {dE:.1f})", (x, max(20, y-8)), (0,255,0))
            else:
                unlabeled += 1
                if not args.hide_unlabeled:
                    cv2.drawContours(vis, [cnt], -1, (0,0,255), 2)
                    draw_label(vis, "unlabeled", (x, max(20, y-8)), (0,0,255))

        # HUD
        t_now = time.time()
        fps_hist.append(1.0/max(1e-6, t_now - t_last)); t_last = t_now
        fps = sum(fps_hist)/len(fps_hist)

        put_panel(vis, [
            f"FPS {fps:.1f}   Labeled:{labeled}  Unlabeled:{unlabeled}",
            "[b] capture BG   [u] adapt BG   [w] save palette   [p] print palette   [q] quit",
            "[n] new class from largest   [key] add sample to that class",
            "Palette: " + pal.legend()
        ], top_left=(10,10))

        # In-window UI for name/key entry
        if ui_mode == "name":
            put_panel(vis, [
                "NEW CLASS: type name, Enter=confirm, Esc=cancel",
                f"Name: {input_name}",
                "Press Tab to also set a key after name"
            ], top_left=(10,110))
        elif ui_mode == "key":
            put_panel(vis, [
                f"NEW CLASS '{input_name}': press ONE key to assign, Enter=skip, Esc=cancel",
                f"Key: {input_key}"
            ], top_left=(10,110))

        cv2.imshow("regrind", vis)

        # Key handling
        k = cv2.waitKey(1) & 0xFF
        if ui_mode == "normal":
            if k in (27, ord('q')): break
            elif k == ord('b'):
                lab0 = cv2.cvtColor(cv2.GaussianBlur(frame,(5,5),0), cv2.COLOR_BGR2LAB).astype(np.float32)
                bg.init_from(lab0)
            elif k == ord('u'):
                args.update_bg = not args.update_bg
            elif k == ord('w'):
                pal.save()
            elif k == ord('p'):
                print("Palette:", pal.legend())
                for i,c in enumerate(pal.colors):
                    print(f"  {i}: key={c.get('key')} name={c['name']} centroid={c.get('centroid')} samples={c.get('samples',0)}")
            elif k == ord('n') and largest_i is not None:
                ui_mode = "name"; input_name = ""; input_key = ""
            elif k != 255 and largest_i is not None:
                # Try add-sample by class key
                ch = chr(k) if 32 <= k < 127 else None
                if ch:
                    idx = pal.key_to_index(ch)
                    if idx is not None:
                        pal.add_sample(idx, lab[labels == largest_i].reshape(-1,3))
        elif ui_mode == "name":
            if k in (27,):  # Esc
                ui_mode = "normal"; input_name = ""; input_key = ""
            elif k in (13, 10):  # Enter → create class (no key) OR go to key mode if Tab was used
                # create now (no key)
                entry = pal.add_class_from_pixels(input_name or f"class_{len(pal.colors)+1}",
                                                  lab[labels == largest_i].reshape(-1,3))
                ui_mode = "normal"; input_name = ""; input_key = ""
            elif k == 9:  # Tab → proceed to key entry
                ui_mode = "key"
            elif k in (8, 127):  # Backspace/Delete
                input_name = input_name[:-1]
            elif 32 <= k < 127:
                input_name += chr(k)
        elif ui_mode == "key":
            if k in (27,):  # Esc cancel
                ui_mode = "normal"; input_name = ""; input_key = ""
            elif k in (13, 10):  # Enter skip key, create with auto key
                entry = pal.add_class_from_pixels(input_name or f"class_{len(pal.colors)+1}",
                                                  lab[labels == largest_i].reshape(-1,3),
                                                  key=None)
                ui_mode = "normal"; input_name = ""; input_key = ""
            elif 32 <= k < 127:
                input_key = chr(k)
                entry = pal.add_class_from_pixels(input_name or f"class_{len(pal.colors)+1}",
                                                  lab[labels == largest_i].reshape(-1,3),
                                                  key=input_key[0])
                ui_mode = "normal"; input_name = ""; input_key = ""

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()