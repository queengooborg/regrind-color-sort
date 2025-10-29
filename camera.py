# Detect plastic regrind against background
# Mostly written by ChatGPT

import cv2, numpy as np, argparse, time
from collections import deque

def bgr_to_hex(mean_bgr):
    b,g,r = np.clip(mean_bgr,0,255).astype(int); return f"#{r:02X}{g:02X}{b:02X}"

def hue_to_name(h):
    if h < 5 or h >= 175: return "red"
    if h < 15: return "red-orange"
    if h < 25: return "orange"
    if h < 35: return "yellow"
    if h < 45: return "yellow-green"
    if h < 70: return "green"
    if h < 85: return "teal"
    if h < 100: return "cyan"
    if h < 125: return "blue"
    if h < 140: return "indigo"
    if h < 160: return "purple"
    return "magenta"

def classify_color(bgr_pixels, hsv_pixels, sat_cut=25, v_black=60, v_white=200):
    if len(bgr_pixels) == 0: return {"name":"unknown"}
    v = hsv_pixels[:,2]
    keep = (v > 10) & (v < 245)
    if keep.sum() < 50: keep = np.ones_like(v, bool)
    bgr = bgr_pixels[keep]; hsv = hsv_pixels[keep]
    sat = float(np.median(hsv[:,1])); val = float(np.median(hsv[:,2]))
    hexcode = bgr_to_hex(np.median(bgr, axis=0))
    if sat < sat_cut:
        name = "black" if val < v_black else ("white" if val > v_white else "gray")
        return {"name": name, "hex": hexcode, "saturation": sat, "value": val}
    h_med = float(np.median(hsv[:,0]))
    return {"name": hue_to_name(h_med), "hex": hexcode, "hue": h_med, "saturation": sat, "value": val}

class PixelBG:
    """Per-pixel background model in Lab; tracks mean a*,b* (+ L* for shadow gating)."""
    def __init__(self, H, W):
        self.mu_lab = np.zeros((H,W,3), np.float32)
        self.var_ab = np.ones((H,W,2), np.float32) * 9.0  # variance for a*,b*
        self.ready = False
    def init_from(self, lab):
        self.mu_lab[:] = lab
        self.var_ab[:] = 9.0
        self.ready = True
    def update(self, lab, fg_mask, alpha=0.02):
        if not self.ready: return
        bg = (fg_mask == 0)
        if not np.any(bg): return
        bg3 = bg[:,:,None]
        # update mean all 3 channels; variance only for a,b
        self.mu_lab[bg3] = (1-alpha)*self.mu_lab[bg3] + alpha*lab[bg3]
        delta_ab = lab[:,:,[1,2]] - self.mu_lab[:,:,[1,2]]
        self.var_ab[bg] = (1-alpha)*self.var_ab[bg] + alpha*(delta_ab[bg]**2)

def segment(frame_bgr, bg: PixelBG, d_ab_thresh=12.0, dL_thresh=18.0, use_edges=True):
    blur = cv2.GaussianBlur(frame_bgr, (5,5), 0)
    lab  = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    if not bg.ready:
        return np.zeros(frame_bgr.shape[:2], np.uint8), lab

    # Chroma distance (shadow-robust): ||(a,b) - (a0,b0)||
    ab   = lab[:,:,1:3]; ab0 = bg.mu_lab[:,:,1:3]
    dab  = np.linalg.norm(ab - ab0, axis=2)

    # Ignore shadows: small chroma change but big negative L
    dL = lab[:,:,0] - bg.mu_lab[:,:,0]
    mask_chroma = (dab >= d_ab_thresh).astype(np.uint8)*255

    # Add lightness cue for achromatic objects: if |dL| large while chroma small-ish
    mask_L = ((np.abs(dL) >= dL_thresh) & (dab < d_ab_thresh*0.7)).astype(np.uint8)*255

    mask = cv2.bitwise_or(mask_chroma, mask_L)

    if use_edges:
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)
        mask = cv2.bitwise_or(mask, edges)

    # Clean
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), 1)
    return mask, lab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--d_ab", type=float, default=12.0, help="Chroma (a*,b*) threshold")
    ap.add_argument("--dL", type=float, default=18.0, help="Lightness threshold")
    ap.add_argument("--update_bg", action="store_true", help="Adapt background over time")
    ap.add_argument("--min_area", type=int, default=1500)
    ap.add_argument("--no_edge", action="store_true")
    args = ap.parse_args()

    src = int(args.src) if args.src.isdigit() else args.src
    cap = cv2.VideoCapture(src)
    if args.width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Could not open video source.")

    bg = None
    t_last = time.time(); fps_hist = deque(maxlen=30)
    print("[q] quit   [b] capture background   [u] toggle adapt BG   [s] save frames")

    while True:
        ok, frame = cap.read()
        if not ok: break

        if bg is None:
            H,W = frame.shape[:2]; bg = PixelBG(H,W)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break
        if key == ord('b'):
            lab0 = cv2.cvtColor(cv2.GaussianBlur(frame,(5,5),0), cv2.COLOR_BGR2LAB).astype(np.float32)
            bg.init_from(lab0); print("Background captured.")
        if key == ord('u'):
            args.update_bg = not args.update_bg; print(f"Adaptive BG update: {args.update_bg}")
        if key == ord('s'):
            ts=int(time.time())
            try:
                cv2.imwrite(f"annotated_{ts}.png", vis); cv2.imwrite(f"mask_{ts}.png", mask)
                print(f"Saved annotated_{ts}.png and mask_{ts}.png")
            except NameError:
                print("Nothing to save yet.")

        mask, lab = segment(frame, bg, args.d_ab, args.dL, use_edges=not args.no_edge)

        if bg.ready and args.update_bg:
            bg.update(lab, mask, alpha=0.02)

        # Components + color names
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        vis = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        count = 0
        for i in range(1, num):
            x,y,w,h,area = stats[i]
            if area < args.min_area: continue
            comp = (labels == i)
            color = classify_color(frame[comp], hsv[comp])
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(vis, f"{color['name']} {color.get('hex','')}", (x, max(10,y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            count += 1

        # HUD
        t_now = time.time()
        fps_hist.append(1.0/max(1e-6, t_now - t_last)); t_last = t_now
        fps = sum(fps_hist)/len(fps_hist)
        cv2.putText(vis, f"FPS: {fps:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(vis, f"Objects: {count}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("annotated", vis)
        cv2.imshow("mask", mask)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()