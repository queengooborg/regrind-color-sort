import cv2
import numpy as np

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
def segment(frame_bgr, bg, settings):
	blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
	lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
	if not bg.ready:
		return np.zeros(frame_bgr.shape[:2], np.uint8), lab

	ab = lab[:, :, 1:3]
	ab0 = bg.mu_lab[:, :, 1:3]
	dab = np.linalg.norm(ab - ab0, axis=2)

	dL = lab[:, :, 0] - bg.mu_lab[:, :, 0]

	mask_chroma = (dab >= settings["d_ab"]).astype(np.uint8) * 255
	mask_L = ((np.abs(dL) >= settings["dL"]) & (dab < settings["d_ab"] * 0.7)).astype(np.uint8) * 255
	mask = cv2.bitwise_or(mask_chroma, mask_L)

	if settings["use_edges"]:
		gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 60, 180)
		edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
		mask = cv2.bitwise_or(mask, edges)

	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), 2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)

	return mask, lab