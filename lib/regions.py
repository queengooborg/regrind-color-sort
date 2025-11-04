# regrind-color-sort - lib/regions.py
# 
# This program uses OpenCV to identify plastic regrind against a background and classify it by colors as specified by the user
# (c) 2025 ChatGPT, Vinyl Da.i'gyu-Kazotetsu

import cv2
import numpy as np

from .ui import draw_label

class PixelBG:
	def __init__(self, H, W):
		self.mu_lab = np.zeros((H, W, 3), np.float32)
		self.var_ab = np.ones((H, W, 2), np.float32) * 9.0
		self.ready = False

	def init_from(self, frame):
		blur = cv2.GaussianBlur(frame, (5, 5), 0)
		lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
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

def find_regions(frame, bg, vis, pal, settings):
	mask, lab = segment(frame, bg, settings)

	num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
	largest = None
	labeled = 0
	unlabeled = 0

	for i in range(1, num):
		x = int(stats[i, cv2.CC_STAT_LEFT])
		y = int(stats[i, cv2.CC_STAT_TOP])
		w = int(stats[i, cv2.CC_STAT_WIDTH])
		h = int(stats[i, cv2.CC_STAT_HEIGHT])
		area = int(stats[i, cv2.CC_STAT_AREA])

		if area < settings["min_area"]:
			continue

		cnt = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)
		if not settings['bounding_boxes']:
			comp_mask = (labels == i).astype(np.uint8) * 255
			contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			if contours:
				cnt = max(contours, key=cv2.contourArea)

		lab_pixels = lab[labels == i].reshape(-1, 3)
		match = pal.classify_lab(lab_pixels)

		if not largest or area > largest["area"]:
			largest = {
				"area": area,
				"i": i,
				"match": match,
				"cnt": cnt
			}

		if match:
			name, score, idx = match
			labeled += 1
			draw_label(vis, f"{name} (dE {score:.2f})", (x, y), (0, 255, 0))
		else:
			unlabeled += 1
			draw_label(vis, "unlabeled", (x, max(20, y - 8)), (0, 0, 255))

		cv2.drawContours(vis, [cnt], -1, (0, 255, 0) if match else (0, 0, 255), 2)

	# Highlight largest area
	if largest:
		cv2.drawContours(vis, [largest["cnt"]], -1, (0, 255, 0) if largest["match"] else (0, 0, 255), 8)

	return [labeled, unlabeled, largest, mask, lab, labels]
