# regrind-color-sort - lib/ui.py
# 
# This program uses OpenCV to identify plastic regrind against a background and classify it by colors as specified by the user
# (c) 2025 ChatGPT, Vinyl Da.i'gyu-Kazotetsu

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ========================= UI helpers =========================
def put_panel(img, lines, pos=(10, 10), pad=8, alpha=0.6, color=(0, 255, 0)):
	x, y = pos
	scale = 0.55
	thick = 1
	line_gap = 6

	text_sizes = [cv2.getTextSize(ln, FONT, scale, thick)[0] for ln in lines]
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
		cv2.putText(img, ln, (x, y_text), FONT, scale, color, 1, cv2.LINE_AA)
		y_text += line_h

def put_banner(img, text, color=(0, 255, 255)):
	H, W = img.shape[:2]
	scale = 0.7
	thick = 2
	(tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
	pad = 10
	x = (W - tw) // 2
	y = H - 45
	overlay = img.copy()
	cv2.rectangle(overlay, (x - pad, y), (x + tw + pad, y + th + pad * 2), (0, 0, 0), -1)
	cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
	cv2.putText(img, text, (x, y + th + pad // 2), FONT, scale, color, thick, cv2.LINE_AA)

def draw_label(img, text, org, color=(0, 255, 0)):
	cv2.putText(img, text, org, FONT, 0.6, color, 2, cv2.LINE_AA)
