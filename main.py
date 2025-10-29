# Detect plastic regrind against background and classify color
# Mostly written by ChatGPT

import cv2

from threading import Thread, Lock

class FrameGrabber:
	def __init__(self, cap):
		self.cap = cap
		self.lock = Lock()
		self.frame = None
		self.stopped = False
		self.t = Thread(target=self._loop, daemon=True)

	def start(self):
		self.t.start()
		return self

	def _loop(self):
		while not self.stopped:
			ok, f = self.cap.read()
			if not ok:
				continue
			with self.lock:
				self.frame = f

	def read(self):
		with self.lock:
			return None if self.frame is None else self.frame.copy()

	def stop(self):
		self.stopped = True
import numpy as np
import time
import json
import os
from collections import deque

from lib.ui import *
from lib.palette import *
from lib.bg import *

# ========================= Defaults (changed via Settings panel) =========================
SETTINGS = {
	"d_ab": 12.0,
	"dL": 18.0,
	"use_edges": True,
	"min_area": 1500,
	"mah_thresh": 3.0,
	"min_margin": 0.4,
	"use_mahalanobis": True,
	"deltaE_thresh": 18.0,
	"hide_unlabeled": False,
	"camera_index": 0,
	"palette_path": "palette.json"
}

class SettingsIO:
	@staticmethod
	def load(path, defaults):
		import json, os
		if not os.path.exists(path):
			return defaults.copy()
		try:
			with open(path, "r", encoding="utf-8") as f:
				data = json.load(f)
		except Exception:
			return defaults.copy()
		out = defaults.copy()
		for k, v in data.items():
			if k in out:
				out[k] = v
		return out

	@staticmethod
	def save(path, settings):
		import json, os
		tmp = f"{path}.tmp"
		with open(tmp, "w", encoding="utf-8") as f:
			json.dump(settings, f, indent=2)
		os.replace(tmp, path)

# ========================= Settings panel (in-window) =========================
class SettingsUI:
	def __init__(self):
		self.fields = [
			("d_ab", "float", 1.0),
			("dL", "float", 1.0),
			("use_edges", "bool", None),
			("min_area", "int", 100),
			("mah_thresh", "float", 0.1),
			("min_margin", "float", 0.1),
			("use_mahalanobis", "bool", None),
			("deltaE_thresh", "float", 1.0),
			("hide_unlabeled", "bool", None),
			("palette_path", "str", None),
			("camera_index", "int", 1)
		]
		self.idx = 0
		self.editing_text = False
		self.text_buf = ""

	def show(self, img):
		lines = ["SETTINGS (o/esc to close, ↑/↓ navigate, ←/→ change, Enter edit/apply)"]
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

		# text edit mode
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

		# navigation keys
		is_up = k in (82, 63232, 2490368)
		is_down = k in (84, 63233, 2621440)
		is_left = k in (81, 63234, 2424832)
		is_right = k in (83, 63235, 2555904)

		if is_up:
			self.idx = (self.idx - 1) % len(self.fields)
			return
		if is_down:
			self.idx = (self.idx + 1) % len(self.fields)
			return
		if is_left:
			self._bump(-1, cap_ref)
			return
		if is_right:
			self._bump(+1, cap_ref)
			return

		# toggle / edit
		if k in (13, 10) or is_right:
			if typ == "bool":
				SETTINGS[name] = not SETTINGS[name]
				return
			if typ == "str":
				self.editing_text = True
				self.text_buf = str(SETTINGS[name])
				return

		if k in (27, ord('o')):
			return 'exit'

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
	global SETTINGS
	SETTINGS = SettingsIO.load("settings.json", SETTINGS)
	cap = cv2.VideoCapture(SETTINGS["camera_index"])
	grab = FrameGrabber(cap).start()

	if not cap.isOpened():
		raise SystemExit("Could not open video source (change camera_index in Settings).")

	pal = Palette(SETTINGS)

	bg = None
	t_last = time.time()
	fps_hist = deque(maxlen=30)

	ui_mode = "normal"
	input_name = ""
	input_key = ""
	settings_ui = SettingsUI()

	cv2.namedWindow("regrind", cv2.WINDOW_NORMAL)

	while True:
		frame = grab.read()
		if frame is None:
			continue

		if bg is None:
			H, W = frame.shape[:2]
			bg = PixelBG(H, W)

		mask, lab = segment(frame, bg, SETTINGS)

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
				name, score, idx = match
				labeled += 1
				cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
				metric = "dM" if SETTINGS.get("use_mahalanobis", True) else "dE"
				draw_label(vis, f"{name} ({metric} {score:.2f})", (x, y), (0, 255, 0), size=18)
			else:
				unlabeled += 1
				if not SETTINGS["hide_unlabeled"]:
					cv2.drawContours(vis, [cnt], -1, (0, 0, 255), 2)
					draw_label(vis, "unlabeled", (x, max(20, y - 8)), (0, 0, 255), size=18)
		t_now = time.time()
		dt = max(1e-6, t_now - t_last)
		fps_hist.append(1.0 / dt)
		t_last = t_now
		fps = sum(fps_hist) / len(fps_hist)

		pil_vis, draw = begin_text(vis)
		put_panel_ctx(pil_vis, draw, vis.shape, [f"FPS {fps:.1f}   Labeled:{labeled}  Unlabeled:{unlabeled}"], top_left=(10, 10), size=18)

		if ui_mode == "normal":
			put_panel_ctx(pil_vis, draw, vis.shape, [
				"[space] capture BG   [,] settings   [esc] quit",
				"[.] new class from largest   [key] add sample to that class"
			], top_left=(10, 50), size=18)

			put_panel_ctx(pil_vis, draw, vis.shape, [
				"Palette:",
				*pal.legend()
			], top_left=(10, 110), size=18)
		if not bg.ready and ui_mode != "exit":
			put_banner_ctx(pil_vis, draw, vis.shape, "BACKGROUND NOT SET — press [space] on a clean background", (0, 255, 255), size=20)
		if ui_mode == "name":
			put_panel_ctx(
				pil_vis, draw, vis.shape,
				[
					"NEW PALETTE COLOR",
					"[enter] confirm   [tab] edit keybind   [esc] cancel",
					f"Name: {input_name}▌",
					f"Key: {input_key or pal.auto_key(input_name)}"
				],
				top_left=(10, 110),
				size=18
			)
		if ui_mode == "key":
			put_panel_ctx(
				pil_vis, draw, vis.shape,
				[
					"NEW PALETTE COLOR",
					"press ONE key to assign   [enter] auto   [esc] cancel",
					f"Name: {input_name}",
					f"Key: {input_key}▌"
				],
				top_left=(10, 110),
				size=18
			)
		if ui_mode == "exit":
			put_banner_ctx(pil_vis, draw, vis.shape, "Save settings and palette? [y]/[n]", (50, 50, 255), size=20)

		if ui_mode == "settings":
			settings_ui.show(vis)
			n_lines = 1 + len(settings_ui.fields)
			put_panel_ctx(pil_vis, draw, vis.shape, ["Note: camera/size changes apply on restart."], top_left=(10, 110 + 24 * n_lines), size=18)
		vis = end_text(pil_vis)
		cv2.imshow("regrind", vis)

		k = cv2.waitKeyEx(1)

		if ui_mode == "normal":
			if k == 27:
				ui_mode = "exit"

			if k == ord(' '):
				blur = cv2.GaussianBlur(frame, (5, 5), 0)
				lab0 = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
				bg.init_from(lab0)

			if k == ord(','):
				ui_mode = "settings"

			if k == ord('.') and largest_i is not None:
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
			if settings_ui.handle_key(k, cap) == 'exit':
				pal.settings = SETTINGS
				ui_mode = "normal"

		elif ui_mode == "exit":
			if k in (ord('y'), ord('n')):
				if k == ord('y'):
					try:
						SettingsIO.save("settings.json", SETTINGS)
						pal.save()
					except Exception:
						pass

				try:
					grab.stop()
				except Exception:
					pass
				cap.release()
				cv2.destroyAllWindows()
				break


if __name__ == "__main__":
	main()