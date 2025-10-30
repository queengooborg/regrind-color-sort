# Detect plastic regrind against background and classify color
# Mostly written by ChatGPT

import cv2
import numpy as np
import time
import json
import os
from collections import deque
from cv2_enumerate_cameras import enumerate_cameras

from lib.ui import *
from lib.palette import *
from lib.bg import *
from lib.framegrabber import *

# ========================= Settings =========================
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
	"camera": 0,
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
			("camera", "int", 1)
		]
		self.idx = 0
		self.editing_text = False
		self.text_buf = ""
		self.cameras = enumerate_cameras()

	def show(self, img):
		lines = [
			"SETTINGS   [esc] to close",
			"[up]/[down] navigate   [left]/[right] change   [enter] toggle/edit)"
		]
		for i, (k, typ, step) in enumerate(self.fields):
			val = SETTINGS[k]
			prefix = "> " if i == self.idx else "  "
			if k == "camera":
				val_str = self.cameras[val].name
			elif self.editing_text and i == self.idx and typ == "str":
				val_str = f"{self.text_buf}_"
			else:
				val_str = str(val)
			lines.append(f"{prefix}{k}: {val_str}")
		put_panel(img, lines, pos=(10, 110))

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
		if is_right or k in (13, 10):
			if typ == "bool":
				SETTINGS[name] = not SETTINGS[name]
				return
			if typ == "str":
				self.editing_text = True
				self.text_buf = str(SETTINGS[name])
				return

		if k == 27:
			return 'exit'

	def _bump(self, direction, cap_ref):
		name, typ, step = self.fields[self.idx]
		if typ == "bool":
			SETTINGS[name] = not SETTINGS[name]
			return
		if name == "camera":
			print(name, typ, step)
			delta = (step or 1) * direction
			val = SETTINGS[name] + delta
			if val < 0:
				val = len(self.cameras) - 1
			if val >= len(self.cameras):
				val = 0
			SETTINGS[name] = val
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
	pal = Palette(SETTINGS)

	cap = cv2.VideoCapture(SETTINGS["camera"])
	if not cap.isOpened():
		print("Could not load specified camera, switching to default...")
		cap = cv2.VideoCapture(0)
		if not cap.isOpened():
			raise SystemExit("Could not load default camera, please check if a camera is connected and if it is in use by other programs")
		SETTINGS["camera"] = 0

	grab = FrameGrabber(cap).start()

	bg = None
	t_last = time.time()
	fps_hist = deque(maxlen=30)

	ui_mode = "normal"
	input_name = ""
	input_key = ""
	settings_ui = SettingsUI()

	cv2.namedWindow("Filament Regrind Color Classifier", cv2.WINDOW_NORMAL)

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
		largest = None
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

			comp_mask = (labels == i).astype(np.uint8) * 255
			contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			if not contours:
				continue

			cnt = max(contours, key=cv2.contourArea)

			lab_pixels = lab[labels == i].reshape(-1, 3)
			match = pal.classify_lab(lab_pixels)

			if not largest or area > largest["area"]:
				largest = {
					"area": area,
					"i": i,
					"cnt": cnt,
					"match": match
				}

			if match:
				name, score, idx = match
				labeled += 1
				cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
				metric = "dM" if SETTINGS.get("use_mahalanobis", True) else "dE"
				draw_label(vis, f"{name} ({metric} {score:.2f})", (x, y), (0, 255, 0))
			else:
				unlabeled += 1
				if not SETTINGS["hide_unlabeled"]:
					cv2.drawContours(vis, [cnt], -1, (0, 0, 255), 2)
					draw_label(vis, "unlabeled", (x, max(20, y - 8)), (0, 0, 255))

		# Highlight largest area
		if largest:
			cv2.drawContours(vis, [largest["cnt"]], -1, (0, 255, 0) if largest["match"] else (0, 0, 255), 8)

		t_now = time.time()
		dt = max(1e-6, t_now - t_last)
		fps_hist.append(1.0 / dt)
		t_last = t_now
		fps = sum(fps_hist) / len(fps_hist)

		put_panel(vis, [f"FPS {fps:.1f}   Labeled:{labeled}  Unlabeled:{unlabeled}   Total:{labeled+unlabeled}"], pos=(10, 10))

		if ui_mode == "normal":
			put_panel(vis, [
				"[space] capture BG   [,] settings   [esc] quit",
				"[.] new class from largest   [key] add sample to that class"
			], pos=(10, 50))

			put_panel(vis, [
				"Palette:",
				*pal.legend()
			], pos=(10, 110))
		if not bg.ready and ui_mode != "exit":
			put_banner(vis, "BACKGROUND NOT SET   Press [space] on a clean background", (0, 255, 255))
		if ui_mode in ["name", "key"]:
			put_panel(
				vis,
				[
					"NEW PALETTE COLOR",
					"[enter] confirm   [tab] switch field   [esc] cancel" + ("   [backspace] auto key" if ui_mode == "key" else ""),
					f"Name: {input_name}{'_' if ui_mode == "name" else ''}",
					f"Key: {input_key or pal.auto_key(input_name)}{'_' if ui_mode == "key" else ''}"
				],
				pos=(10, 110)
			)
		if ui_mode == "exit":
			put_banner(vis, "Save settings and palette? [y]/[n]", (50, 50, 255))

		if ui_mode == "settings":
			settings_ui.show(vis)
			n_lines = 1 + len(settings_ui.fields)
			put_panel(vis, ["Note: camera/size changes apply on restart."], pos=(10, 110 + 24 * n_lines))
		cv2.imshow("regrind", vis)

		# Handle keyboard inputs
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

			if k == ord('.') and largest is not None:
				ui_mode = "name"
				input_name = ""
				input_key = ""

			if k != 255 and largest is not None:
				if 32 <= k < 127:
					ch = chr(k)
				else:
					ch = None
				if ch is not None:
					idx = pal.key_to_index(ch)
					if idx is not None:
						pal.add_sample(idx, lab[labels == largest.get("i")].reshape(-1, 3))

		elif ui_mode in ["name", "key"]:
			if k == 27:
				ui_mode = "normal"
				input_name = ""
				input_key = ""

			elif k in (13, 10):
				name_final = input_name if input_name != "" else f"class_{len(pal.colors) + 1}"
				pal.add_class_from_pixels(name_final, lab[labels == largest.get("i")].reshape(-1, 3))
				ui_mode = "normal"
				input_name = ""
				input_key = ""

			elif k == 9:
				ui_mode = "key" if ui_mode == "name" else "name"

			else:
				if ui_mode == "name":
					if k in (8, 127):
						input_name = input_name[:-1]

					elif 32 <= k < 127:
						input_name += chr(k)
				else:
					if k in (8, 127):
						input_key = ""
					elif 32 <= k < 127:
						input_key = chr(k)

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
