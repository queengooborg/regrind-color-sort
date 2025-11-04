# regrind-color-sort - main.py
# 
# This program uses OpenCV to identify plastic regrind against a background and classify it by colors as specified by the user
# (c) 2025 ChatGPT, Vinyl Da.i'gyu-Kazotetsu

import cv2
import numpy as np
import time
import sys
from collections import deque

try:
	from picamera2 import Picamera2
	from libcamera import controls
except ImportError:
	Picamera2 = None

from lib import *

assert sys.version_info >= (3, 11), "Requires Python 3.11 or newer"

def main():
	settings = Settings()
	pal = Palette(settings)

	if Picamera2:
		cap = Picamera2()
		cap.configure(cap.create_video_configuration(
			main={"format": "RGB888", "size": (1280, 720)}
		))
		cap.set_controls({"AfMode": controls.AfModeEnum.Continuous})
		cap.start()
	else:
		cap = cv2.VideoCapture(settings["camera"])
		if not cap.isOpened():
			print("Could not load specified camera, switching to default...")
			cap = cv2.VideoCapture(0)
			if not cap.isOpened():
				raise SystemExit("Could not load default camera, please check if a camera is connected and if it is in use by other programs")
			settings["camera"] = 0

	grab = FrameGrabber(cap, bool(Picamera2)).start()

	bg = None
	t_last = time.time()
	fps_hist = deque(maxlen=30)

	ui_mode = "normal"
	input_name = ""
	input_key = ""
	settings_ui = SettingsUI(settings)

	cv2.namedWindow("Filament Regrind Color Classifier", cv2.WINDOW_NORMAL)

	while True:
		frame = grab.read()
		if frame is None:
			continue

		if bg is None:
			H, W = frame.shape[:2]
			bg = PixelBG(H, W)

		vis = frame.copy()

		labeled, unlabeled, largest, mask, lab, labels = find_regions(frame, bg, vis, pal, settings)

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
			put_banner(vis, "Save settings and palette? [y]es/[n]o/[s]ettings only/[p]alette only", (50, 50, 255))

		if ui_mode == "settings":
			settings_ui.show(vis)
		cv2.imshow("regrind", vis)

		# Handle keyboard inputs
		k = cv2.waitKeyEx(1)

		if ui_mode == "normal":
			if k == 27:
				ui_mode = "exit"

			if k == ord(' '):
				bg.init_from(frame)

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
				pal.settings = settings
				ui_mode = "normal"

		elif ui_mode == "exit":
			if k in (ord('y'), ord('n'), ord('s'), ord('p')):
				if k in (ord('y'), ord('s')):
					try:
						settings.save()
					except Exception:
						pass

				if k in (ord('y'), ord('p')):
					try:
						pal.save()
					except Exception:
						pass

				try:
					grab.stop()
				except Exception:
					pass
				if Picamera2:
					cap.stop()
				else:
					cap.release()
				cv2.destroyAllWindows()
				break


if __name__ == "__main__":
	main()
