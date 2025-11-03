# regrind-color-sort - lib/settings.py
# 
# This program uses OpenCV to identify plastic regrind against a background and classify it by colors as specified by the user
# (c) 2025 ChatGPT, Vinyl Da.i'gyu-Kazotetsu

import cv2
import numpy as np
import json
import os
from cv2_enumerate_cameras import enumerate_cameras

from .ui import put_panel

class Settings:
	def __init__(self):
		self._FIELDS = {
			"d_ab": {"default": 12.0, "type": "float", "step": 1.0},
			"dL": {"default": 18.0, "type": "float", "step": 1.0},
			"use_edges": {"default": True, "type": "bool"},
			"min_area": {"default": 1500, "type": "int", "step": 100},
			"deltaE_thresh": {"default": 18.0, "type": "float", "step": 1.0},
			"bounding_boxes": {"default": False, "type": "bool"},
			"camera": {"default": 0, "type": "int", "step": 1},
			"palette_path": {"default": "palette.json", "type": "str"}
		}

		self.data = {}
		self.load()

	def load(self):
		try:
			with open("settings.json", "r") as f:
				self.data = json.load(f)
		except Exception:
			self.data = {key: value["default"] for key, value in self._FIELDS.items()}

	def save(self):
		with open("settings.json", "w", encoding="utf-8") as f:
			json.dump(self.data, f)

	def __getitem__(self, key):
		return self.data[key]

	def __setitem__(self, key, value):
		self.data[key] = value


# ========================= Settings panel (in-window) =========================
class SettingsUI:
	def __init__(self, settings):
		self.settings = settings
		self.fields = list(self.settings._FIELDS.items())
		self.idx = 0
		self.editing_text = False
		self.text_buf = ""
		self.cameras = enumerate_cameras()
		self.original_camera = self.settings["camera"]

	def show(self, img):
		lines = [
			"SETTINGS   [esc] to close",
			"[up]/[down] navigate   [left]/[right] change   [enter] toggle/edit)"
		]
		for i, (k, v) in enumerate(self.fields):
			val = self.settings[k]
			prefix = "> " if i == self.idx else "  "
			if k == "camera":
				val_str = self.cameras[val].name
				if self.original_camera != val:
					val_str += " (Changes on next launch)"
			elif self.editing_text and i == self.idx and v["type"] == "str":
				val_str = f"{self.text_buf}_"
			else:
				val_str = str(val)
			lines.append(f"{prefix}{k}: {val_str}")
		put_panel(img, lines, pos=(10, 110))

	def handle_key(self, k, cap_ref):
		kchar = chr(k) if 32 <= k < 127 else None
		name, v = self.fields[self.idx]

		# text edit mode
		if self.editing_text:
			if k == 27:
				self.editing_text = False
				self.text_buf = ""
				return
			if k in (13, 10):
				if self.text_buf != "":
					self.settings[name] = self.text_buf
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
			if v["type"] == "bool":
				self.settings[name] = not self.settings[name]
				return
			if v["type"] == "str":
				self.editing_text = True
				self.text_buf = str(self.settings[name])
				return

		if k == 27:
			return 'exit'

	def _bump(self, direction, cap_ref):
		name, v = self.fields[self.idx]

		print(name, v)

		if v["type"] == "bool":
			self.settings[name] = not self.settings[name]
			return

		if name == "camera":
			delta = (v["step"] or 1) * direction
			val = self.settings[name] + delta
			if val < 0:
				val = len(self.cameras) - 1
			if val >= len(self.cameras):
				val = 0
			self.settings[name] = val
			return

		if v["type"] in ("int", "float"):
			delta = (v["step"] or 1) * direction
			if v["type"] == "int":
				self.settings[name] = max(0, int(self.settings[name] + delta))
			else:
				self.settings[name] = float(self.settings[name] + delta)
			return
