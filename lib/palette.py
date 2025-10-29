import os
import json
import numpy as np

class Palette:
	def __init__(self, settings):
		self.settings = settings
		self.path = self.settings["palette_path"]
		self.meta = {"deltaE_thresh": self.settings["deltaE_thresh"]}
		self.colors = []
		if self.path and os.path.exists(self.path):
			self.load(self.path)

	def load(self, path):
		with open(path, "r") as f:
			data = json.load(f)
		self.meta = data.get("meta", self.meta)
		self.colors = data.get("colors", [])
		self.path = path
		self.settings["deltaE_thresh"] = float(self.meta.get("deltaE_thresh", self.settings["deltaE_thresh"]))

	def save(self, path=None):
		if path is None:
			path = self.path or self.settings["palette_path"]
		self.meta["deltaE_thresh"] = self.settings["deltaE_thresh"]
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

	def auto_key(self, name):
		used = {c.get("key") for c in self.colors if c.get("key")}
		for l in list(name.lower()) + list("1234567890-=[]\\;'/`"):
			if l not in used:
				return l
		return None

	@staticmethod
	def _deltaE76(c1, c2):
		# Prefer perceptual ΔE2000 (skimage). Fall back to SciPy euclidean, then NumPy.
		try:
			from skimage.color import deltaE_cie2000
			c1a = np.array(c1, np.float32).reshape(1, 1, 3)
			c2a = np.array(c2, np.float32).reshape(1, 1, 3)
			return float(deltaE_cie2000(c1a, c2a)[0, 0])
		except Exception:
			try:
				from scipy.spatial.distance import cdist
				a = np.asarray([c1], dtype=np.float32)
				b = np.asarray([c2], dtype=np.float32)
				return float(cdist(a, b, metric="euclidean")[0, 0])
			except Exception:
				c1v = np.array(c1, np.float32)
				c2v = np.array(c2, np.float32)
				return float(np.linalg.norm(c1v - c2v))

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

		if best_d <= float(self.settings["deltaE_thresh"]):
			return best_name, best_d, best_idx

		return None

	def add_class_from_pixels(self, name, lab_pixels, key=None):
		if len(lab_pixels) == 0:
			return None

		L = float(np.median(lab_pixels[:, 0]))
		a = float(np.median(lab_pixels[:, 1]))
		b = float(np.median(lab_pixels[:, 2]))

		if key is None or len(key) == 0:
			key = self.auto_key(name)

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
