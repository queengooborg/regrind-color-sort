import os
import json
import numpy as np

class Palette:
	def __init__(self, settings):
		self.settings = settings
		self.path = self.settings["palette_path"]
		self.meta = {
			"deltaE_thresh": self.settings["deltaE_thresh"],
			"mah_thresh": self.settings["mah_thresh"],
			"min_margin": self.settings["min_margin"],
			"use_mahalanobis": self.settings["use_mahalanobis"]
		}
		self.colors = []
		if self.path and os.path.exists(self.path):
			self.load(self.path)

	def load(self, path):
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
		
		self.meta.update(data["meta"])
		self.settings["deltaE_thresh"] = self.meta["deltaE_thresh"]
		self.settings["mah_thresh"] = self.meta["mah_thresh"]
		self.settings["min_margin"] = self.meta["min_margin"]
		self.settings["use_mahalanobis"] = self.meta["use_mahalanobis"]

		self.colors = data.get("colors", [])
		for c in self.colors:
			if "centroid" in c and "mu" not in c:
				c["mu"] = c["centroid"]
			if "mu" not in c:
				continue
			if "cov" not in c:
				c["cov"] = [[25.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 60.0]]
			if "samples" not in c:
				c["samples"] = 1
		self.path = path

	def save(self, path=None):
		if path is None:
			path = self.path or self.settings.get("palette_path", "palette.json")
		self.meta["deltaE_thresh"] = self.settings["deltaE_thresh"]
		self.meta["mah_thresh"] = self.settings["mah_thresh"]
		self.meta["min_margin"] = self.settings["min_margin"]
		self.meta["use_mahalanobis"] = self.settings["use_mahalanobis"]
		with open(path, "w", encoding="utf-8") as f:
			json.dump({"version": 2, "meta": self.meta, "colors": self.colors}, f, indent=2)
		self.path = path

	def legend(self):
		if not self.colors:
			return "(empty)"
		parts = []
		for c in self.colors:
			key = c.get("key")
			name = c.get("name", "?")
			parts.append(f"{key}:{name}" if key else f"•:{name}")
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
	def _deltaE00(c1, c2):
		try:
			from skimage.color import deltaE_cie2000
			a = np.array(c1, np.float32).reshape(1, 1, 3)
			b = np.array(c2, np.float32).reshape(1, 1, 3)
			return float(deltaE_cie2000(a, b)[0, 0])
		except Exception:
			c1v = np.array(c1, np.float32)
			c2v = np.array(c2, np.float32)
			return float(np.linalg.norm(c1v - c2v))

	@staticmethod
	def _mahalanobis(x, mu, cov):
		xv = np.asarray(x, np.float32).reshape(3, 1)
		muv = np.asarray(mu, np.float32).reshape(3, 1)
		S = np.asarray(cov, np.float32).reshape(3, 3)
		eps = 1e-5
		Ss = S + np.eye(3, dtype=np.float32) * eps
		try:
			Si = np.linalg.inv(Ss)
		except Exception:
			Si = np.linalg.pinv(Ss)
		d2 = float((xv - muv).T @ Si @ (xv - muv))
		return float(np.sqrt(max(d2, 0.0)))

	def _robust_repr(self, lab_pixels):
		if len(lab_pixels) == 0:
			return None
		L = np.percentile(lab_pixels[:, 0], [25, 50, 75]).astype(np.float32)
		a = np.percentile(lab_pixels[:, 1], [25, 50, 75]).astype(np.float32)
		b = np.percentile(lab_pixels[:, 2], [25, 50, 75]).astype(np.float32)
		return [float(L[1]), float(a[1]), float(b[1])]

	def classify_lab(self, lab_pixels):
		if len(self.colors) == 0:
			return None
		if len(lab_pixels) == 0:
			return None
		x = self._robust_repr(lab_pixels)
		if x is None:
			return None
		useM = bool(self.settings.get("use_mahalanobis", True))
		best = None
		second = None
		for i, c in enumerate(self.colors):
			mu = c.get("mu") or c.get("centroid")
			if mu is None:
				continue
			if useM:
				cov = c.get("cov")
				if cov is None:
					cov = [[25.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 60.0]]
				score = self._mahalanobis(x, mu, cov)
			else:
				score = self._deltaE00(x, mu)
			item = (score, i, c.get("name", "?"))
			if best is None or score < best[0]:
				second = best
				best = item
			elif second is None or score < second[0]:
				second = item
		if best is None:
			return None
		if self.settings.get("use_mahalanobis", True):
			if best[0] > float(self.settings.get("mah_thresh", 3.0)):
				return None
			if second is not None and (second[0] - best[0]) < float(self.settings.get("min_margin", 0.4)):
				return None
			return best[2], best[0], best[1]
		else:
			if best[0] > float(self.settings.get("deltaE_thresh", 18.0)):
				return None
			if second is not None and (second[0] - best[0]) < float(self.settings.get("min_margin", 0.4)):
				return None
			return best[2], best[0], best[1]

	def add_class_from_pixels(self, name, lab_pixels, key=None):
		if len(lab_pixels) == 0:
			return None
		x = self._robust_repr(lab_pixels)
		if x is None:
			return None
		if key is None or len(key) == 0:
			key = self.auto_key(name)

		entry = {
			"name": name,
			"key": key if key else None,
			"mu": [float(x[0]), float(x[1]), float(x[2])],
			"cov": [[25.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 60.0]],
			"samples": 1
		}
		self.colors.append(entry)
		return entry

	def add_sample(self, idx, lab_pixels):
		if idx is None:
			return
		if idx < 0 or idx >= len(self.colors):
			return
		if len(lab_pixels) == 0:
			return
		x = self._robust_repr(lab_pixels)
		if x is None:
			return
		c = self.colors[idx]
		mu = np.array(c.get("mu", x), np.float32)
		S = np.array(c.get("cov", np.diag([25.0, 60.0, 60.0]).tolist()), np.float32)
		n = int(c.get("samples", 1))
		xv = np.array(x, np.float32)
		n1 = n + 1
		delta = xv - mu
		mu_new = mu + delta / n1
		delta2 = xv - mu_new
		C = S * max(n - 1, 1)
		C_new = C + np.outer(delta, delta2)
		S_new = C_new / max(n1 - 1, 1)
		lmb = 0.05
		S_shrink = (1 - lmb) * S_new + lmb * np.diag(np.diag(S_new) + 1.0)
		c["mu"] = [float(mu_new[0]), float(mu_new[1]), float(mu_new[2])]
		c["cov"] = [[float(S_shrink[0,0]), float(S_shrink[0,1]), float(S_shrink[0,2])],
		            [float(S_shrink[1,0]), float(S_shrink[1,1]), float(S_shrink[1,2])],
		            [float(S_shrink[2,0]), float(S_shrink[2,1]), float(S_shrink[2,2])]]
		c["samples"] = n1
