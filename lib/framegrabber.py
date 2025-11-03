# regrind-color-sort - lib/framegrabber.py
# 
# This program uses OpenCV to identify plastic regrind against a background and classify it by colors as specified by the user
# (c) 2025 ChatGPT, Vinyl Da.i'gyu-Kazotetsu

from threading import Thread, Lock

class FrameGrabber:
	def __init__(self, cap, is_picam):
		self.cap = cap
		self.is_picam = is_picam
		self.lock = Lock()
		self.frame = None
		self.stopped = False
		self.t = Thread(target=self._loop, daemon=True)

	def start(self):
		self.t.start()
		return self

	def _loop(self):
		while not self.stopped:
			if self.is_picam:
				f = self.cap.capture_array()
			else:
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