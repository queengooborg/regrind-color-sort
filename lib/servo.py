# regrind-color-sort - lib/servo.py
# 
# This program uses OpenCV to identify plastic regrind against a background and classify it by colors as specified by the user
# (c) 2026 Vinyl Da.i'gyu-Kazotetsu

try:
	from adafruit_servokit import ServoKit
	kit = ServoKit(channels=16)
except ImportError:
	kit = None

# Servos:
# 0: Deposit Flap
# 1: Reject Bin
# 2: Front-Back Stage (Stage 1)
# 3: Left-Right First Stage (Stage 2)
# 4: Left-Right Second Stage (Stage 3, Part 1)
# 5: Left-Right Final Stage (Stage 3, Part 2)

# Angles (Pending Actual Measurements):
# 0: Left
# 90: Center
# 180: Right

class Servos:
	def __init__(self):
		set_angle([0, 1, 2, 3, 4, 5], 90)

	def set_angle(self, i, angle):
		if not kit:
			print(f"Emulation: Setting {i} to {angle}...")
			return

		print(f"Setting {i} to {angle}...")
		if type(i) == int:
			kit.servo[i].angle = angle
		elif type(i) == list:
			for _i in list(i):
				kit.servo[_i].angle = angle

	def select_bin(self, i):
		# Bin Count: 16 + Reject

		if i == -1:
			set_angle(1, 180)
			return

		set_angle(1, 90)

		set_angle(2, 0 if i < 8 else 180)
		set_angle(3, 0 if i % 8 < 4 else 180)
		set_angle(4, 0 if i % 4 < 2 else 180)
		set_angle(5, 0 if i % 2 < 1 else 180)

	def drop_piece(self):
		set_angle(0, 180)

	def reset(self):
		set_angle(0, 90)