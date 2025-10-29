import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
	from matplotlib import font_manager as _fm
	_HAS_MPL = True
except Exception:
	_HAS_MPL = False

# ========================= System font discovery (Unicode) =========================
_FONT_CACHE = {}

def _find_system_font_path():
	candidates = [
		"HelveticaNeue.ttf",
		"Helvetica Neue.ttf",
		"SegoeUI.ttf",
		"Segoe UI.ttf",
		"Arial.ttf",
		"Helvetica.ttf",
		"DejaVuSans.ttf",
		"NotoSans-Regular.ttf",
		"FreeSans.ttf",
		"Apple Color Emoji.ttc",
		"Arial Unicode.ttf",
		"NotoSansCJK-Regular.ttc"
	]
	paths = []
	if _HAS_MPL:
		try:
			paths = _fm.findSystemFonts(fontpaths=None, fontext="ttf")
			paths += _fm.findSystemFonts(fontpaths=None, fontext="ttc")
		except Exception:
			paths = []
	if not paths:
		common_dirs = []
		if os.name == "nt":
			common_dirs = [r"C:\Windows\Fonts"]
		elif sys.platform == "darwin":
			common_dirs = ["/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
		else:
			common_dirs = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]
		for root in common_dirs:
			if not os.path.isdir(root):
				continue
			for dirpath, _, filenames in os.walk(root):
				for fn in filenames:
					if fn.lower().endswith((".ttf", ".ttc", ".otf")):
						paths.append(os.path.join(dirpath, fn))
	paths_lower = [p.lower() for p in paths]
	for cand in candidates:
		cl = cand.lower()
		for p, pl in zip(paths, paths_lower):
			if cl in pl:
				return p
	return None

def get_default_font(size=18):
	key = f"{size}"
	if key in _FONT_CACHE:
		return _FONT_CACHE[key]
	path = _find_system_font_path()
	if path is not None:
		try:
			font = ImageFont.truetype(path, size)
			_FONT_CACHE[key] = font
			return font
		except Exception:
			pass
	font = ImageFont.load_default()
	_FONT_CACHE[key] = font
	return font

# ========================= Pillow text helpers =========================
def _text_bbox(font, text):
	img = Image.new("L", (2, 2), 0)
	draw = ImageDraw.Draw(img)
	# bbox = (left, top, right, bottom), top may be negative
	return draw.textbbox((0, 0), text, font=font)

def _measure_text(font, text):
	l, t, r, b = _text_bbox(font, text)
	return r - l, b - t

def _draw_unicode_text(img_bgr, text, org, color=(0, 255, 0), font=None):
	if font is None:
		font = get_default_font(18)
	# adjust origin by the font's top bearing so (x, y) is the VISUAL top-left
	l, t, r, b = _text_bbox(font, text)
	x, y = org
	y_adj = y - t
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	pil_img = Image.fromarray(img_rgb)
	draw = ImageDraw.Draw(pil_img)
	draw.text((x, y_adj), text, font=font, fill=(color[2], color[1], color[0]))
	return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def begin_text(img_bgr):
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	pil_img = Image.fromarray(img_rgb)
	draw = ImageDraw.Draw(pil_img)
	return pil_img, draw

def end_text(pil_img):
	return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ========================= UI helpers (Unicode text via Pillow) =========================
def put_panel(img, lines, top_left=(10, 10), pad=8, alpha=0.6, color=(0, 255, 0), size=18):
	x, y = top_left
	if not lines:
		return
	font = get_default_font(size)
	line_gap = 6

	# measure each line using true bbox
	widths = []
	heights = []
	tops = []
	for ln in lines:
		l, t, r, b = _text_bbox(font, ln)
		widths.append(r - l)
		heights.append(b - t)
		tops.append(t)

	panel_w = max(widths)
	# uniform line height = max visual height among lines
	line_h = max(heights)
	total_h = line_h * len(lines) + line_gap * (len(lines) - 1)

	overlay = img.copy()
	cv2.rectangle(overlay, (x - pad, y - pad), (x + panel_w + pad, y + total_h + pad), (0, 0, 0), -1)
	cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

	# draw each line at visual top-left with top-bearing compensation
	y_cursor = y
	for ln in lines:
		img[:] = _draw_unicode_text(img, ln, (x, y_cursor), color=color, font=font)
		y_cursor += line_h + line_gap

def put_banner(img, text, color=(0, 255, 255), size=22):
	H, W = img.shape[:2]
	font = get_default_font(size)
	w, h = _measure_text(font, text)
	pad = 10
	x = (W - w) // 2
	y_top = 15
	overlay = img.copy()
	cv2.rectangle(overlay, (x - pad, y_top), (x + w + pad, y_top + h + pad * 2), (0, 0, 0), -1)
	cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
	# draw with top-bearing compensation so text sits flush inside the box
	img[:] = _draw_unicode_text(img, text, (x, y_top + pad), color=color, font=font)

def draw_label(img, text, org, color=(0, 255, 0), size=18):
	font = get_default_font(size)
	img[:] = _draw_unicode_text(img, text, org, color=color, font=font)

def put_panel_ctx(pil_img, draw, img_bgr_shape, lines, top_left=(10, 10), pad=8, alpha=0.6, color=(0, 255, 0), size=18):
	# Reuse existing put_panel but avoid repeated conversions by drawing on pil_img directly
	# We call the original put_panel's inner logic by simulating _draw_unicode_text on the PIL context.
	# To keep it simple and non-invasive, we'll draw a filled rectangle via OpenCV overlay on the final image.
	# Here, we just compute positions and draw the text via PIL.
	if not lines:
		return pil_img
	font = get_default_font(size)
	line_gap = 6
	widths = []
	heights = []
	for ln in lines:
		w, h = _measure_text(font, ln)
		widths.append(w)
		heights.append(h)
	panel_w = max(widths)
	line_h = max(heights)
	total_h = line_h * len(lines) + line_gap * (len(lines) - 1)
	x, y = top_left
	# Draw background rectangle using PIL (semi-transparent effect approximated by a solid fill; final alpha overlay handled in OpenCV side if needed)
	draw.rectangle([x - pad, y - pad, x + panel_w + pad, y + total_h + pad], fill=(0, 0, 0))
	# Draw lines
	y_cursor = y
	for ln in lines:
		draw.text((x, y_cursor), ln, font=font, fill=(color[2], color[1], color[0]))
		y_cursor += line_h + line_gap
	return pil_img

def put_banner_ctx(pil_img, draw, img_bgr_shape, text, color=(0, 255, 255), size=22):
	H, W = img_bgr_shape[:2]
	font = get_default_font(size)
	w, h = _measure_text(font, text)
	pad = 10
	x = (W - w) // 2
	y_top = 15
	draw.rectangle([x - pad, y_top, x + w + pad, y_top + h + pad * 2], fill=(0, 0, 0))
	draw.text((x, y_top + pad), text, font=font, fill=(color[2], color[1], color[0]))
	return pil_img

def draw_label_ctx(pil_img, draw, text, org, color=(0, 255, 0), size=18):
	font = get_default_font(size)
	draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
	return pil_img
