from PIL import Image, ImageDraw, ImageFont
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
os.makedirs(OUT_DIR, exist_ok=True)

SIZES = [1024, 512, 256, 128, 64, 48, 32]

NEON_START = (124, 58, 237)
NEON_END = (6, 182, 212)


def vertical_gradient(draw, bbox, c1, c2):
    x0, y0, x1, y1 = bbox
    h = y1 - y0
    for i in range(h):
        r = int(c1[0] + (c2[0] - c1[0]) * (i / max(1, h - 1)))
        g = int(c1[1] + (c2[1] - c1[1]) * (i / max(1, h - 1)))
        b = int(c1[2] + (c2[2] - c1[2]) * (i / max(1, h - 1)))
        draw.line([(x0, y0 + i), (x1, y0 + i)], fill=(r, g, b))


def draw_shell_symbol(draw, center, size):
    cx, cy = center
    w = size
    # outer rounded rectangle plaque
    plaque = [cx - w * 0.6, cy - w * 0.35, cx + w * 0.6, cy + w * 0.35]
    draw.rounded_rectangle(plaque, radius=int(w * 0.06), fill=(5,24,36))

    # shell body: an arc-like filled polygon
    shell_top = cy - int(w * 0.08)
    coords = [
        (cx - int(w * 0.55), shell_top),
        (cx - int(w * 0.32), cy + int(w * 0.28)),
        (cx + int(w * 0.32), cy + int(w * 0.28)),
        (cx + int(w * 0.55), shell_top)
    ]
    draw.polygon(coords, fill=(7,24,42))

    # inner terminal box
    tb_w = int(w * 0.34)
    tb_h = int(w * 0.22)
    tb = [cx - tb_w // 2, cy - tb_h // 2, cx + tb_w // 2, cy + tb_h // 2]
    draw.rounded_rectangle(tb, radius=int(w * 0.03), fill=(0,17,35))

    # prompt text >_ with neon gradient-ish color (solid for Pillow)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', int(w * 0.12))
    except Exception:
        font = ImageFont.load_default()
    text = '>_'
    # center text
    if hasattr(draw, 'textbbox'):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    else:
        tw, th = draw.textsize(text, font=font)
    draw.text((cx - tw / 2, cy - th / 2), text, font=font, fill=NEON_END)

    # neon diagonal accent
    acc_w = int(w * 0.04)
    acc = [cx + int(w * 0.32), cy - int(w * 0.45), cx + int(w * 0.41), cy + int(w * 0.05)]
    draw.rounded_rectangle(acc, radius=int(acc_w/2), fill=NEON_START)


def create_icons():
    for size in SIZES:
        im = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(im)
        # background rounded square with dark base
        draw.rounded_rectangle((0, 0, size, size), radius=int(size * 0.18), fill=(7, 12, 20))
        # gradient overlay
        vertical_gradient(draw, (0, 0, size, size), (11, 16, 32), (6, 11, 24))

        # draw shell symbol centered a bit above center
        center = (size // 2, int(size * 0.42))
        draw_shell_symbol(draw, center, size)

        # label text below (optional small)
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', int(size * 0.06))
        except Exception:
            font = ImageFont.load_default()
        label = 'COOLSHELL'
        if size >= 128:
            bbox = draw.textbbox((0,0), label, font=font) if hasattr(draw, 'textbbox') else (0,0, draw.textsize(label, font=font)[0], draw.textsize(label, font=font)[1])
            tw = bbox[2] - bbox[0]
            draw.text(((size - tw) / 2, int(size * 0.78)), label, font=font, fill=(167,243,208))

        out_path = os.path.join(OUT_DIR, f'coolshell_icon_{size}.png')
        im.save(out_path)
        print('Saved', out_path)


if __name__ == '__main__':
    create_icons()
