from PIL import Image, ImageDraw, ImageFont
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
os.makedirs(OUT_DIR, exist_ok=True)

SIZE = 1024
bg_gradient = [(45,108,223), (26,167,215)]  # start -> end rgb

def draw_gradient(im, c1, c2):
    w, h = im.size
    base = Image.new('RGB', (w, h), c1)
    top = Image.new('RGB', (w, h), c2)
    mask = Image.new('L', (w, h))
    for y in range(h):
        a = int(255 * (y / (h - 1)))
        Image.Draw.Draw(mask).line([(0, y), (w, y)], fill=a)
    im.paste(base, (0,0))
    im.paste(top, (0,0), mask)


def create_icon(size=1024):
    im = Image.new('RGBA', (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(im)

    # vertical gradient background
    for y in range(size):
        r = int(bg_gradient[0][0] + (bg_gradient[1][0] - bg_gradient[0][0]) * (y / size))
        g = int(bg_gradient[0][1] + (bg_gradient[1][1] - bg_gradient[0][1]) * (y / size))
        b = int(bg_gradient[0][2] + (bg_gradient[1][2] - bg_gradient[0][2]) * (y / size))
        draw.line([(0,y),(size,y)], fill=(r,g,b))

    # rounded rect background with padding
    pad = int(size * 0.03)
    radius = int(size * 0.18)
    rect_bbox = [pad, pad, size - pad, size - pad]
    draw.rounded_rectangle(rect_bbox, radius=radius, fill=None, outline=(0,0,0,30), width=0)

    # central dark circle
    cx = cy = size // 2 - int(size * 0.08)
    r = int(size * 0.215)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(14,27,45,220))

    # two T shapes (monogram)
    # left T
    tw = int(size * 0.14)
    th = int(size * 0.04)
    tx = cx - int(size * 0.07)
    ty = cy - int(size * 0.12)
    draw.rectangle((tx - tw//2, ty - th//2, tx + tw//2, ty + th//2), fill=(247,201,72))
    draw.rectangle((tx - tw//8, ty - th//2, tx + tw//8, ty + int(size * 0.12)), fill=(247,201,72))
    # right T (mirrored)
    tx2 = cx + int(size * 0.07)
    draw.rectangle((tx2 - tw//2, ty - th//2, tx2 + tw//2, ty + th//2), fill=(247,201,72))
    draw.rectangle((tx2 - tw//8, ty - th//2, tx2 + tw//8, ty + int(size * 0.12)), fill=(247,201,72))

    # subtle circle stroke
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(42,167,255,40), width=int(size*0.006))

    # brand text under the symbol
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size * 0.055))
    except Exception:
        font = ImageFont.load_default()
    text = "TAOTAOAPP"
    # Use textbbox for modern Pillow to compute text size
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
    draw.text((size/2 - tw/2, cy + r + int(size * 0.04)), text, font=font, fill=(255,255,255))

    return im


if __name__ == '__main__':
    im = create_icon(1024)
    im_512 = im.resize((512,512), Image.LANCZOS)
    im_256 = im.resize((256,256), Image.LANCZOS)
    im_512.save(os.path.join(OUT_DIR, 'taotaoapp_icon_512.png'))
    im_256.save(os.path.join(OUT_DIR, 'taotaoapp_icon_256.png'))
    print('Generated icons:', os.path.join(OUT_DIR, 'taotaoapp_icon_512.png'))