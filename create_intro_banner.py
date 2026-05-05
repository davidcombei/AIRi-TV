import fitz
from PIL import Image, ImageDraw, ImageFont
import sys
import os

if len(sys.argv) != 4:
    print("Usage: python create_intro_banner.py <pdf> <author_name> <author_img>")
    sys.exit(1)

pdf_path = sys.argv[1]
author_name = sys.argv[2]
author_img_path = sys.argv[3]

# --- output fix ---
output_path = "assets/banner.png"

# --- extrage titlu ---
doc = fitz.open(pdf_path)
title = doc.metadata.get("title", "")

if not title:
    page = doc[0]
    text = page.get_text().split("\n")
    title = text[0]

# --- canvas ---
W, H = 1280, 720
img = Image.new("RGB", (W, H), (15, 15, 25))
draw = ImageDraw.Draw(img)

# --- font ---
font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 60)
font_author = ImageFont.truetype("DejaVuSans.ttf", 40)

# --- text ---
draw.text((60, 120), title[:80], font=font_title, fill=(255,255,255))
draw.text((60, 220), f"By {author_name}", font=font_author, fill=(180,180,180))

# --- imagine autor ---
author_img = Image.open(author_img_path).convert("RGB").resize((300,300))
img.paste(author_img, (900, 200))

# --- salvare ---
os.makedirs("assets", exist_ok=True)
img.save(output_path)

print(f"[IntroBanner] Banner saved at {output_path}")
