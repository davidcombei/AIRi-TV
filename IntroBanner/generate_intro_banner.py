import sys
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def get_fitting_font(draw, text, font_path, max_width, start_size=120):
    font_size = start_size

    while font_size > 10:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            return font

        font_size -= 2

    return ImageFont.truetype(font_path, 10)


def main():
    if len(sys.argv) < 4:
        print("Usage: python generate_intro_banner.py \"Titlu\" \"Nume Autor\" /path/to/author_image.jpg")
        sys.exit(1)

    title = sys.argv[1].upper()
    author = sys.argv[2]
    author_image_path = sys.argv[3]

    base_dir = "/workspace/AIRI-TV-TEST/AIRi-TV/assets"
    input_path = os.path.join(base_dir, "schelet_banner.jpeg")
    output_path = os.path.join(base_dir, "banner_output.jpeg")
    font_path = os.path.join(base_dir, "BebasNeue-Regular.ttf")

    image = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # =========================
    # TITLU (identic)
    # =========================
    max_width = image.size[0] - 100
    title_font = get_fitting_font(draw, title, font_path, max_width)

    bbox = draw.textbbox((0, 0), title, font=title_font)
    text_width = bbox[2] - bbox[0]

    image_width, _ = image.size
    x = (image_width - text_width) // 2
    y = 40

    text_color = (31, 46, 92)

    draw.text((x, y), title, font=title_font, fill=text_color)

    # =========================
    # POZA AUTOR (HIGH QUALITY FIX)
    # =========================

    avatar_center_x = 264
    avatar_center_y = 289

    # dimensiunea FINALĂ (NU o schimbăm)
    avatar_width = 220
    avatar_height = 200

    # 🔥 scale intern pentru calitate
    scale_factor = 2
    big_w = avatar_width * scale_factor
    big_h = avatar_height * scale_factor

    author_img = Image.open(author_image_path).convert("RGB")

    # crop pătrat din centru
    w, h = author_img.size
    min_dim = min(w, h)

    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    author_img = author_img.crop((left, top, right, bottom))

    # 🔥 resize mare (calitate)
    author_img = author_img.resize(
        (big_w, big_h),
        Image.Resampling.LANCZOS
    )

    # 🔥 sharpen ușor
    author_img = author_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # mască mare
    mask = Image.new("L", (big_w, big_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0, big_w, big_h), fill=255)

    # aplicare mască
    avatar = Image.new("RGBA", (big_w, big_h))
    avatar.paste(author_img, (0, 0), mask)

    # 🔥 downscale la dimensiunea finală
    avatar = avatar.resize(
        (avatar_width, avatar_height),
        Image.Resampling.LANCZOS
    )

    top_left = (
        avatar_center_x - avatar_width // 2,
        avatar_center_y - avatar_height // 2
    )

    image.paste(avatar, top_left, avatar)

    # =========================
    # AUTOR TEXT (identic)
    # =========================

    author_font = ImageFont.truetype(font_path, 34)

    avatar_bottom_y = avatar_center_y + avatar_height // 2

    bbox = draw.textbbox((0, 0), author, font=author_font)
    author_width = bbox[2] - bbox[0]

    author_x = avatar_center_x - (author_width // 2)
    author_y = avatar_bottom_y + 20

    draw.text((author_x, author_y), author, font=author_font, fill=text_color)

    # =========================

    image.save(output_path)
    print(f"Banner generat: {output_path}")


if __name__ == "__main__":
    main()
