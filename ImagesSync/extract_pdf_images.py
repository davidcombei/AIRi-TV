import fitz  # PyMuPDF
import os
import argparse


def extract_images_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    total_images = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            filename = f"page{page_index}_img{img_index}.{image_ext}"
            save_path = os.path.join(output_dir, filename)

            with open(save_path, "wb") as f:
                f.write(image_bytes)

            total_images += 1

    return total_images


def main():
    parser = argparse.ArgumentParser(
        description="Extract images from a PDF and save them to assets/pdf_images"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Full or relative path to the PDF file"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        print(f"ERROR: PDF not found: {args.pdf_path}")
        return

    # Determină folderul proiectului (AIRi-TV)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Creează assets/pdf_images/<pdf_name>/
    pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
    output_dir = os.path.join(project_root, "assets", "pdf_images")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting images from: {args.pdf_path}")
    print(f"Saving images to: {output_dir}")

    count = extract_images_from_pdf(args.pdf_path, output_dir)

    print(f"Done. {count} images extracted.")


if __name__ == "__main__":
    main()


