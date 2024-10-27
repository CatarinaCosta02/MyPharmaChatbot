import pytesseract
from PIL import Image
import os

def perform_ocr_on_images(base_folder):
    image_sets = { # nº de paginas de cada, automatizar este processo (tem que contar o numero de imagens criadas)
        'Condotril': 2,
        'Duobiotic': 1,
        'Neurofil': 2
    }

    for image_prefix, num_pages in image_sets.items():
        
        print(f"\nimage_prefix: ", image_prefix)
        print("num_pages: ", num_pages, f"\n")
        
        combined_text = ""

        folder_path = os.path.join(base_folder, image_prefix)
        
        for i in range(num_pages):
            image_path = os.path.join(folder_path, f"{image_prefix}_farmacia_page_{i}.jpg")
            print("image_path: ", image_path)
            
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='por')  # Realiza o OCR na imagem
            combined_text += text + "\n"  # Adiciona o texto de cada página
            
            print(f"{image_prefix} - Page {i + 1} processed.")

        output_txt_path = os.path.join(folder_path, f"{image_prefix}_farmacia.txt")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
        
        print(f"Text extraction complete for {image_prefix}. Saved to {output_txt_path}")


base_folder = "documents"

perform_ocr_on_images(base_folder)