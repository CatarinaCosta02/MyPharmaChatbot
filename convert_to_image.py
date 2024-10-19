import os
from pdf2image import convert_from_path

def convert_pdf_to_images(base_folder):
    poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin" # têm que adicionar o poppler nas variáveis de sistema
    
    subfolders = ['Condotril', 'Duobiotic', 'Neurofil']

    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)

        for pdf_file in os.listdir(folder_path):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, pdf_file)
                
                pages = convert_from_path(pdf_path, poppler_path=poppler_path)
                
                for i, page in enumerate(pages):
                    output_filename = f"{pdf_file[:-4]}_page_{i}.jpg"
                    output_path = os.path.join(folder_path, output_filename)
                    
                    page.save(output_path, "JPEG")
                
                print(f"Converted {pdf_file} to images in {folder_path}.")

base_folder = "documents"

convert_pdf_to_images(base_folder)