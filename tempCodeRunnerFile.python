import os
import re
from PyPDF2 import PdfReader

def clean_text(text):
    """
    Cleans the input text by ensuring proper spacing around punctuation marks
    and removing unwanted characters.
    """
    # Ensure there's a space after punctuation marks if missing
    text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
    # Fix stuck-together words due to missing spaces
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Remove unwanted special characters (but keep punctuation)
    text = re.sub(r'[^A-Za-z0-9\s.,!?;:\'\"]', '', text)
    # Remove multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

def extract_text_from_pdf(pdf_path, output_txt_path):
    """
    Extracts text from a PDF file, cleans it, and saves it to a text file.
    """
    try:
        reader = PdfReader(pdf_path)
        extracted_text = ""
        
        # Iterate through each page and extract text
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += clean_text(text) + "\n\n"
        
        # Save the cleaned text to the output file
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(extracted_text)
        
        print(f"Text successfully extracted and saved to {output_txt_path}")
    
    except Exception as e:
        print(f"An error occurred while processing {pdf_path}: {e}")

def process_folder(input_folder, output_folder):
    """
    Processes all PDF files in the specified folder, extracting text and
    saving it to text files with the same base name.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, file_name)
            output_txt_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")
            extract_text_from_pdf(pdf_path, output_txt_path)

# Folder paths
input_folder = "C:/Users/tuant/Downloads/OneDrive_2024-12-08 (2)/Annual and Integrated 2"  # Replace with the path to your folder containing PDFs
output_folder = "C:/Users/tuant/Downloads/OneDrive_2024-12-08 (2)/extracted"  # Replace with the desired output folder path

# Process all PDF files in the folder
process_folder(input_folder, output_folder)
