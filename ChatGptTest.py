import pandas as pd
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import base64
from matplotlib import pyplot as plt
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2.service_account import Credentials
import io
import re

# Load environment variables from .env
load_dotenv()

def process_image_with_openai(image_path):
    """
    Process an image through preprocessing steps and then use OpenAI's OCR capabilities
    to extract and structure text data.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Structured JSON response from OpenAI
    """
    # --- Preprocessing Steps ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
     # Create a unique preprocessed image name based on the original filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    preprocessed_path = f"preprocessed_{base_name}.jpg"

    if not os.path.exists(f"Documents/{preprocessed_path}"):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Denoise using Non-Local Means Denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=30, templateWindowSize=7, searchWindowSize=21)
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 25, 15)
        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(f"Documents/{preprocessed_path}", closed)
    else:
        print(f"Preprocessed image already exists: {preprocessed_path}")

    # --- OCR with OpenAI ---
    # Function to encode the image
    def encode_image(img_path):
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # Encode the preprocessed image
    encoded_image = encode_image(f"Documents/{preprocessed_path}")

    if not os.path.exists(f"Documents/result_{base_name}.json"):
        response = client.responses.create(
        model="gpt-4.1-2025-04-14",  # Use the latest model
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text", "text": "Can you extract the text from this image and provide it in a structured JSON format according to this schema: {\n\"Persons\": [],\n\"Places\": [],\n\"companies\": [],\n\"commodities\": [ { \"name\": \"string\", \"date\": \"string\", \"price\": \"string\" } ],\n\"extracted_text\": \"string\",\n\"language\": \"ar or fr\",\n\"Translation_to_English\": \"string\",\n\"date\": \"string\"\n}, make sure you do not abbreviate the names and write everything in full, and also provide the translation to English if the text is in Arabic or French. Also make the JSON attributes values in English if the text is in Arabic or French except extracted text attribute, since the Translation to english attribute exists.",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                ],
            }
        ],
    )
         # Clean up the temporary file
        # os.remove(preprocessed_path)
        # Uncomment above line if you want to delete the preprocessed image
        
        return response.output_text
    else:
        return None
   



# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'config/documentextraction-465311-6d37979e03e0.json'  # Path to your credentials file

# Authenticate and create the Drive API service
def get_drive_service():
    creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

# Download all images from the source folder, process, and upload results
def process_drive_folder(source_folder_id, result_folder_id):
    service = get_drive_service()
   # List all image files in the source folder (handle pagination)
    files = []
    files_in_results = []
    page_token = None
    while True:
        results = service.files().list(
            q=f"'{source_folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,  # Increase page size if needed (max 1000)
            pageToken=page_token
        ).execute()
        files.extend(results.get('files', []))
        page_token = results.get('nextPageToken', None)
        if page_token is None:
            break
    while True:
        # Get files in the results folder
        results = service.files().list(
            q=f"'{result_folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,  # Increase page size if needed (max 1000)
            pageToken=page_token
        ).execute()
        files_in_results.extend(results.get('files', []))
        page_token = results.get('nextPageToken', None)
        if page_token is None:
            break

    print(f"Found {len(files)} images in source folder.")
    # Only process files with names that are numbers (e.g., 001.jpg, 002.jpg, ..., 124.jpg)
    numbered_files = [file for file in files if re.match(r"^\d+\.jpg$", file['name'])]
    numbered_files = sorted(
        numbered_files, key=lambda x: int(os.path.splitext(x['name'])[0])
    )
    print(f"Found {len(numbered_files)} numbered images in source folder.")
    for file in numbered_files:
        file_id = file['id']
        file_name = file['name']
        print(f"Processing {file_name}...")
        # Download the image
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        # Save to disk
        documents_dir = 'Documents'
        if not os.path.exists(documents_dir):
            os.makedirs(documents_dir)
        local_image_path = os.path.join(documents_dir, file_name)
        with open(local_image_path, 'wb') as f:
            f.write(fh.read())
        # Run OCR
        ocr_result = process_image_with_openai(local_image_path)
        if ocr_result is not None:
            # Save result as JSON
            result_filename = f"result_{os.path.splitext(file_name)[0]}.json"
            result_path = os.path.join(documents_dir, result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(ocr_result)
        else:
            print(f"Result for {file_name} is already processed")
            result_filename = f"result_{os.path.splitext(file_name)[0]}.json"
            result_path = os.path.join(documents_dir, result_filename)
        # Upload result to result folder
        file_metadata = {
            'name': result_filename,
            'parents': [result_folder_id]
        }
         # Open the file, upload, then close before deleting
        if result_filename not in [f['name'] for f in files_in_results]:
            media = MediaFileUpload(result_path, mimetype='application/json')
            service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()
            print(f"Uploaded {result_filename} to result folder.")
        else:
            print(f"Result file {result_filename} already exists in the result folder, skipping upload.")

   
        try:
            os.remove(local_image_path)
            os.remove(result_path)
        except PermissionError as e:
            print(f"Could not delete file: {e}")

# Example usage
if __name__ == "__main__":
    # image_path = "Documents/behna_ar.jpg"
    # result = process_image_with_openai(image_path)
    # print("OCR Response:")
    # print(result)

    # Pipeline usage
    source_folder_id = "1loLg-htSD0XtU5MgzofbVCd4lFMsiKpg"
    result_folder_id = "1NN8_VERmh4xe0Z2mZiQSqYkwxmj5fNRs"
    process_drive_folder(source_folder_id, result_folder_id)
    print("Pipeline complete.")