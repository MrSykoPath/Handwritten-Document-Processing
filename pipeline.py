import pandas as pd
import numpy as np
import cv2
import os
import base64
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2.service_account import Credentials
import io
import re

# Load environment variables from .env (useful for local tests)
load_dotenv()

# -------------------------
# IMAGE PROCESSING + OCR
# -------------------------
def process_image_with_openai(image_path, ai_model, api_key):
    """
    Preprocess an image and run OCR using OpenAI API.

    Args:
        image_path (str): Path to the image file.
        ai_model (str): OpenAI model name.
        api_key (str): OpenAI API key.

    Returns:
        str: OCR result in JSON format.
    """
    # --- Preprocessing Steps ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    preprocessed_path = f"Documents/preprocessed_{base_name}.jpg"

    if not os.path.exists(preprocessed_path):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=30, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        thresh = cv2.adaptiveThreshold(
            contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(preprocessed_path, closed)

    # Encode the preprocessed image
    def encode_image(img_path):
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    encoded_image = encode_image(preprocessed_path)

    # Initialize OpenAI client
    if not api_key:
        raise ValueError("Missing API key for OpenAI")
    client = OpenAI(api_key=api_key)

    # Run OCR request
    response = client.responses.create(
        model=ai_model,
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Extract the text from this image and provide JSON with schema: "
                        "{ Persons: [], Places: [], Companies: [], "
                        "Commodities: [{ name: string, date: string, price: string }], "
                        "Extracted_text: string, Language: ar|fr, "
                        "Translation_to_English: string, date: string }."
                        "Write names fully (no abbreviations). "
                        "If text is Arabic/French, keep 'Extracted_text' original but translate in 'Translation_to_English'."
                    )
                },
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
            ]
        }]
    )

    return response.output_text


# -------------------------
# GOOGLE DRIVE
# -------------------------
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'config/documentextraction-465311-6d37979e03e0.json'

def get_drive_service():
    creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)


# -------------------------
# MAIN PIPELINE
# -------------------------
def process_drive_folder(source_folder_id, result_folder_id, ai_model, api_key):
    """
    Download images from Google Drive, run OCR with OpenAI, upload JSON results back.
    
    Args:
        source_folder_id (str): Google Drive source folder ID.
        result_folder_id (str): Google Drive result folder ID.
        ai_model (str): OpenAI model name.
        api_key (str): OpenAI API key.
    
    Returns:
        dict: Summary of processed files, skipped files, and errors.
    """
    service = get_drive_service()
    files, files_in_results = [], []
    page_token = None

    # Fetch source files
    while True:
        results = service.files().list(
            q=f"'{source_folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
            pageToken=page_token
        ).execute()
        files.extend(results.get('files', []))
        page_token = results.get('nextPageToken')
        if page_token is None:
            break

    # Fetch existing results
    page_token = None
    while True:
        results = service.files().list(
            q=f"'{result_folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
            pageToken=page_token
        ).execute()
        files_in_results.extend(results.get('files', []))
        page_token = results.get('nextPageToken')
        if page_token is None:
            break

    numbered_files = [f for f in files if re.match(r"^\d+\.jpg$", f['name'])]
    numbered_files.sort(key=lambda x: int(os.path.splitext(x['name'])[0]))

    if not numbered_files:
        raise ValueError("No numbered image files found in source folder")

    documents_dir = 'Documents'
    os.makedirs(documents_dir, exist_ok=True)

    # Summary tracking
    summary = {"processed": 0, "skipped": 0, "errors": 0}

    for file in numbered_files:
        file_id, file_name = file['id'], file['name']
        result_filename = f"result_{os.path.splitext(file_name)[0]}_{ai_model}.json"
        result_path = os.path.join(documents_dir, result_filename)

        # Skip already processed files
        if result_filename in [f['name'] for f in files_in_results]:
            print(f"{result_filename} already exists, skipping")
            summary["skipped"] += 1
            continue

        print(f"Processing {file_name}...")

        # Download image
        try:
            fh = io.BytesIO()
            request = service.files().get_media(fileId=file_id)
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.seek(0)

            local_image_path = os.path.join(documents_dir, file_name)
            with open(local_image_path, 'wb') as f:
                f.write(fh.read())
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")
            summary["errors"] += 1
            continue

        # Run OCR
        try:
            ocr_result = process_image_with_openai(local_image_path, ai_model, api_key)
        except Exception as e:
            print(f"OCR failed for {file_name}: {e}")
            summary["errors"] += 1
            os.remove(local_image_path)
            continue

        # Save result JSON locally
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(ocr_result)

        # Upload to Drive
        try:
            media = MediaFileUpload(result_path, mimetype='application/json')
            service.files().create(
                body={'name': result_filename, 'parents': [result_folder_id]},
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()
            print(f"Uploaded {result_filename}")
        except Exception as e:
            print(f"Upload failed for {result_filename}: {e}")
            summary["errors"] += 1

        # Clean up local files
        try:
            os.remove(local_image_path)
            os.remove(result_path)
        except PermissionError:
            print(f"Could not delete temp files for {file_name}")

        summary["processed"] += 1

    return summary
