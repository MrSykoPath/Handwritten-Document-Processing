import pandas as pd
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import base64
from matplotlib import pyplot as plt
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# --- Preprocessing Steps ---
image_path = "Documents/behna_ar.jpg"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

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

# Save the preprocessed image to a temporary file
preprocessed_path = "preprocessed_ar.jpg"
cv2.imwrite(preprocessed_path, closed)

#-----------------------------
# --- OCR with OpenAI ---
# Initialize OpenAI client

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

Api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=Api_key)

# Encode the preprocessed image
encoded_image = encode_image(preprocessed_path)

response = client.responses.create(
    model="o3-2025-04-16",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text", "text": "Can you extract the text from this image and provide it in a structured JSON format?"
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded_image}",
                },
            ],
        }
    ],
)

print("OCR Response:")
print(response.output_text)