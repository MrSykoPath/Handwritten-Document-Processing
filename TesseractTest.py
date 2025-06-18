import pandas as pd
import numpy as np
import cv2
import os
from mistralai import Mistral
from dotenv import load_dotenv
import base64
from IPython.display import display, Markdown
from pydantic import BaseModel, Field
from mistralai.extra import response_format_from_pydantic_model
import anthropic
from google.cloud import vision
from matplotlib import pyplot as plt

image_path = "Documents/behna_fr2.jpg"
#Load image from the specified path
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1000, 800)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Display the grayscale image
cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Grayscale Image", 1000, 800)
cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Denoise using Non-Local Means Denoising
denoised = cv2.fastNlMeansDenoising(gray, h=30, templateWindowSize=7, searchWindowSize=21)
#display the denoised image
cv2.namedWindow("Denoised Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Denoised Image", 1000, 800)
cv2.imshow("Denoised Image", denoised)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
contrast = clahe.apply(denoised)
# Display the denoised and contrast-enhanced image
cv2.namedWindow("Contrast Enhanced Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contrast Enhanced Image", 1000, 800)
cv2.imshow("Contrast Enhanced Image", contrast)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Adaptive thresholding to create a binary image
thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 25, 15)
# Display the processed image
cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Processed Image", 1000, 800)
cv2.imshow("Processed Image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.namedWindow("Closed Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Closed Image", 1000, 800)
cv2.imshow("Closed Image", closed)
cv2.waitKey(0)
cv2.destroyAllWindows()