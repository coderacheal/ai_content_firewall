# main.py
import os
import cv2
import requests
import shutil
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
from tensorflow.keras.models import load_model
from io import BytesIO
from urllib.parse import urljoin
import joblib


# Model paths
MODEL_PATH = "./ml/models/content_firewall_model.h5"
TEXT_MODEL_PATH = "./ml/models/Logistic_Regression.joblib"
IMAGE_LABEL_ENCODER_PATH = "./ml/models/label_encoder.pkl"
TEXT_LABEL_ENCODER_PATH = "./ml/models/label_encoder.joblib"


# Global variables for models and encoders (to be loaded only once)
cnn_model = None
text_model = None
image_label_encoder = None
text_label_encoder = None


def load_models():
    """Load models and encoders into global variables."""
    global cnn_model, text_model, image_label_encoder, text_label_encoder
    cnn_model = load_model(MODEL_PATH)
    text_model = joblib.load(TEXT_MODEL_PATH)

    with open(IMAGE_LABEL_ENCODER_PATH, "rb") as f:
        image_label_encoder = joblib.load(f)
    with open(TEXT_LABEL_ENCODER_PATH, "rb") as f:
        text_label_encoder = joblib.load(f)

    print("Model read")


# Call load_models() once when the API starts.
load_models()

# Folder to save scraped images
SCRAPED_IMAGE_DIR = "../api/scraped_images"
os.makedirs(SCRAPED_IMAGE_DIR, exist_ok=True)


def scrape_images(url):
    """Extract image URLs from a webpage."""
    image_urls = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/111.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for img_tag in soup.find_all("img"):
            img_url = img_tag.get("src")
            if img_url:
                full_url = urljoin(url, img_url)
                if full_url.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_urls.append(full_url)
    except Exception as e:
        print(f"Error scraping images: {e}")
    return image_urls


def scrape_text(url):
    """Extract and clean text content from a webpage."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/111.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        cleaned_text = " ".join(text.split())
        return cleaned_text
    except Exception as e:
        print(f"Error scraping text: {e}")
        return ""


def preprocess_image(image_path):
    """Open, resize, normalize, and prepare a local image for CNN prediction."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, image_path
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None


def download_and_save_image(image_url):
    """Download an image from a URL and save it locally."""
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            filename = os.path.basename(image_url)
            image_path = os.path.join(SCRAPED_IMAGE_DIR, filename)
            img.save(image_path)
            return image_path
    except Exception as e:
        print(f"Error downloading image {image_url}: {e}")
    return None


def predict_image(image_path):
    """Predict the class probabilities for an image using the CNN model and determine the predicted label."""
    preprocessed_image, _ = preprocess_image(image_path)
    if preprocessed_image is None:
        return None
    preds = cnn_model.predict(preprocessed_image)[0]  # shape: (num_classes,)
    # Determine predicted index and label using inverse transformation:
    predicted_index = int(np.argmax(preds))
    predicted_label = image_label_encoder.inverse_transform([predicted_index])[0]
    
    # Create a dictionary with each probability and include the predicted label.
    prob_dict = dict(zip(image_label_encoder.classes_, preds))
    prob_dict["predicted_label"] = predicted_label
    print(prob_dict)
    return prob_dict


def predict_text_content(text):
    """Predict the class probabilities for text using the text classification model."""
    # If your text model requires preprocessing (tokenization, etc.), add it here.
    # Using predict_proba if available.
    preds = text_model.predict_proba([text])
    preds = preds[0]  # Assuming output shape (1, num_classes)
    labels = text_label_encoder.classes_
    prob_dict = dict(zip(labels, preds))
    return prob_dict


def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0  # duration in seconds

    # Determine the number of frames to extract
    if duration > 120:  # longer than 2 minutes
        num_frames = 10
    else:
        num_frames = 5

    # If the video has fewer frames than desired, adjust
    num_frames = min(num_frames, total_frames) if total_frames > 0 else 0
    
    # Get evenly spaced indices
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    else:
        indices = []

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def save_frames(frames, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    paths = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(folder, f"frame_{i}.jpg")
        cv2.imwrite(frame_path, frame)
        paths.append(frame_path)
    return paths


def clean_prediction(pred):
    """Convert NumPy types in the prediction dict to native Python types."""
    cleaned = {}
    for k, v in pred.items():
        key = str(k)
        if isinstance(v, np.floating):
            value = float(v)
        elif isinstance(v, np.integer):
            value = int(v)
        elif isinstance(v, np.ndarray):
            value = v.tolist()
        else:
            value = str(v) if isinstance(v, np.str_) else v
        cleaned[key] = value
    return cleaned