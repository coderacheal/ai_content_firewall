import os
import requests
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
from tensorflow.keras.models import load_model
from io import BytesIO
# from urllib.parse import urljoin
from .preprocess_func import scrape_images, scrape_text, save_frames, extract_frames_from_video
from .preprocess_func import download_and_save_image, predict_image, predict_text_content, load_models, clean_prediction


app = FastAPI()
history = [] 

load_models()

# Request model for website prediction
class WebsiteRequest(BaseModel):
    url: str


@app.post("/predict/website")
async def predict_website(website: WebsiteRequest):
    url = website.url
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Unable to retrieve URL: {e}"}
    
    # Scrape images and text from the URL
    image_urls = scrape_images(url)
    image_predictions = []
    for img_url in image_urls:
        img_path = download_and_save_image(img_url)
        if img_path:
            pred = predict_image(img_path)
            if pred is not None:
                image_predictions.append(pred)
    
    # Average image predictions if available
    avg_image_pred = {"Safe": 0, "Harmful": 0, "Adult": 0, "Suicide": 0}
    if image_predictions:
        for pred in image_predictions:
            for key in avg_image_pred:
                avg_image_pred[key] += pred.get(key, 0)
        for key in avg_image_pred:
            avg_image_pred[key] /= len(image_predictions)
    
    # Scrape text and get text prediction
    combined_text = scrape_text(url)
    text_pred = predict_text_content(combined_text) if combined_text else {'Safe': 0, 'Harmful': 0, 'Adult': 0, 'Suicide': 0}

    # Combine predictions (adjust weights as needed)
    overall = {
        "Safe": (avg_image_pred.get("Safe", 0) + text_pred.get("Safe", 0)) / 2,
        "Harmful": (avg_image_pred.get("Harmful", 0) + text_pred.get("Harmful", 0)) / 2,
        "Adult": (avg_image_pred.get("Adult", 0) + text_pred.get("Adult", 0)) / 2,
        "Suicide": (avg_image_pred.get("Suicide", 0) + text_pred.get("Suicide", 0)) / 2,
    }
    verdict = "Safe" if overall["Safe"] > max(overall["Harmful"], overall["Adult"], overall["Suicide"]) else "Unsafe"
    result = {"overall": overall, "verdict": verdict}
    history.append({"url": url, "result": result})
    return result


@app.post("/predict/image")
async def predict_single_image(file: UploadFile = File(...)):
    temp_dir = "./temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    pred = predict_image(file_path)
    # Clean the dictionary so it can be JSON serialized
    pred_clean = clean_prediction(pred)
    verdict = "Safe" if pred_clean["Safe"] > max(pred_clean["Harmful"], pred_clean["Adult"], pred_clean["Suicide"]) else "Unsafe"
    history.append({"file": file.filename, "result": {"prediction": pred_clean, "verdict": verdict}})
    return {"prediction": pred_clean, "verdict": verdict}


@app.post("/predict/text")
async def predict_text_endpoint(text: str = Form(...)):
    pred = predict_text_content(text)
    verdict = "Safe" if pred["Safe"] > max(pred["Harmful"], pred["Adult"], pred["Suicide"]) else "Unsafe"
    history.append({"text": text[:30] + "...", "result": {"prediction": pred, "verdict": verdict}})
    return {"prediction": pred, "verdict": verdict}


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    temp_video_dir = "./api/temp_videos"
    os.makedirs(temp_video_dir, exist_ok=True)
    video_path = os.path.join(temp_video_dir, file.filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    frames = extract_frames_from_video(video_path)
    frame_folder = "./api/temp_video_frames"
    frame_paths = save_frames(frames, frame_folder)
    
    frame_preds = []
    for fp in frame_paths:
        pred = predict_image(fp)
        if pred is not None:
            frame_preds.append(pred)
    
    avg_frame_pred = {"Safe": 0, "Harmful": 0, "Adult": 0, "Suicide": 0}
    if frame_preds:
        for pred in frame_preds:
            for key in avg_frame_pred:
                avg_frame_pred[key] += pred.get(key, 0)
        for key in avg_frame_pred:
            avg_frame_pred[key] /= len(frame_preds)
    
    # Clean the prediction dictionary so it contains native Python types.
    avg_frame_pred = clean_prediction(avg_frame_pred)
    
    verdict = "safe" if avg_frame_pred["Safe"] > max(avg_frame_pred["Harmful"], avg_frame_pred["Adult"], avg_frame_pred["Suicide"]) else "Unsafe"
    history.append({"video": file.filename, "result": {"prediction": avg_frame_pred, "verdict": verdict}})
    return {"prediction": avg_frame_pred, "verdict": verdict}



@app.get("/history")
async def get_history():
    return history

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
