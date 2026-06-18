"""
CroplandQuantum - FastAPI Backend
Serves ML predictions and weather-based crop recommendations
"""

import os
import io
import requests
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(title="CroplandQuantum API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Device + Model
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best_model.pth")

def _load_model():
    weights = MobileNet_V2_Weights.DEFAULT
    m = mobilenet_v2(weights=weights)
    m.classifier[1] = nn.Linear(m.last_channel, 2)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m = m.to(DEVICE)
    m.eval()
    return m

model = _load_model()

# ─────────────────────────────────────────────
# Image transform
# ─────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def classify_image(img: Image.Image):
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return int(predicted.item()), float(confidence.item())

# ─────────────────────────────────────────────
# Location helpers
# ─────────────────────────────────────────────
INDIA_LOCATIONS = {
    "Andhra Pradesh": (15.9129, 78.6675), "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376), "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661), "Goa": (15.2993, 73.8243),
    "Gujarat": (22.2587, 71.1924), "Haryana": (29.0588, 77.0745),
    "Himachal Pradesh": (31.1048, 77.1734), "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139), "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9375, 78.6553), "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063), "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1815, 92.9789), "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985), "Punjab": (31.5204, 74.3587),
    "Rajasthan": (27.0238, 74.2179), "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569), "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882), "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193), "West Bengal": (24.5155, 88.3289),
    "Hyderabad": (17.3850, 78.4867), "Bangalore": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777), "Delhi": (28.7041, 77.1025),
    "Pune": (18.5204, 73.8567), "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639), "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873), "Lucknow": (26.8467, 80.9462),
}

def get_coordinates(location: str):
    clean = location.strip().title()
    for key, coords in INDIA_LOCATIONS.items():
        if key.lower() == clean.lower():
            return coords[0], coords[1], key, "India"
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        r = requests.get(url, params={"name": location, "country": "IN", "language": "en", "limit": 1}, timeout=5)
        data = r.json()
        if data.get("results"):
            res = data["results"][0]
            return res["latitude"], res["longitude"], res["name"], res.get("admin1", "India")
    except Exception:
        pass
    return None, None, None, None

def get_weather_data(lat, lon):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
            },
            timeout=10,
        )
        return r.json()
    except Exception:
        return None

def get_crop_recommendations(temp_avg: float, humidity: float, precipitation: float):
    if temp_avg < 10:
        crops = ["Wheat", "Barley", "Rye", "Oats"]
        temp_note = f"Cold ({temp_avg:.1f}°C) – ideal for winter crops."
    elif temp_avg < 18:
        crops = ["Alfalfa", "Clover", "Pasture Grass", "Timothy Hay"]
        temp_note = f"Cool ({temp_avg:.1f}°C) – great for forage crops."
    elif temp_avg < 24:
        crops = ["Corn", "Soybeans", "Rice", "Sunflower"]
        temp_note = f"Warm ({temp_avg:.1f}°C) – perfect for summer crops."
    else:
        crops = ["Cotton", "Sugar Cane", "Millet", "Sorghum"]
        temp_note = f"Hot ({temp_avg:.1f}°C) – best for heat-tolerant crops."

    if humidity < 30:
        hum_note = f"Very dry ({humidity}%) – focus on drought-tolerant varieties."
    elif humidity < 50:
        hum_note = f"Moderate humidity ({humidity}%) – good for most crops."
    elif humidity < 70:
        hum_note = f"High humidity ({humidity}%) – consider crop rotation."
    else:
        hum_note = f"Very humid ({humidity}%) – watch for fungal diseases."

    if precipitation == 0:
        rain_note = "No rainfall – irrigation required."
    elif precipitation < 10:
        rain_note = f"Light rain ({precipitation:.1f} mm) – supplemental irrigation advised."
    elif precipitation < 25:
        rain_note = f"Moderate rain ({precipitation:.1f} mm) – good for growth."
    else:
        rain_note = f"Heavy rain ({precipitation:.1f} mm) – ensure proper drainage."

    return crops, temp_note, hum_note, rain_note

def get_seasonal_crops():
    month = datetime.now().month
    if month in [12, 1, 2]:
        return ["Wheat", "Barley", "Rye", "Oats"], "Winter"
    elif month in [3, 4, 5]:
        return ["Corn", "Soybeans", "Rice", "Sunflower"], "Spring"
    elif month in [6, 7, 8]:
        return ["Cotton", "Sugar Cane", "Millet", "Sorghum"], "Summer"
    else:
        return ["Alfalfa", "Hay", "Pasture Grass", "Clover"], "Year-Round"

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "CroplandQuantum API running"}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    location: Optional[str] = Form(None),
):
    """Classify a single satellite image and optionally return crop recommendations."""
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    predicted, confidence = classify_image(img)
    is_cropland = predicted == 1
    label = "Cropland" if is_cropland else "Non-Cropland"

    result = {"label": label, "is_cropland": is_cropland, "confidence": round(confidence * 100, 1)}

    # Crop & weather recommendations only when cropland with good confidence
    if is_cropland and confidence >= 0.55:
        if location and location.strip():
            lat, lon, loc_name, region = get_coordinates(location.strip())
            if lat:
                weather = get_weather_data(lat, lon)
                if weather:
                    try:
                        current = weather["current"]
                        daily = weather["daily"]
                        temp_min = daily["temperature_2m_min"][0]
                        temp_max = daily["temperature_2m_max"][0]
                        temp_avg = (temp_min + temp_max) / 2
                        humidity = current["relative_humidity_2m"]
                        precip = daily["precipitation_sum"][0] or 0
                        wind = current["wind_speed_10m"]

                        crops, temp_note, hum_note, rain_note = get_crop_recommendations(temp_avg, humidity, precip)

                        result["weather"] = {
                            "location": f"{loc_name}, {region}",
                            "temperature": round(temp_avg, 1),
                            "temp_min": round(temp_min, 1),
                            "temp_max": round(temp_max, 1),
                            "humidity": humidity,
                            "wind_speed": round(wind, 1),
                            "precipitation": round(precip, 1),
                            "temp_note": temp_note,
                            "humidity_note": hum_note,
                            "rainfall_note": rain_note,
                        }
                        result["recommended_crops"] = crops
                        result["recommendation_source"] = "weather"
                    except Exception:
                        pass
            if "recommended_crops" not in result:
                result["location_error"] = f"Could not find '{location}'"

        if "recommended_crops" not in result:
            crops, season = get_seasonal_crops()
            result["recommended_crops"] = crops
            result["season"] = season
            result["recommendation_source"] = "seasonal"

    return JSONResponse(result)


@app.post("/api/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Classify multiple satellite images."""
    results = []
    for f in files:
        data = await f.read()
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            predicted, confidence = classify_image(img)
            is_cropland = predicted == 1
            results.append({
                "filename": f.filename,
                "label": "Cropland" if is_cropland else "Non-Cropland",
                "is_cropland": is_cropland,
                "confidence": round(confidence * 100, 1),
            })
        except Exception:
            results.append({"filename": f.filename, "error": "Invalid image"})

    cropland_count = sum(1 for r in results if r.get("is_cropland"))
    avg_conf = round(
        sum(r["confidence"] for r in results if "confidence" in r) / max(len(results), 1), 1
    )
    return JSONResponse({
        "results": results,
        "total": len(results),
        "cropland_count": cropland_count,
        "non_cropland_count": len(results) - cropland_count,
        "avg_confidence": avg_conf,
    })


class WeatherRequest(BaseModel):
    location: str


@app.post("/api/weather")
def weather_analysis(body: WeatherRequest):
    """Return weather data and crop recommendations for a given location."""
    lat, lon, loc_name, region = get_coordinates(body.location)
    if not lat:
        raise HTTPException(404, f"Location '{body.location}' not found")

    weather = get_weather_data(lat, lon)
    if not weather:
        raise HTTPException(502, "Could not fetch weather data")

    current = weather["current"]
    daily = weather["daily"]
    temp_min = daily["temperature_2m_min"][0]
    temp_max = daily["temperature_2m_max"][0]
    temp_avg = (temp_min + temp_max) / 2
    humidity = current["relative_humidity_2m"]
    precip = daily["precipitation_sum"][0] or 0
    wind = current["wind_speed_10m"]

    crops, temp_note, hum_note, rain_note = get_crop_recommendations(temp_avg, humidity, precip)

    # 7-day forecast
    forecast = []
    dates = daily.get("time", [])
    maxes = daily.get("temperature_2m_max", [])
    mins = daily.get("temperature_2m_min", [])
    precips = daily.get("precipitation_sum", [])
    for i in range(min(7, len(dates))):
        forecast.append({
            "date": dates[i],
            "temp_max": round(maxes[i], 1) if i < len(maxes) else None,
            "temp_min": round(mins[i], 1) if i < len(mins) else None,
            "precipitation": round(precips[i], 1) if i < len(precips) else None,
        })

    return JSONResponse({
        "location": f"{loc_name}, {region}",
        "current": {
            "temperature": round(temp_avg, 1),
            "temp_min": round(temp_min, 1),
            "temp_max": round(temp_max, 1),
            "humidity": humidity,
            "wind_speed": round(wind, 1),
            "precipitation": round(precip, 1),
        },
        "forecast": forecast,
        "analysis": {
            "temp_note": temp_note,
            "humidity_note": hum_note,
            "rainfall_note": rain_note,
        },
        "recommended_crops": crops,
    })
