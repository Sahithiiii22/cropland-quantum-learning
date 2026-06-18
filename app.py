import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import os
import random
import requests
import json
from datetime import datetime, timedelta

# =========================
# Crop Suggestion Database
# =========================
CROP_SUGGESTIONS = {
    "Winter": {
        "crops": ["Wheat", "Barley", "Rye", "Oats"],
        "description": "Perfect for winter planting",
        "emoji": "🌾"
    },
    "Spring": {
        "crops": ["Corn", "Soybeans", "Rice", "Sunflower"],
        "description": "Ideal spring planting season",
        "emoji": "🌱"
    },
    "Summer": {
        "crops": ["Cotton", "Sugar Cane", "Millet"],
        "description": "Heat-tolerant crops for summer",
        "emoji": "☀️"
    },
    "Year-Round": {
        "crops": ["Alfalfa", "Hay", "Pasture Grass", "Clover"],
        "description": "Suitable for livestock feed",
        "emoji": "🌿"
    }
}

# Indian States and Major Cities Coordinates
INDIA_LOCATIONS = {
    "Andhra Pradesh": (15.9129, 78.6675),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 73.8243),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 77.0745),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9375, 78.6553),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1815, 92.9789),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.5204, 74.3587),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (24.5155, 88.3289),
    # Major Cities
    "Hyderabad": (17.3850, 78.4867),
    "Bangalore": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Pune": (18.5204, 73.8567),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
}

def get_crop_suggestion():
    """Get seasonal crop suggestion"""
    import datetime
    month = datetime.datetime.now().month
    
    if month in [12, 1, 2]:
        season = "Winter"
    elif month in [3, 4, 5]:
        season = "Spring"
    elif month in [6, 7, 8]:
        season = "Summer"
    else:
        season = "Year-Round"
    
    return season

# =========================
# Weather API Integration
# =========================
@st.cache_data(ttl=3600)
def get_coordinates(location_name):
    """Get coordinates for a location"""
    location_name_clean = location_name.strip().title()
    
    # Check INDIA_LOCATIONS first (case-insensitive)
    for key, coords in INDIA_LOCATIONS.items():
        if key.lower() == location_name_clean.lower():
            return coords[0], coords[1], key, "India"
    
    # Fallback to API
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": location_name,
            "country": "IN",
            "language": "en",
            "limit": 1
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data.get("results"):
            result = data["results"][0]
            return result["latitude"], result["longitude"], result["name"], result.get("admin1", "India")
        else:
            return None, None, None, None
    except:
        return None, None, None, None

@st.cache_data(ttl=86400)
def get_weather_data(latitude, longitude):
    """Fetch real-time weather data from Open-Meteo API"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto"
        }
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except:
        return None

def get_weather_based_crops(temp_avg, humidity, precipitation):
    """Get crop recommendations based on temperature"""
    temp_rec = ""
    humidity_rec = ""
    rainfall_rec = ""
    
    # Temperature-based recommendations
    if temp_avg < 10:
        crops = ["Wheat", "Barley", "Rye", "Oats"]
        temp_rec = f"Cold conditions ({temp_avg:.1f}°C). Excellent for winter crops."
    elif 10 <= temp_avg < 18:
        crops = ["Alfalfa", "Clover", "Pasture Grass", "Timothy Hay"]
        temp_rec = f"Cool season ({temp_avg:.1f}°C). Ideal for forage crops."
    elif 18 <= temp_avg < 24:
        crops = ["Corn", "Soybeans", "Rice", "Sunflower"]
        temp_rec = f"Warm season ({temp_avg:.1f}°C). Perfect for summer crops."
    else:
        crops = ["Cotton", "Sugar Cane", "Millet", "Sorghum"]
        temp_rec = f"Hot season ({temp_avg:.1f}°C). Best for heat-tolerant crops."
    
    # Humidity adjustments
    if humidity < 30:
        humidity_rec = f"Very dry ({humidity}%). Focus on drought-tolerant crops."
        crops = [c for c in crops if c in ["Wheat", "Millet", "Sorghum", "Barley", "Rye", "Cotton"]]
    elif 30 <= humidity < 50:
        humidity_rec = f"Moderate humidity ({humidity}%). Good for most crops."
    elif 50 <= humidity < 70:
        humidity_rec = f"High humidity ({humidity}%). Consider crop rotation."
    else:
        humidity_rec = f"Very humid ({humidity}%). Watch for fungal diseases."
    
    # Rainfall analysis
    if precipitation == 0:
        rainfall_rec = "No rainfall. Irrigation required for all crops."
    elif precipitation < 10:
        rainfall_rec = f"Light rain ({precipitation:.1f}mm). May need irrigation."
    elif precipitation < 25:
        rainfall_rec = f"Moderate rain ({precipitation:.1f}mm). Good for growth."
    else:
        rainfall_rec = f"Heavy rain ({precipitation:.1f}mm). Ensure proper drainage."
    
    return crops, temp_rec, humidity_rec, rainfall_rec

# =========================
# Streamlit Config
# =========================
st.set_page_config(
    page_title="Cropland Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS for Professional Look
# =========================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f0f7 100%);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #003f87 0%, #0071bc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%);
        border-left: 5px solid #2d7a1f;
        box-shadow: 0 2px 8px rgba(0, 63, 135, 0.1);
    }
    .crop-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #fff5e1 0%, #fffacd 100%);
        border: 1px solid #d4a574;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s;
        min-height: 200px;
        width: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .crop-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .crop-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    .crop-title {
        font-size: 0.95rem;
        font-weight: bold;
        color: #2d7a1f;
        margin: 0;
    }
    .recommendations-container {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f0fdf4 0%, #f5fff3 100%);
        border: 2px solid #86efac;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-top: 1rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFACD 0%, #FFFFE0 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# =========================
# Sidebar Navigation
# =========================
st.sidebar.markdown("## 📋 Navigation")
option = st.sidebar.radio(
    "Select an option:",
    ["🖼️ Upload Satellite Image", "📊 Upload Dataset", "🌤️ Weather Analysis"],
    label_visibility="collapsed"
)

# =========================
# Main Content
# =========================
st.markdown('<div class="header-title">🌾 Cropland Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Advanced Satellite Image Classification</div>', unsafe_allow_html=True)

# =========================
# Option 1: Upload Satellite Image
# =========================
if option == "🖼️ Upload Satellite Image":
    st.markdown("#### Image Upload & Location")
    
    col_upload, col_location = st.columns([2, 1], gap="medium")
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a satellite image",
            type=["jpg", "png", "jpeg"],
            key="single_image"
        )
    
    with col_location:
        location_input = st.text_input(
            "Your location (optional)",
            placeholder="e.g., Telangana, Punjab, Hyderabad",
            key="location_input"
        )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1.2, 1.8], gap="large")
        
        with col1:
            st.markdown("#### Original Image")
            st.image(image, caption="Uploaded Satellite Image", width=250)
        
        with col2:
            st.markdown("#### Prediction Results")
            
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
            
            img = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted = predicted.item()
            
            if predicted == 1:
                result = "🌾 Cropland"
                color = "#2d7a1f"
                is_cropland = True
            else:
                result = "🏙️ Non-Cropland"
                color = "#1F77B4"
                is_cropland = False
            
            st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: {color}; margin: 0;">{result}</h3>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0;">
                        Confidence: <strong>{confidence*100:.1f}%</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show crop suggestions if it's cropland with good confidence
            if is_cropland and confidence >= 0.55:
                recommended_crops = None
                weather_analysis = None
                weather_data = None  # Initialize to None
                
                # If location is provided, fetch weather data
                if location_input and location_input.strip():
                    with st.spinner(f"🔍 Searching weather for {location_input}..."):
                        latitude, longitude, location_name, region = get_coordinates(location_input.strip())
                    
                    if latitude and longitude:
                        with st.spinner("📡 Fetching weather data..."):
                            weather_data = get_weather_data(latitude, longitude)
                        
                        if weather_data:
                            try:
                                current = weather_data['current']
                                daily = weather_data['daily']
                                
                                temp_min = daily['temperature_2m_min'][0]
                                temp_max = daily['temperature_2m_max'][0]
                                temp_avg = (temp_min + temp_max) / 2
                                
                                humidity = current['relative_humidity_2m']
                                precipitation = daily['precipitation_sum'][0] if daily['precipitation_sum'][0] else 0
                                wind_speed = current['wind_speed_10m']
                                
                                # Get crop recommendations based on REAL weather data
                                recommended_crops, temp_rec, humidity_rec, rainfall_rec = get_weather_based_crops(
                                    temp_avg, humidity, precipitation
                                )
                                
                                # Store info for display
                                weather_analysis = {
                                    "temp": temp_rec,
                                    "humidity": humidity_rec,
                                    "rainfall": rainfall_rec
                                }
                                
                                st.success(f"✅ Found: {location_name}, {region} | Data source: Open-Meteo API")
                                
                            except Exception as e:
                                st.error(f"Error processing weather data: {str(e)}")
                                recommended_crops = None
                        else:
                            st.warning("⚠️ Could not fetch weather data. Using seasonal recommendations.")
                    else:
                        st.error(f"❌ Location '{location_input}' not found. Try:\n- City names (e.g., 'Hyderabad', 'Bangalore')\n- State/Region names\n- Full address")
                
                # If no weather data, use seasonal recommendations
                if recommended_crops is None:
                    season = get_crop_suggestion()
                    season_data = CROP_SUGGESTIONS[season]
                    recommended_crops = season_data['crops']
                    weather_analysis = None
                
                # Display centered weather data, analysis, and suitable crops sections
                st.markdown("---")
                
                # --- Centered Weather & Analysis Section (below image) ---
                def render_centered_sections(weather_data=None, analysis_text=None, crops_list=None):
                    weather_html = ""
                    analysis_html = ""
                    crops_html = ""

                    if weather_data:
                        # Basic current weather display
                        current = weather_data.get("current_weather") or weather_data.get("current") or {}
                        temp = current.get("temperature") or current.get("temperature_2m") or "--"
                        wind = current.get("windspeed") or current.get("wind_speed_10m") or "--"
                        weather_html = f"""<div class="crop-card"><div><h4 class="crop-title">🌦️ Current Weather</h4><p style="margin:0.25rem 0;">Temperature: <strong>{temp}°C</strong></p><p style="margin:0.25rem 0;">Wind: <strong>{wind} m/s</strong></p></div></div>"""

                    if analysis_text:
                        analysis_html = f"""<div class="crop-card"><div><h4 class="crop-title">📊 Weather Analysis</h4><p style="margin:0.25rem 0; font-size:0.9rem;">{analysis_text}</p></div></div>"""

                    if crops_list and len(crops_list) > 0:
                        crops_text = ', '.join(str(c) for c in crops_list)
                        crops_html = f"""<div class="crop-card"><div><h4 class="crop-title">🌾 Suitable Crops</h4><p style="margin:0.25rem 0;">{crops_text}</p></div></div>"""

                    combined = f"""
                    <div style="display:flex; gap:2.5rem; justify-content:center; align-items:center; margin:3rem auto; flex-wrap:wrap; max-width:1400px; width:100%;">
                        {weather_html}{analysis_html}{crops_html}
                    </div>
                    """

                    st.markdown(combined, unsafe_allow_html=True)

                # If user provided a location, fetch and display weather and crop hints
                if location_input:
                    try:
                        coords = get_coordinates(location_input)
                        if coords:
                            latitude, longitude, location_name, region = coords
                            weather = get_weather_data(latitude, longitude)

                            # Prepare analysis and suitable crops using existing helper
                            if weather and weather.get("daily"):
                                # compute simple averages from daily arrays if present
                                temps = weather.get("daily", {}).get("temperature_2m_max", [])
                                prec = weather.get("daily", {}).get("precipitation_sum", [])
                                avg_temp = None
                                total_prec = None
                                try:
                                    if temps:
                                        avg_temp = sum(temps) / len(temps)
                                    if prec:
                                        total_prec = sum(prec)
                                except:
                                    avg_temp = None
                                    total_prec = None

                                # fallback current values
                                current = weather.get("current_weather") or weather.get("current") or {}
                                humidity = 50
                                if "relative_humidity_2m" in current:
                                    humidity = current["relative_humidity_2m"]
                                analysis_text = ""
                                crops_rec, temp_rec, hum_rec, rain_rec = get_weather_based_crops(avg_temp or 0, humidity if isinstance(humidity, (int, float)) else 50, total_prec or 0)
                                analysis_text = f"{temp_rec} {hum_rec} {rain_rec}"

                                render_centered_sections(weather_data=weather, analysis_text=analysis_text, crops_list=crops_rec)
                            else:
                                render_centered_sections(weather_data=None, analysis_text="Weather not available.", crops_list=crops_list if crops_list else recommended_crops)
                        else:
                            render_centered_sections(weather_data=None, analysis_text="Location not found.", crops_list=recommended_crops)
                    except Exception as e:
                        render_centered_sections(weather_data=None, analysis_text="Error fetching weather.", crops_list=recommended_crops)
                else:
                    # No location provided - just show seasonal crops
                    render_centered_sections(weather_data=None, analysis_text="Seasonal Recommendation", crops_list=recommended_crops)
                
            elif is_cropland and confidence < 0.55:
                st.warning("⚠️ Cropland detected but confidence is low (< 55%). Enter location for accurate recommendations.")
            else:
                st.info("ℹ️ This appears to be non-cropland. Crop recommendations not applicable.")

# =========================
# Option 2: Upload Dataset
# =========================
elif option == "📊 Upload Dataset":
    st.markdown("#### Dataset Upload")
    uploaded_files = st.file_uploader(
        "Choose satellite images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key="dataset_images"
    )
    
    if uploaded_files:
        st.markdown("---")
        st.markdown(f"### Processing {len(uploaded_files)} images...")
        
        results = []
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
            
            img = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted = predicted.item()
            
            results.append({
                "Image": uploaded_file.name,
                "Prediction": "Cropland" if predicted == 1 else "Non-Cropland",
                "Confidence": f"{confidence*100:.1f}%"
            })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        st.markdown("---")
        st.markdown("### Results Summary")
        
        st.dataframe(results, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        cropland_count = sum(1 for r in results if r["Prediction"] == "Cropland")
        non_cropland_count = len(results) - cropland_count
        avg_confidence = sum(float(r["Confidence"].rstrip('%')) for r in results) / len(results)
        
        with col1:
            st.metric("Total Images", len(results))
        with col2:
            st.metric("🌾 Cropland", cropland_count)
        with col3:
            st.metric("🏙️ Non-Cropland", non_cropland_count)
        
        st.metric("Average Confidence", f"{avg_confidence:.1f}%")

# =========================
# Option 3: Weather Analysis
# =========================
elif option == "🌤️ Weather Analysis":
    st.markdown("#### Enter Location Details")
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        city_name = st.text_input(
            "Enter your city/location",
            placeholder="e.g., New Delhi, Punjab, Mumbai",
            key="city_input"
        )
    
    with col2:
        search_btn = st.button("🔍 Search Weather", use_container_width=True)
    
    if search_btn and city_name:
        latitude, longitude, location_name, region = get_coordinates(city_name)
        
        if latitude and longitude:
            st.success(f"✅ Location found: **{location_name}, {region}**")
            
            # Fetch weather data
            weather_data = get_weather_data(latitude, longitude)
            
            if weather_data:
                # Extract current weather
                current = weather_data['current']
                daily = weather_data['daily']
                
                # Display current weather summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🌡️ Temperature", f"{current['temperature_2m']}°C")
                with col2:
                    st.metric("💧 Humidity", f"{current['relative_humidity_2m']}%")
                with col3:
                    st.metric("💨 Wind Speed", f"{current['wind_speed_10m']} km/h")
                with col4:
                    st.metric("📊 Temp Range", f"{daily['temperature_2m_min'][0]:.0f}°-{daily['temperature_2m_max'][0]:.0f}°")
                
                st.markdown("---")
                
                # Weather-based crop recommendations
                temp_min = daily['temperature_2m_min'][0]
                temp_max = daily['temperature_2m_max'][0]
                humidity = current['relative_humidity_2m']
                precipitation = daily['precipitation_sum'][0] if daily['precipitation_sum'][0] else 0
                
                recommended_crops, temp_note, humidity_note, precip_note = get_weather_based_crops(
                    (temp_min + temp_max) / 2, humidity, precipitation
                )
                
                st.markdown("#### 🌱 Weather-Based Crop Recommendations")
                
                col1, col2, col3, col4 = st.columns(4, gap="small")
                
                with col1:
                    st.info(f"**Temperature**\n\n{temp_note}")
                with col2:
                    st.info(f"**Humidity**\n\n{humidity_note}")
                with col3:
                    st.info(f"**Rainfall**\n\n{precip_note}")
                with col4:
                    st.success(f"**Forecast**\n\nNext 7 days outlook available")
                
                st.markdown("#### 🌾 Suitable Crops")
                crop_cols = st.columns(4, gap="small")
                
                for idx, crop in enumerate(recommended_crops):
                    with crop_cols[idx % 4]:
                        st.markdown(f"""
                        <div class="crop-card">
                            <div class="crop-title">✓ {crop}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.error(f"❌ Location '{city_name}' not found. Please try another city.")