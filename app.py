import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import pytz
import joblib
import os

# Configuration
API_KEY = "bfb984eb0cd5e6bd816cc85ab42b7fd7"  # Consider using st.secrets for API keys in production
BASE_URL = "https://api.openweathermap.org/data/2.5/"
MODEL_PATH = "models.pkl"  # Changed from rain_model.pkl to store all models
DATA_PATH = "weather.csv"

# Load or train models
@st.cache_resource
def load_or_train_models():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Try to load pre-trained models
    if os.path.exists(MODEL_PATH):
        try:
            models = joblib.load(MODEL_PATH)
            # Validate loaded models
            if all(key in models for key in ['rain_model', 'temp_model', 'hum_model', 'le_wind', 'le_rain']):
                return models
        except Exception as e:
            st.warning(f"Failed to load pre-trained models: {str(e)}. Training new models...")
    
    # Train new models if loading fails
    historical_data = read_historical_data(DATA_PATH)
    if historical_data is None:
        return None
        
    x, y, le_wind, le_rain = prepare_data(historical_data)
    if x is None or y is None:
        return None
    
    # Train rain model
    rain_model = train_rain_model(x, y)
    
    # Train regression models
    x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
    temp_model = train_regression_model(x_temp, y_temp)
    hum_model = train_regression_model(x_hum, y_hum)
    
    # Save models for future use
    models = {
        'rain_model': rain_model,
        'temp_model': temp_model,
        'hum_model': hum_model,
        'le_wind': le_wind,
        'le_rain': le_rain
    }
    
    try:
        joblib.dump(models, MODEL_PATH)
    except Exception as e:
        st.warning(f"Failed to save models: {str(e)}")
    
    return models

def get_current_weather(city):
    if not city or not isinstance(city, str):
        st.error("Invalid city name")
        return None

    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises exception for 4XX/5XX errors
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

    data = response.json()

    try:
        return {
            'city': data.get('name', 'Unknown'),
            'current_temp': round(float(data['main']['temp']), 1),
            'feels_like': round(float(data['main']['feels_like']), 1),
            'temp_min': round(float(data['main']['temp_min']), 1),
            'temp_max': round(float(data['main']['temp_max']), 1),
            'humidity': int(data['main']['humidity']),
            'description': data['weather'][0]['description'].title(),
            'country': data['sys']['country'],
            'wind_gust_dir': float(data['wind'].get('deg', 0)),
            'pressure': int(data['main']['pressure']),
            'wind_speed': float(data['wind'].get('speed', 0))
        }
    except (KeyError, TypeError, ValueError) as e:
        st.error(f"Error processing weather data: {str(e)}")
        return None

def read_historical_data(filename):
    try:
        if not os.path.exists(filename):
            st.error(f"Data file '{filename}' not found.")
            return None
            
        df = pd.read_csv(filename)
        if df.empty:
            st.error("Data file is empty")
            return None
            
        return df.dropna().drop_duplicates()
    except Exception as e:
        st.error(f"Error reading data file: {str(e)}")
        return None

def prepare_data(data):
    if data is None:
        return None, None, None, None
        
    try:
        le_wind = LabelEncoder()
        le_rain = LabelEncoder()

        data = data.copy()
        
        # Check if required columns exist
        required_columns = ['WindGustDir', 'RainTomorrow', 'MinTemp', 'MaxTemp', 'Humidity', 'Pressure', 'Temp']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None, None, None, None

        data['WindGustDir'] = data['WindGustDir'].fillna('Unknown')
        data['RainTomorrow'] = data['RainTomorrow'].fillna('No')

        data['WindGustDir'] = le_wind.fit_transform(data['WindGustDir'])
        data['RainTomorrow'] = le_rain.fit_transform(data['RainTomorrow'])

        x = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'Humidity', 'Pressure', 'Temp']]
        y = data['RainTomorrow']

        return x, y, le_wind, le_rain
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None, None, None, None

def train_rain_model(x, y):
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        
        # Calculate and display accuracy
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Rain model trained successfully (Accuracy: {accuracy:.2%})")
        return model
    except Exception as e:
        st.error(f"Error training rain model: {str(e)}")
        return None

def prepare_regression_data(data, feature):
    try:
        if feature not in data.columns:
            st.error(f"Feature '{feature}' not found in data")
            return None, None
            
        x, y = [], []
        for i in range(len(data) - 1):
            x.append(float(data[feature].iloc[i]))
            y.append(float(data[feature].iloc[i + 1]))
        return np.array(x).reshape(-1, 1), np.array(y)
    except Exception as e:
        st.error(f"Error preparing regression data for {feature}: {str(e)}")
        return None, None

def train_regression_model(x, y):
    try:
        if x is None or y is None:
            return None
            
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x, y)
        return model
    except Exception as e:
        st.error(f"Error training regression model: {str(e)}")
        return None

def predict_future(model, current_value):
    try:
        if model is None:
            return None
            
        predictions = [float(current_value)]
        for _ in range(5):
            next_value = model.predict(np.array([[predictions[-1]]]))[0]
            predictions.append(float(next_value))
        return predictions[1:]
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def get_compass_direction(wind_deg, le_wind):
    try:
        wind_deg = float(wind_deg) % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75), ("N", 348.75, 360)
        ]
        compass_direction = next((point for point, start, end in compass_points if start <= wind_deg < end), "N")
        
        if le_wind is not None and hasattr(le_wind, 'classes_') and compass_direction in le_wind.classes_:
            compass_direction_encoded = le_wind.transform([compass_direction])[0]
        else:
            compass_direction_encoded = -1
            
        return compass_direction, compass_direction_encoded
    except Exception as e:
        st.error(f"Error determining wind direction: {str(e)}")
        return "Unknown", -1

def display_weather(current_weather, rain_prediction, future_temp, future_humidity):
    if not current_weather:
        return
        
    # Generate future times
    timezone = pytz.timezone('UTC')
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    future_times = [(next_hour + timedelta(hours=i)).strftime("%H:%M") for i in range(5)]
    
    # Display results
    st.subheader(f"Current Weather in {current_weather['city']}, {current_weather['country']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Temperature", f"{current_weather['current_temp']}°C", f"Feels like {current_weather['feels_like']}°C")
        st.metric("Humidity", f"{current_weather['humidity']}%")
        st.metric("Pressure", f"{current_weather['pressure']} hPa")
        
    with col2:
        st.metric("Weather", current_weather['description'])
        st.metric("Wind", f"{current_weather['wind_speed']} m/s {get_compass_direction(current_weather['wind_gust_dir'], None)[0]}")
        st.metric("Rain Prediction", "Likely" if rain_prediction else "Unlikely")
    
    # Future predictions
    st.subheader("Future Predictions")
    
    tab1, tab2 = st.tabs(["Temperature", "Humidity"])
    
    with tab1:
        st.write("Predicted temperature for the next 5 hours:")
        if future_temp:
            for time, temp in zip(future_times, future_temp):
                st.write(f"{time}: {round(temp, 1)}°C")
        else:
            st.warning("Temperature predictions unavailable")
            
    with tab2:
        st.write("Predicted humidity for the next 5 hours:")
        if future_humidity:
            for time, humidity in zip(future_times, future_humidity):
                st.write(f"{time}: {round(humidity, 1)}%")
        else:
            st.warning("Humidity predictions unavailable")

def main():
    st.set_page_config(
        page_title="Weather Prediction App", 
        page_icon="⛅",
        layout="wide"
    )
    
    st.title("🌦️ Weather Prediction App")
    st.write("Get current weather and predictions for any city worldwide")
    
    # Initialize session state
    if 'weather_data' not in st.session_state:
        st.session_state.weather_data = None
    
    # Sidebar for additional controls
    with st.sidebar:
        st.header("Settings")
        use_cached = st.checkbox("Use cached models", value=True)
        refresh_data = st.button("Refresh Data")
    
    # Load or train models
    if refresh_data or not use_cached:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
    
    with st.spinner("Loading weather models..."):
        models = load_or_train_models()
    
    if models is None:
        st.error("Failed to initialize weather models. Please check the data files.")
        return
    
    # Main input form
    with st.form("weather_form"):
        city = st.text_input("Enter city name:", "London")
        submitted = st.form_submit_button("Get Weather")
    
    if submitted:
        if not city:
            st.warning("Please enter a city name")
            return
            
        with st.spinner("Fetching weather data..."):
            current_weather = get_current_weather(city)
            if current_weather is None:
                return
            
            # Get wind direction
            compass_direction, compass_direction_encoded = get_compass_direction(
                current_weather['wind_gust_dir'], models['le_wind']
            )
            
            # Prepare current data for prediction
            current_data = pd.DataFrame([{
                'MinTemp': current_weather['temp_min'],
                'MaxTemp': current_weather['temp_max'],
                'WindGustDir': compass_direction_encoded,
                'Humidity': current_weather['humidity'],
                'Pressure': current_weather['pressure'],
                'Temp': current_weather['current_temp']
            }])
            
            # Make predictions
            rain_prediction = bool(models['rain_model'].predict(current_data)[0]) if models['rain_model'] else False
            future_temp = predict_future(models['temp_model'], current_weather['current_temp']) if models['temp_model'] else None
            future_humidity = predict_future(models['hum_model'], current_weather['humidity']) if models['hum_model'] else None
            
            # Store in session state
            st.session_state.weather_data = {
                'current': current_weather,
                'predictions': {
                    'rain': rain_prediction,
                    'temp': future_temp,
                    'humidity': future_humidity
                }
            }
    
    # Display results if available
    if st.session_state.weather_data:
        display_weather(
            st.session_state.weather_data['current'], 
            st.session_state.weather_data['predictions']['rain'],
            st.session_state.weather_data['predictions']['temp'],
            st.session_state.weather_data['predictions']['humidity']
        )

if __name__ == "__main__":
    main()


