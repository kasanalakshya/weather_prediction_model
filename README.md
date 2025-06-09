 Weather Prediction Model
This project predicts:

 Rainfall (Yes/No)

 Temperature (°C)

 Humidity (%)

It uses Machine Learning (Random Forest models) trained on historical weather data, and fetches live weather data using the OpenWeatherMap API.

 Features
Predict whether it will rain tomorrow based on inputs like temperature, pressure, humidity, and wind direction.

Forecast temperature and humidity using regression models.

Fetch and display live weather data from any city in the world.

Built with Streamlit for an interactive web interface.

 ML Models Used
RandomForestClassifier: For Rain Prediction (RainTomorrow)

RandomForestRegressor: For predicting:

Temperature (Temp)

Humidity (Humidity)

LabelEncoder: To encode categorical features like wind direction and rain labels.

 Project Structure

 weather_prediction_model/
│
 weather.csv              # Historical weather data
models.pkl               # Saved ML models (rain, temp, humidity)
 app.py                   # Streamlit application
 requirements.txt         # Required Python packages
 README.md                # Project description
🔑 API Used
OpenWeatherMap API

Used for real-time city weather

Make sure to replace API_KEY in the code with your own key

 How It Works
Load or Train Models:
If models.pkl exists, it loads models. Otherwise, it trains them using weather.csv.

Model Training:

train_rain_model() trains a classifier using wind, humidity, pressure, etc.

train_regression_model() trains regressors for temperature and humidity.

Weather API Fetching:
get_current_weather(city) fetches live weather from the OpenWeather API.

Streamlit Frontend:
The user can input a city name, view current weather, and see predictions.

 Installation
bash
Copy
Edit
git clone https://github.com/kasanalakshya/weather_prediction_model.git
cd weather_prediction_model
pip install -r requirements.txt
streamlit run app.py
 Sample Output
Rain Prediction Accuracy: ~XX% (based on your dataset)

Temp/Humidity Forecasts: Numerical values displayed with live data

 Tech Stack
Python

Streamlit

Pandas / NumPy

Scikit-learn

OpenWeatherMap API


