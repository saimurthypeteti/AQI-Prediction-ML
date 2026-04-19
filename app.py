from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# =========================
# LOAD DATASETS
# =========================
df1 = pd.read_csv('india_city_aqi_2015_2023.csv')
df2 = pd.read_csv('Air_Quality_Dataset.csv')

# =========================
# STANDARDIZE COLUMNS
# =========================
df1 = df1.rename(columns={'pm2.5': 'pm25'})

df2 = df2.rename(columns={
    'PM2.5': 'pm25',
    'PM10': 'pm10',
    'NO2': 'no2',
    'SO2': 'so2',
    'CO': 'co',
    'O3': 'o3',
    'AQI': 'aqi'
})

cols = ['co','no2','so2','o3','pm25','pm10','aqi']

df1 = df1[cols]
df2 = df2[cols]

# =========================
# MERGE DATASETS
# =========================
df = pd.concat([df1, df2], ignore_index=True)

# =========================
# CLEAN DATA
# =========================
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# =========================
# REMOVE OUTLIERS
# =========================
for col in ['co','no2','so2','o3','pm25','pm10']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]

# =========================
# FEATURES
# =========================
X = df[['co','no2','so2','o3','pm25','pm10']]
y = np.log1p(df['aqi'])

# =========================
# SPLIT (CORRECT ORDER)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# SCALE (AFTER SPLIT)
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=220,
    max_depth=10,
    min_samples_split=12,
    min_samples_leaf=6,
    max_features='sqrt',
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# CROSS VALIDATION (CORRECT)
# =========================
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("CV R2:", cv_scores.mean())

# =========================
# TEST EVALUATION
# =========================
test_pred = model.predict(X_test)

test_pred_real = np.expm1(test_pred)
y_test_real = np.expm1(y_test)

test_mse = mean_squared_error(y_test_real, test_pred_real)
test_r2 = r2_score(y_test_real, test_pred_real)

print("Test MSE:", test_mse)
print("Test R2:", test_r2)

# =========================
# TRAIN EVALUATION (CHECK OVERFITTING)
# =========================
train_pred = model.predict(X_train)

train_pred_real = np.expm1(train_pred)
y_train_real = np.expm1(y_train)

train_mse = mean_squared_error(y_train_real, train_pred_real)
train_r2 = r2_score(y_train_real, train_pred_real)

print("Train MSE:", train_mse)
print("Train R2:", train_r2)

# =========================
# FEATURE IMPORTANCE
# =========================
importance = dict(zip(['CO','NO2','SO2','O3','PM2.5','PM10'], model.feature_importances_))

# =========================
# CATEGORY FUNCTION
# =========================
def get_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():

    values = [
        float(request.form.get('CO')),
        float(request.form.get('NO2')),
        float(request.form.get('SO2')),
        float(request.form.get('O3')),
        float(request.form.get('PM25')),
        float(request.form.get('PM10'))
    ]

    scaled = scaler.transform([values])
    ml_aqi = np.expm1(model.predict(scaled)[0])

    category = get_category(ml_aqi)

    return render_template(
        'index.html',
        prediction=round(ml_aqi, 2),
        category=category,
        mse=round(test_mse, 2),
        r2=round(test_r2, 2),
        importance=importance,
        values=values
    )

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)