import pandas as pd
import numpy as np
import glob
import os
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#LOAD AND MERGING DATA
files = glob.glob("D:/CAREERS/SUM_QUALITY/*.xlsm")
all_data = []

for file in files:
    filename = os.path.basename(file)
    date_str = filename.replace("SUM_QUALITY ", "").replace(".xlsm", "")
    try:
        df = pd.read_excel(file, sheet_name='sum_PSY', header=6, engine='openpyxl')
        df['date_raw'] = date_str
        all_data.append(df)
    except Exception as e:
        print(f"Error reading {filename}: {e}")

data = pd.concat(all_data, ignore_index=True)

#DATA CLEANING
data.columns = data.columns.str.strip().str.replace('.', '', regex=False).str.replace(' ', '_')
data = data.loc[:, ~data.columns.duplicated()]

if 'CV_(ADB)' in data.columns:
    data = data.rename(columns={'CV_(ADB)': 'CV_ADB'})

required_cols = ['RF', 'INV', 'TM', 'M', 'CV_ADB', 'date_raw']
data_clean = data[[c for c in required_cols if c in data.columns]].copy()

bulan_map = {'Mei': 'May', 'Agu': 'Aug', 'Okt': 'Oct', 'Des': 'Dec'}
data_clean['date_str'] = data_clean['date_raw'].replace(bulan_map, regex=True)
data_clean['date_temp'] = pd.to_datetime(data_clean['date_str'], format='%d %b', errors='coerce')

data_clean['year'] = np.where(data_clean['date_temp'].dt.month >= 10, 2025, 2026)
data_clean['date'] = pd.to_datetime(
    data_clean['date_temp'].dt.strftime('%d-%m-') + data_clean['year'].astype(str),
    format='%d-%m-%Y'
)

for col in ['INV', 'TM', 'M', 'CV_ADB']:
    data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')

data_clean = data_clean.dropna(subset=['TM', 'date'])
data_clean = data_clean[(data_clean['INV'] > 0) & (data_clean['TM'] > 5) & (data_clean['RF'] != 'TOTAL')]
data_clean['RF'] = data_clean['RF'].str.replace(' ', '')
data_clean = data_clean.sort_values(['RF', 'date'])


#FEATURE ENGINEERING
data_clean['IM'] = data_clean['M']
data_clean['SM'] = data_clean['TM'] - data_clean['IM']

data_clean['lag_SM'] = data_clean.groupby('RF')['SM'].shift(1)
data_clean['lag2_SM'] = data_clean.groupby('RF')['SM'].shift(2)
data_clean['inv_diff'] = data_clean.groupby('RF')['INV'].diff().fillna(0)
data_clean['reset_flag'] = (data_clean['inv_diff'] > 0).astype(int)
data_clean['storage_group'] = data_clean.groupby('RF')['reset_flag'].cumsum()
data_clean['storage_days'] = data_clean.groupby(['RF', 'storage_group']).cumcount()


data_clean = data_clean.drop(columns=['reset_flag', 'storage_group'])

#GET WEATHER DATA
def get_weather_history(start, end, lat, lon):
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR,T2M,RH2M,WS2M&community=AG&start={start}&end={end}&latitude={lat}&longitude={lon}&format=JSON"
    res = requests.get(url).json()
    p = res['properties']['parameter']
    return pd.DataFrame({
        'date': pd.to_datetime(list(p['PRECTOTCORR'].keys())),
        'rainfall': list(p['PRECTOTCORR'].values()),
        'temp': list(p['T2M'].values()),
        'humidity': list(p['RH2M'].values()),
        'wind': list(p['WS2M'].values())
    })

weather = get_weather_history("20251001", "20260409", 0.13, 117.5)
data_model = data_clean.merge(weather, on='date', how='left')
data_model = data_model.loc[:, ~data_model.columns.duplicated()]

weather_cols = ['rainfall', 'temp', 'humidity', 'wind']

data_model[weather_cols] = data_model.groupby('RF')[weather_cols].transform(
    lambda x: x.ffill().bfill()
)

data_model[weather_cols] = data_model[weather_cols].fillna(0)

#PHYSICS FEATURE
data_model['rain_3d'] = data_model.groupby('RF')['rainfall'].transform(lambda x: x.rolling(3).sum())
data_model['rain_7d'] = data_model.groupby('RF')['rainfall'].transform(lambda x: x.rolling(7).sum())

data_model['drying_index'] = ((100 - data_model['humidity']) / 100 * np.exp(0.06 * data_model['temp']) * (1 + 0.1 * data_model['wind']))
data_model['effective_rain'] = data_model['rainfall'] * (1 - np.exp(-0.2 * data_model['storage_days']))
data_model['net_moisture'] = data_model['effective_rain'] - (data_model['drying_index'] * 0.5)

data_model['CV_AR_actual'] = data_model['CV_ADB'] * ((100 - data_model['TM']) / (100 - data_model['IM']))

data_model = data_model.dropna(subset=['SM', 'lag_SM', 'lag2_SM', 'net_moisture'])

data_model = data_model.sort_values('date')

#MODEL TRAINING
features = ['lag_SM', 'lag2_SM', 'rainfall', 'rain_3d', 'rain_7d', 'temp', 'humidity', 'wind', 'drying_index', 'storage_days', 'effective_rain', 'net_moisture']
X = data_model[features]
y = data_model['SM']

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))

print(f"--- Model Trained ---")
print(f"Model RMSE: {rmse:.2f}")

#FORECAST ENGINE
def get_forecast(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean,windspeed_10m_max&timezone=Asia%2FBangkok"
    res = requests.get(url).json()
    return pd.DataFrame({
        'date': pd.to_datetime(res['daily']['time']),
        'rainfall': res['daily']['precipitation_sum'],
        'temp': res['daily']['temperature_2m_mean'],
        'humidity': res['daily']['relative_humidity_2m_mean'],
        'wind': res['daily']['windspeed_10m_max']
    })

forecast_weather = get_forecast(0.13, 117.5)
last_actuals = data_model.sort_values('date').groupby('RF').last().reset_index()

psy_rf_list = [
    'RF24-1','RF24-2-A','RF24-2-B','RF25-1','RF25-2-A','RF25-2-B',
    'RF25-3-A','RF25-3-B','RF42-2','RF42-4A','RF42-4B','RF42-5A',
    'RF42-5B','RF42-6A','RF42-6B','TC24-1-A','TC24-1-B','TC24-2-A',
    'TC24-2-B','TC25-1-A','TC25-1-B','TC25-2-A','TC25-2-B','TC42-1A',
    'TC42-1B','TC42-2A','TC42-2B'
]

last_actuals = last_actuals[last_actuals['RF'].isin(psy_rf_list)]

forecast_results = []

for rf in last_actuals['RF'].unique():
    current_state = last_actuals[last_actuals['RF'] == rf].iloc[0].copy()
    rain_hist = list(data_model[data_model['RF'] == rf]['rainfall'].tail(7))

    for i in range(len(forecast_weather)):
        f_day = forecast_weather.iloc[i]
        rain_hist.append(f_day['rainfall'])

        row = current_state.copy()
        row['date'] = f_day['date']
        row['rainfall'] = f_day['rainfall']
        row['temp'] = f_day['temp']
        row['humidity'] = f_day['humidity']
        row['wind'] = f_day['wind']

        row['rain_3d'] = sum(rain_hist[-3:])
        row['rain_7d'] = sum(rain_hist[-7:])
        row['storage_days'] += 1
        row['inv_diff'] = 0

        row['drying_index'] = ((100 - row['humidity']) / 100 * np.exp(0.06 * row['temp']) * (1 + 0.1 * row['wind']))
        row['effective_rain'] = row['rainfall'] * (1 - np.exp(-0.2 * row['storage_days']))
        row['net_moisture'] = row['effective_rain'] - (row['drying_index'] * 0.5)

        X_pred = pd.DataFrame([row[features]])
        pred_sm = model.predict(X_pred)[0]

        row['lag2_SM'] = row['lag_SM']
        row['SM'] = np.clip(pred_sm, 0.5, 25)
        row['lag_SM'] = row['SM']
        row['TM'] = row['IM'] + row['SM']

        row['CV_AR_pred'] = row['CV_ADB'] * ((100 - row['TM']) / (100 - row['IM']))

        if row['CV_AR_pred'] < 5300:
            row['risk'] = 'HIGH RISK'
        elif row['CV_AR_pred'] < 5400:
            row['risk'] = 'MEDIUM'
        else:
            row['risk'] = 'SAFE'

        forecast_results.append(row)
        current_state = row


#FINALE
forecast_df = pd.DataFrame(forecast_results)

final_columns = [
    'RF', 'INV', 'TM', 'M', 'CV_ADB', 'date', 'IM', 'SM', 'lag_SM', 
    'inv_diff', 'storage_days', 'rainfall', 'temp', 'humidity', 'wind', 
    'rain_3d', 'rain_7d', 'effective_rain', 'drying_index', 'net_moisture', 
    'CV_AR_actual', 'CV_AR_pred', 'risk'
]

forecast_df = forecast_df[final_columns]
forecast_df.to_csv("D:/CAREERS/RISK/forecast3.csv", index=False)
print(f"Berhasil! File disimpan di D:/CAREERS/RISK/forecast3.csv dengan {len(forecast_df['RF'].unique())} stockpile.")