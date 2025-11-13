from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import ta
from sklearn.linear_model import LinearRegression
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# ---------------------------
# 1️⃣ Load and prepare data
# ---------------------------
df = pd.read_csv("apple_1984_2024.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Compute indicators
df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
macd = ta.trend.MACD(df['Close'])
df['MACD'] = macd.macd()
df['Daily_Change'] = df['Close'].pct_change() * 100
df.dropna(inplace=True)

# ---------------------------
# 2️⃣ Fuzzy logic setup
# ---------------------------
rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
macd_in = ctrl.Antecedent(np.arange(-10, 10, 1), 'macd')
daily_change = ctrl.Antecedent(np.arange(-5, 5.1, 0.1), 'daily_change')
trend = ctrl.Consequent(np.arange(0, 101, 1), 'trend')

rsi['low'] = fuzz.trimf(rsi.universe, [0, 0, 30])
rsi['neutral'] = fuzz.trimf(rsi.universe, [20, 50, 80])
rsi['high'] = fuzz.trimf(rsi.universe, [70, 100, 100])

macd_in['bearish'] = fuzz.trimf(macd_in.universe, [-10, -10, 0])
macd_in['neutral'] = fuzz.trimf(macd_in.universe, [-5, 0, 5])
macd_in['bullish'] = fuzz.trimf(macd_in.universe, [0, 10, 10])

daily_change['down'] = fuzz.trimf(daily_change.universe, [-5, -5, 0])
daily_change['neutral'] = fuzz.trimf(daily_change.universe, [-1, 0, 1])
daily_change['up'] = fuzz.trimf(daily_change.universe, [0, 5, 5])

trend['down'] = fuzz.trimf(trend.universe, [0, 0, 50])
trend['neutral'] = fuzz.trimf(trend.universe, [25, 50, 75])
trend['up'] = fuzz.trimf(trend.universe, [50, 100, 100])

# Fuzzy rules
rule1 = ctrl.Rule(rsi['high'] & macd_in['bullish'], trend['up'])
rule2 = ctrl.Rule(rsi['low'] & macd_in['bearish'], trend['down'])
rule3 = ctrl.Rule(rsi['neutral'] | daily_change['neutral'], trend['neutral'])
rule4 = ctrl.Rule(daily_change['up'] & rsi['neutral'], trend['up'])
rule5 = ctrl.Rule(daily_change['down'] & rsi['neutral'], trend['down'])
rule6 = ctrl.Rule(daily_change['up'] & macd_in['bullish'], trend['up'])
rule7 = ctrl.Rule(daily_change['down'] & macd_in['bearish'], trend['down'])

trend_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
trend_sim = ctrl.ControlSystemSimulation(trend_ctrl)

# Generate Trend_Score
trend_scores = []
for i in range(len(df)):
    trend_sim.input['rsi'] = float(df['RSI'].iloc[i])
    trend_sim.input['macd'] = float(df['MACD'].iloc[i])
    trend_sim.input['daily_change'] = float(df['Daily_Change'].iloc[i])
    trend_sim.compute()
    trend_scores.append(trend_sim.output['trend'])

df['Trend_Score'] = trend_scores

# ---------------------------
# 3️⃣ Train regression model
# ---------------------------
X = df[['RSI', 'MACD', 'Daily_Change', 'Trend_Score']]
y = df['Close']
model = LinearRegression()
model.fit(X, y)

# ---------------------------
# 4️⃣ Flask routes
# ---------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    rsi_val = float(request.form['rsi'])
    macd_val = float(request.form['macd'])
    daily_val = float(request.form['daily_change'])

    # Fuzzy trend prediction
    trend_sim.input['rsi'] = rsi_val
    trend_sim.input['macd'] = macd_val
    trend_sim.input['daily_change'] = daily_val
    trend_sim.compute()
    trend_score = trend_sim.output['trend']

    if trend_score < 40:
        trend_label = "Down"
    elif trend_score < 60:
        trend_label = "Neutral"
    else:
        trend_label = "Up"

    # Predict future price
    pred_price = model.predict([[rsi_val, macd_val, daily_val, trend_score]])[0]

    return render_template('index.html',
                           rsi=rsi_val,
                           macd=macd_val,
                           daily=daily_val,
                           trend=trend_label,
                           price=round(pred_price, 2))

# ---------------------------
# 5️⃣ Main entry point
# ---------------------------
if __name__ == "__main__":
    from flask_cors import CORS
    CORS(app)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
