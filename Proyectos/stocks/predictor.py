import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

def fetch_data(ticker="NVDA", period="5y"):
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.dropna(inplace=True)
    return df

def engineer_features(df):
    print("Engineering features...")
    # Basic Price features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # RSI
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Moving Averages
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    # Price Relative to MAs
    df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
    
    # Previous day's returns (Lag features)
    for i in range(1, 4):
        df[f'Lag_{i}_Returns'] = df['Returns'].shift(i)
        
    df = df.dropna()
    return df

def create_target(df):
    # Target: 1 if tomorrow's close is higher than today's close, else 0
    # Note: df['Close'].shift(-1) gets tomorrow's close
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df

def train_and_predict(df):
    print("Training model...")
    # Features to use for prediction
    features = ['Returns', 'Log_Returns', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 
                'SMA_20', 'SMA_50', 'EMA_20', 'BB_High', 'BB_Low', 'BB_Mid', 'Price_to_SMA20',
                'Lag_1_Returns', 'Lag_2_Returns', 'Lag_3_Returns']
    
    # Split data chronologically
    # We want to predict tomorrow, so the last row has effectively NaN target if we just shift
    # We already created target. The last row of our dataframe has no "tomorrow's close" yet.
    # So we must separate the last row for prediction!
    
    # The last row has NaNs for 'Target' since we shifted -1, or False if astype(int) made it 0 falsely.
    # Actually, df['Target'] on the last row is invalid because we don't know tomorrow's close.
    # Let's fix that.
    
    # The true last row for which we HAVE features but NO target is the last date available.
    predict_row = df.iloc[[-1]][features]
    last_date = df.index[-1].strftime('%Y-%m-%d')
    
    # Remove the last row from training data
    train_df = df.iloc[:-1]
    
    X = train_df[features]
    y = train_df['Target']
    
    # Split into train and test sets (80% train, 20% test)
    split_idx = int(len(train_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, min_samples_split=10)
    rf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Recent Test Data: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print Feature Importances
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nTop 5 Feature Importances:")
    print(importances.head(5))
    
    # Predict tomorrow's movement using the last row
    prediction = rf.predict(predict_row)[0]
    probabilities = rf.predict_proba(predict_row)[0]
    
    print(f"\n{'='*50}")
    print(f"PREDICTION FOR NEXT TRADING DAY (after {last_date}):")
    if prediction == 1:
        print("Direction: HIGHER")
        print(f"Confidence: {probabilities[1]*100:.2f}%")
    else:
        print("Direction: LOWER")
        print(f"Confidence: {probabilities[0]*100:.2f}%")
    print(f"{'='*50}\n")
    
    return prediction, probabilities

def main():
    # 1. Fetch
    df = fetch_data("NVDA", period="5y")
    
    # 2. Engineer features
    df = engineer_features(df)
    
    # 3. Create target
    df = create_target(df)
    
    # 4. Train and predict
    train_and_predict(df)

if __name__ == "__main__":
    main()
