import yfinance as yf
import pandas as pd
import warnings
import datetime
import sys

warnings.filterwarnings('ignore')

# Universo de Acciones (Mega Caps de EE.UU. + SPY como Benchmark del S&P 500)
tickers = [
    'SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 
    'JPM', 'V', 'WMT', 'JNJ', 'PG', 'UNH', 'HD', 'MA', 'CVX', 'LLY'
]

print(f"Descargando datos históricos para {len(tickers)} acciones...")

results = []
for ticker in tickers:
    try:
        # Descargamos datos ocultando warnings
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty: continue
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if 'Close' not in df.columns:
            continue
            
        close = df['Close'].dropna()
        if len(close) < 200: continue
        
        precio_actual = float(close.iloc[-1].item() if hasattr(close.iloc[-1], 'item') else close.iloc[-1])
        sma_50 = float(close.rolling(50).mean().iloc[-1].item() if hasattr(close.rolling(50).mean().iloc[-1], 'item') else close.rolling(50).mean().iloc[-1])
        sma_200 = float(close.rolling(200).mean().iloc[-1].item() if hasattr(close.rolling(200).mean().iloc[-1], 'item') else close.rolling(200).mean().iloc[-1])
        
        # Calculo RSI de 14 dias
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_14 = float((100 - (100 / (1 + rs))).iloc[-1].item() if hasattr((100 - (100 / (1 + rs))).iloc[-1], 'item') else (100 - (100 / (1 + rs))).iloc[-1])
        
        # Reglas Cuantitativas del MVP (Pullback en Tendencia Alcista)
        tendencia_alcista = precio_actual > sma_200
        sobrevendida_corto_plazo = rsi_14 < 45  # Ajustado a 45 para encontrar pullbacks leves
        sobrecomprada = rsi_14 > 70
        
        signal = "HOLD"
        if tendencia_alcista and sobrevendida_corto_plazo:
            signal = "BUY (Pullback en Tendencia)"
        elif not tendencia_alcista and rsi_14 < 30:
            signal = "BUY (Rebote de Riesgo)"
        elif sobrecomprada:
            signal = "SELL (Sobrecomprada)"
            
        results.append({
            'Ticker': ticker,
            'Price ($)': round(precio_actual, 2),
            'RSI_14': round(rsi_14, 2),
            'Above_SMA200': "Si" if tendencia_alcista else "No",
            'Signal': signal
        })
    except Exception as e:
        # print("Error with", ticker, e)
        pass

df_results = pd.DataFrame(results)
print("\n=== ESCÁNER CUANTITATIVO MVP (Horizonte 1-4 Semanas) ===")
print("Fecha de Análisis:", datetime.datetime.now().strftime("%Y-%m-%d"))

if not df_results.empty:
    df_results = df_results.sort_values('RSI_14')
    print(df_results.to_string(index=False))
else:
    print("No se pudieron cargar datos.")
