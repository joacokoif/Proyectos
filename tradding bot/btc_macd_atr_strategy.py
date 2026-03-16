import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# ESTRATEGIA 4: BTC MACD ATR Strategy
# ==========================================
# Lógica de la estrategia en TradingView:
# 1. MACD (12, 26, 9)
# 2. EMA 50 (Filtro de tendencia)
# 3. ATR 14 (Para gestión de riesgo de SL y TP)
# 
# Condiciones Long: Cruce alcista de MACD + Precio sobre EMA 50
# Condiciones Short: Cruce bajista de MACD + Precio bajo EMA 50
# SL y TP basados en el ATR (Ratio Riesgo/Beneficio de 1:1.5)
# ==========================================

# Configuración de Binance API
BASE_URL = 'https://api.binance.com'

def get_klines(symbol='BTCUSDT', interval='1h', limit=1000):
    """
    Obtiene los datos históricos (velas o klines) desde la API de Binance.
    Documentación: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
    """
    url = f"{BASE_URL}/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    # Realizamos la petición HTTP a la API de Binance
    response = requests.get(url, params=params)
    response.raise_for_status() # Lanza un error si la petición falla
    data = response.json()
    
    # Convertimos los datos a un DataFrame de Pandas para su fácil manipulación
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Seleccionamos las columnas que vamos a utilizar
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Convertimos el timestamp de UNIX (milisegundos) a formato de fecha legible
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convertimos las columnas de precios a valores de tipo float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    return df

def calculate_indicators(df):
    """
    Calcula los indicadores MACD, EMA 50 y ATR 14.
    """
    # 1. EMA 50 (Exponential Moving Average)
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # 2. MACD (12, 26, 9)
    # Primero calculamos las dos EMAs rápidas y lentas
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    
    # Línea MACD = EMA(12) - EMA(26)
    df['macd_line'] = ema_12 - ema_26
    
    # Línea de Señal = EMA(9) aplicada sobre la línea MACD
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    
    # 3. ATR (Average True Range) - Periodo 14
    # Calculamos primero el True Range (Rango Verdadero). Es el máximo de:
    # a. High - Low
    # b. Abs(High - Close Previo)
    # c. Abs(Low - Close Previo)
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Para emular el `ta.atr(14)` exacto de TradingView, usamos RMA (Running Moving Average)
    # En pandas esto equivale a un EWM con alpha = 1 / longitud
    alpha = 1 / 14
    df['atr_14'] = df['true_range'].ewm(alpha=alpha, adjust=False).mean()
    
    # Eliminamos las columnas temporales del cálculo ATR para mantener la tabla limpia
    df.drop(columns=['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], inplace=True)
    
    return df

def generate_signals(df):
    """
    Genera las señales Long y Short con la lógica condicional indicada.
    """
    # Desplazamos los indicadores para poder detectar el 'cruce' comparando la vela anterior con la actual
    df['prev_macd'] = df['macd_line'].shift(1)
    df['prev_signal'] = df['signal_line'].shift(1)
    
    # CONDICIÓN LONG: cruce del MACD por encima del Signal (crossover) Y el cierre es mayor a EMA 50
    macd_crossover = (df['macd_line'] > df['signal_line']) & (df['prev_macd'] <= df['prev_signal'])
    df['long_condition'] = macd_crossover & (df['close'] > df['ema_50'])
    
    # CONDICIÓN SHORT: cruce del MACD por debajo del Signal (crossunder) Y el cierre es menor a EMA 50
    macd_crossunder = (df['macd_line'] < df['signal_line']) & (df['prev_macd'] >= df['prev_signal'])
    df['short_condition'] = macd_crossunder & (df['close'] < df['ema_50'])
    
    return df

def run_strategy():
    """
    Función principal que articula todos los procesos de la estrategia.
    """
    symbol = 'BTCUSDT'
    interval = '1h'  # Temporalidad, ej. '1h' = 1 hora, '1d' = 1 día
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando estrategia para {symbol} ({interval})...")
    
    # 1. Obtener datos
    df = get_klines(symbol=symbol, interval=interval, limit=1000)
    
    # 2. Calcular indicadores
    df = calculate_indicators(df)
    
    # 3. Obtener señales
    df = generate_signals(df)
    
    # 4. Analizar la barra más reciente
    last_row = df.iloc[-1]
    
    close_price = last_row['close']
    atr_val = last_row['atr_14']
    
    print("\n" + "="*50)
    print(f"[ DATOS DEL MERCADO ({last_row['timestamp']}) ]")
    print("="*50)
    print(f"Precio actual (Close): {close_price:.2f}")
    print(f"EMA 50:                {last_row['ema_50']:.2f}")
    print(f"MACD Line / Signal:    {last_row['macd_line']:.2f} / {last_row['signal_line']:.2f}")
    print(f"ATR (14):              {atr_val:.2f}")
    print("\n--- SEÑALES ESTRATEGIA  ---")
    
    # 5. Lógica de entrada/salida emulando strategy.entry y strategy.exit
    if last_row['long_condition']:
        print("[LONG] ALERTA LONG (COMPRA)! - Condiciones cumplidas")
        
        # Stop = close - atr, Limit = close + atr * 1.5
        stop_loss = close_price - atr_val
        take_profit = close_price + (atr_val * 1.5)
        
        print(f"    -> Precio Entrada: {close_price:.2f}")
        print(f"    -> Stop Loss    (-1.0 ATR): {stop_loss:.2f}")
        print(f"    -> Take Profit  (+1.5 ATR): {take_profit:.2f}")
        
    elif last_row['short_condition']:
        print("[SHORT] ALERTA SHORT (VENTA)! - Condiciones cumplidas")
        
        # Stop = close + atr, Limit = close - atr * 1.5
        stop_loss = close_price + atr_val
        take_profit = close_price - (atr_val * 1.5)
        
        print(f"    -> Precio Entrada: {close_price:.2f}")
        print(f"    -> Stop Loss    (+1.0 ATR): {stop_loss:.2f}")
        print(f"    -> Take Profit  (-1.5 ATR): {take_profit:.2f}")
        
    else:
        print("[-] No hay condiciones de entrada (LONG o SHORT) en la vela actual.")
        print("   Esperando cruce del MACD o posicionamiento respecto al EMA 50.")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    run_strategy()
