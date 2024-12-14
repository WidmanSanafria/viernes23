import asyncio
import logging
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import ta  # Indicadores técnicos
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import csv
from tqdm import tqdm
from config import TESTNET_API_KEY, TESTNET_API_SECRET, MAINNET_API_KEY, MAINNET_API_SECRET
from telegram import Bot

# ====================
# CONFIGURACIÓN GLOBAL
# ====================
IS_TESTNET = True  # Modo Testnet o Producción
INTERVAL = Client.KLINE_INTERVAL_1MINUTE  # Intervalo de las velas
LOOKBACK = "2000 minutes ago UTC"  # Cantidad de datos históricos
ATR_MULTIPLIER = 1.5  # Multiplicador para Stop-Loss y Take-Profit
TRADE_AMOUNT_USDT = 100  # Monto fijo por operación
SLEEP_INTERVAL = 5  # Tiempo de espera entre ejecuciones
WINDOW_SIZE = 90  # Tamaño de ventana para el modelo LSTM
EPOCHS = 20  # Número de épocas de entrenamiento
BATCH_SIZE = 16  # Tamaño del batch de entrenamiento
MODEL_SAVE_PATH = "lstm_model.keras"  # Ruta para guardar/cargar el modelo
PNL_CSV_PATH = "trading_pnl.csv"  # Archivo para registrar resultados
MAX_CONSECUTIVE_LOSSES = 2  # Máximo de pérdidas consecutivas antes de cambiar de par
pairs = ["DOGEUSDT", "PEPEUSDT", "WBETHUSDT", "WBTCUSDT", "SOLUSDT" ]  # Lista de pares a operar

# Umbral de predicción para abortar la operación
PREDICTION_THRESHOLD = 0.01  # Si la predicción es menor que este valor, abortar la operación

# Variables de control
profit_loss_accumulated = 0
consecutive_losses = 0
current_pair_index = 0

# Configuración de Telegram
TELEGRAM_BOT_TOKEN = "8015944139:AAHF2F-aBugfnXaVMBLbLPQURWaHUpn-koc"  # Reemplaza con tu token
TELEGRAM_USER_ID = "5517617300"  # Reemplaza con tu ID de usuario

# ====================
# CONFIGURACIÓN DE LOGGING
# ====================
logger = logging.getLogger("TradingBot")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("bot_trading.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ====================
# INICIALIZACIÓN DEL CLIENTE BINANCE
# ====================
def initialize_client(is_testnet):
    """Inicializa el cliente de Binance."""
    try:
        if is_testnet:
            logger.info("Inicializando cliente en Testnet...")
            return Client(TESTNET_API_KEY, TESTNET_API_SECRET, testnet=True)
        else:
            logger.info("Inicializando cliente en Producción...")
            return Client(MAINNET_API_KEY, MAINNET_API_SECRET)
    except BinanceAPIException as e:
        logger.error(f"Error al conectar con Binance: {e}")
        return None

client = initialize_client(IS_TESTNET)

# ====================
# FUNCIONES AUXILIARES
# ====================
def add_indicators(df):
    """Agrega indicadores técnicos al DataFrame."""
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=26)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['bollinger_high'], df['bollinger_low'] = ta.volatility.bollinger_hband(df['close']), ta.volatility.bollinger_lband(df['close'])
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    return df.dropna()

async def get_historical_data(symbol):
    """Obtiene datos históricos del par especificado."""
    try:
        klines = await asyncio.to_thread(client.get_historical_klines, symbol, INTERVAL, LOOKBACK)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                           'taker_buy_quote_asset_volume', 'ignore'])
        df[['high', 'low', 'close', 'volume']] = df[['high', 'low', 'close', 'volume']].astype(float)
        return add_indicators(df)
    except Exception as e:
        logger.error(f"Error al obtener datos de {symbol}: {e}")
        return None

def prepare_lstm_data(df, window_size):
    """Prepara los datos para el modelo LSTM."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = df[['ema_fast', 'ema_slow', 'rsi', 'atr', 'macd', 'bollinger_high', 'bollinger_low', 'adx', 'close']]
    scaled_features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(window_size, len(scaled_features)):
        X.append(scaled_features[i-window_size:i])
        y.append(scaled_features[i, -1])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    """Construye el modelo LSTM con mayor precisión."""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),  # Primera capa LSTM con 100 neuronas
        Dropout(0.2),  # Dropout para evitar overfitting
        LSTM(100, return_sequences=False),  # Segunda capa LSTM con 100 neuronas
        Dropout(0.2),  # Dropout para evitar overfitting
        Dense(25, activation='relu'),  # Capa densa con 25 neuronas
        Dense(1)  # Capa de salida
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

async def send_telegram_message(message):
    """Envía un mensaje a través del bot de Telegram."""
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_USER_ID, text=message)

async def check_balance(symbol):
    """Verifica el saldo disponible para el par especificado."""
    try:
        account_info = await asyncio.to_thread(client.get_account)
        balances = account_info['balances']
        for balance in balances:
            if balance['asset'] == symbol.replace("USDT", ""):
                free_balance = float(balance['free'])
                logger.info(f"Saldo disponible para {symbol}: {free_balance:.10f}")
                return free_balance > 0
        return False
    except Exception as e:
        logger.error(f"Error al verificar el saldo: {e}")
        return False

async def execute_order(symbol, side, entry_price, stop_loss, take_profit):
    """Simula la ejecución de una orden y monitoriza su resultado."""
    global consecutive_losses, profit_loss_accumulated
    start_time = datetime.now()  # Tiempo de inicio de la operación
    logger.info(f"{side} {TRADE_AMOUNT_USDT} USDT de {symbol} a {entry_price:.10f}")
    await send_telegram_message(f"🔵 {side} {TRADE_AMOUNT_USDT} USDT de {symbol} a {entry_price:.10f}\n"
                                f"🟠 Stop-Loss: {stop_loss:.10f}, 🟠 Take-Profit: {take_profit:.10f}")

    pnl = 0  # Inicialización para evitar el error
    trailing_stop = take_profit  # Inicializar el trailing stop
    for _ in tqdm(range(84), desc="Monitoreo", ncols=80):  # 84 iteraciones de 5 segundos = 7 minutos
        current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        logger.info(f"Precio actual: {current_price:.10f}, Trailing Stop: {trailing_stop:.10f}")

        if side == "BUY":
            if current_price >= trailing_stop:
                pnl = (trailing_stop - entry_price) * (TRADE_AMOUNT_USDT / entry_price)
                logger.info(f"Vendiendo {symbol} al precio {current_price:.10f} para obtener ganancias.")
                await send_telegram_message(f"🟢 Vendiendo {symbol} al precio {current_price:.10f} para obtener ganancias.")
                break
            elif current_price <= stop_loss:
                pnl = (stop_loss - entry_price) * (TRADE_AMOUNT_USDT / entry_price)
                logger.info(f"Vendiendo {symbol} al precio {current_price:.10f} debido a Stop-Loss.")
                await send_telegram_message(f"🔴 Vendiendo {symbol} al precio {current_price:.10f} debido a Stop-Loss.")
                break
            # Ajustar el trailing stop si el precio sube
            if current_price > trailing_stop:
                trailing_stop = current_price - ATR_MULTIPLIER * df['atr'].iloc[-1]
        elif side == "SELL":
            if current_price <= trailing_stop:
                pnl = (entry_price - trailing_stop) * (TRADE_AMOUNT_USDT / entry_price)
                logger.info(f"Comprando {symbol} al precio {current_price:.10f} para obtener ganancias.")
                await send_telegram_message(f"🟢 Comprando {symbol} al precio {current_price:.10f} para obtener ganancias.")
                break
            elif current_price >= stop_loss:
                pnl = (entry_price - stop_loss) * (TRADE_AMOUNT_USDT / entry_price)
                logger.info(f"Comprando {symbol} al precio {current_price:.10f} debido a Stop-Loss.")
                await send_telegram_message(f"🔴 Comprando {symbol} al precio {current_price:.10f} debido a Stop-Loss.")
                break
            # Ajustar el trailing stop si el precio baja
            if current_price < trailing_stop:
                trailing_stop = current_price + ATR_MULTIPLIER * df['atr'].iloc[-1]
        await asyncio.sleep(5)

    # Si no se alcanza ni el Stop-Loss ni el Take-Profit, cerrar al precio más cercano al Take-Profit
    if pnl == 0:
        current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        if side == "BUY":
            pnl = (current_price - entry_price) * (TRADE_AMOUNT_USDT / entry_price)
            logger.info(f"Cerrando operación al precio actual {current_price:.10f} (más cercano al Take-Profit).")
            await send_telegram_message(f"🟢 Cerrando operación al precio actual {current_price:.10f} (más cercano al Take-Profit).")
        elif side == "SELL":
            pnl = (entry_price - current_price) * (TRADE_AMOUNT_USDT / entry_price)
            logger.info(f"Cerrando operación al precio actual {current_price:.10f} (más cercano al Take-Profit).")
            await send_telegram_message(f"🟢 Cerrando operación al precio actual {current_price:.10f} (más cercano al Take-Profit).")

    end_time = datetime.now()  # Tiempo de finalización de la operación
    elapsed_time = end_time - start_time  # Tiempo transcurrido
    profit_loss_accumulated += pnl
    consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
    logger.info(f"Resultado: PnL: {pnl:.10f} USD, PnL Acumulado: {profit_loss_accumulated:.10f} USD")
    logger.info(f"Tiempo transcurrido: {elapsed_time}")
    await send_telegram_message(f"Resultado: PnL: {pnl:.10f} USD, PnL Acumulado: {profit_loss_accumulated:.10f} USD\n"
                                f"Tiempo transcurrido: {elapsed_time}")

# ====================
# FUNCIÓN PRINCIPAL
# ====================
async def main():
    global current_pair_index, consecutive_losses
    try:
        while True:
            symbol = pairs[current_pair_index]
            logger.info(f"Obteniendo datos históricos para {symbol}...")
            df = await get_historical_data(symbol)
            if df is None or len(df) < WINDOW_SIZE:
                logger.warning(f"No se pudieron obtener datos para {symbol}. Saltando...")
                await asyncio.sleep(SLEEP_INTERVAL)
                continue

            logger.info(f"Preparando datos para {symbol}...")
            X, y, _ = prepare_lstm_data(df, WINDOW_SIZE)
            logger.info(f"Dimensiones de X: {X.shape}")
            model = build_lstm_model((X.shape[1], X.shape[2])) if not os.path.exists(MODEL_SAVE_PATH) else load_model(MODEL_SAVE_PATH)
            early_stopping = EarlyStopping(monitor='loss', patience=3)
            await asyncio.to_thread(model.fit, X, y, EPOCHS, BATCH_SIZE, verbose=0, callbacks=[early_stopping])
            model.save(MODEL_SAVE_PATH)

            prediction = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))[0][0]
            current_price = df['close'].iloc[-1]
            logger.info(f"Predicción del modelo: {prediction:.10f}, Precio actual: {current_price:.10f}")

            # Verificar si la predicción es demasiado baja para abortar la operación
            if prediction < PREDICTION_THRESHOLD:
                logger.info(f"Predicción demasiado baja ({prediction:.10f}). Abortando operación.")
                await send_telegram_message(f"🔴 Predicción demasiado baja ({prediction:.10f}). Abortando operación.")
                await asyncio.sleep(SLEEP_INTERVAL)
                continue

            atr = df['atr'].iloc[-1]
            stop_loss = current_price - ATR_MULTIPLIER * atr if prediction > current_price else current_price + ATR_MULTIPLIER * atr
            take_profit = current_price + ATR_MULTIPLIER * atr if prediction > current_price else current_price - ATR_MULTIPLIER * atr

            side = "BUY" if prediction > current_price else "SELL"

            # Verificar saldo antes de operar
            if side == "SELL" and not await check_balance(symbol):
                logger.info(f"No hay saldo disponible para {symbol}. Esperando orden de compra...")
                await send_telegram_message(f"🔴 No hay saldo disponible para {symbol}. Esperando orden de compra...")
                await asyncio.sleep(SLEEP_INTERVAL)
                continue

            logger.info(f"Ejecutando orden {side} en {symbol}...")
            await execute_order(symbol, side, current_price, stop_loss, take_profit)

            if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                current_pair_index = (current_pair_index + 1) % len(pairs)
                logger.info(f"Cambiando de par a {pairs[current_pair_index]} después de {MAX_CONSECUTIVE_LOSSES} pérdidas consecutivas.")
            await asyncio.sleep(SLEEP_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Tarea asíncrona cancelada. Cerrando...")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")

if __name__ == "__main__":
    asyncio.run(main())