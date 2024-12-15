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
from datetime import datetime, timedelta
import os
import csv
from tqdm import tqdm
from config import API_KEY, API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID, ENVIRONMENT
from telegram import Bot

# ====================
# CONFIGURACIÓN GLOBAL
# ====================
IS_TESTNET = ENVIRONMENT == "testnet"  # Determina si estamos en Testnet o Mainnet
INTERVAL = Client.KLINE_INTERVAL_1MINUTE  # Intervalo de las velas
LOOKBACK = "2000 minutes ago UTC"  # Cantidad de datos históricos
ATR_MULTIPLIER = 1.0  # Multiplicador para Stop-Loss y Take-Profit
TRADE_AMOUNT_USDT = 5  # Monto fijo por operación
SLEEP_INTERVAL = 1  # Tiempo de espera entre ejecuciones
WINDOW_SIZE = 80  # Tamaño de ventana para el modelo LSTM
EPOCHS = 20  # Número de épocas de entrenamiento
BATCH_SIZE = 16  # Tamaño del batch de entrenamiento
MODEL_SAVE_PATH = "lstm_model.keras"  # Ruta para guardar/cargar el modelo
PNL_CSV_PATH = "trading_pnl.csv"  # Archivo para registrar resultados
MAX_CONSECUTIVE_LOSSES = 2  # Máximo de pérdidas consecutivas antes de cambiar de par
MAX_CONSECUTIVE_WINS = 4  # Máximo de ganancias consecutivas para seguir en el mismo par
pairs = ["DOGEUSDT", "PEPEUSDT", "OMUSDT", "WBTCUSDT", "SOLUSDT", "FUNUSDT", "HARDUSDT", "QKCUSDT", "ELFUSDT", "DCRUSDT", "HBARUSDT", "GNSUSDT", "BIFIUSDT", "ARDRUSDT", "RONINUSDT", "PROMUSDT", "CLVUSDT", "POLUSDT"]  # Lista de pares a operar

# Umbral de predicción para abortar la operación
PREDICTION_THRESHOLD = 0.1  # Si la predicción es menor que este valor, abortar la operación

# Tiempo máximo de espera si no hay señales de compra
MAX_WAIT_TIME_MINUTES = 5  # Esperar 5 minutos si no hay señales de compra

# Variables de control
profit_loss_accumulated = 0
consecutive_losses = 0
consecutive_wins = 0
current_pair_index = 0
current_pair = None
last_signal_time = datetime.now()

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
            return Client(API_KEY, API_SECRET, testnet=True)
        else:
            logger.info("Inicializando cliente en Producción...")
            return Client(API_KEY, API_SECRET)
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
        klines = client.get_historical_klines(symbol, INTERVAL, LOOKBACK)
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
        LSTM(60, return_sequences=True, input_shape=input_shape),  # Primera capa LSTM con 60 neuronas
        Dropout(0.2),  # Dropout para evitar overfitting
        LSTM(60, return_sequences=False),  # Segunda capa LSTM con 60 neuronas
        Dropout(0.2),  # Dropout para evitar overfitting
        Dense(25, activation='relu'),  # Capa densa con 25 neuronas
        Dense(1)  # Capa de salida
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

async def send_telegram_message(message):
    """Envía un mensaje a través del bot de Telegram."""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_USER_ID, text=message)
    except Exception as e:
        logger.error(f"Error al enviar mensaje a Telegram: {e}")

async def check_balance(symbol):
    """Verifica el saldo disponible para el par especificado."""
    try:
        account_info = client.get_account()
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
    """Ejecuta una orden y monitoriza su resultado."""
    global consecutive_losses, consecutive_wins, profit_loss_accumulated
    try:
        # Obtener información del símbolo para redondear la cantidad y el precio
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next(filter(lambda f: f['filterType'] == 'LOT_SIZE', symbol_info['filters']))
        price_filter = next(filter(lambda f: f['filterType'] == 'PRICE_FILTER', symbol_info['filters']))

        min_qty = float(lot_size_filter['minQty'])
        max_qty = float(lot_size_filter['maxQty'])
        step_size = float(lot_size_filter['stepSize'])

        min_price = float(price_filter['minPrice'])
        max_price = float(price_filter['maxPrice'])
        tick_size = float(price_filter['tickSize'])

        # Calcular la cantidad redondeada según los requisitos de LOT_SIZE
        quantity = TRADE_AMOUNT_USDT / entry_price
        quantity = round(quantity - quantity % step_size, 8)  # Redondear según step_size
        quantity = max(min(quantity, max_qty), min_qty)  # Asegurar que esté dentro de los límites

        # Redondear el precio según los requisitos de PRICE_FILTER
        entry_price = round(entry_price - entry_price % tick_size, 8)
        entry_price = max(min(entry_price, max_price), min_price)  # Asegurar que esté dentro de los límites

        # Verificar saldo antes de ejecutar la orden
        if side == "BUY":
            asset = "USDT"
        else:
            asset = symbol.replace("USDT", "")

        account_info = client.get_account()
        balances = account_info['balances']
        for balance in balances:
            if balance['asset'] == asset:
                free_balance = float(balance['free'])
                if free_balance < quantity:
                    logger.error(f"Saldo insuficiente para {side} en {symbol}. Saldo disponible: {free_balance:.10f}")
                    await send_telegram_message(f"🔴 Saldo insuficiente para {side} en {symbol}. Saldo disponible: {free_balance:.10f}")
                    return

        # Crear la orden en Binance
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='LIMIT',
            timeInForce='GTC',
            quantity=quantity,
            price=entry_price
        )
        logger.info(f"Orden {side} creada: {order}")
        await send_telegram_message(f"🔵 {side} {TRADE_AMOUNT_USDT} USDT de {symbol} a {entry_price:.10f}\n"
                                    f"🟠 Stop-Loss: {stop_loss:.10f}, 🟠 Take-Profit: {take_profit:.10f}")

        # Monitorizar la orden
        pnl = 0
        start_time = datetime.now()
        while True:
            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
            if side == "BUY":
                if current_price >= take_profit:
                    pnl = (take_profit - entry_price) * quantity
                    logger.info(f"Vendiendo {symbol} al precio {current_price:.10f} para obtener ganancias.")
                    await send_telegram_message(f"🟢 Vendiendo {symbol} al precio {current_price:.10f} para obtener ganancias.")
                    break
                elif current_price <= stop_loss:
                    pnl = (stop_loss - entry_price) * quantity
                    logger.info(f"Vendiendo {symbol} al precio {current_price:.10f} debido a Stop-Loss.")
                    await send_telegram_message(f"🔴 Vendiendo {symbol} al precio {current_price:.10f} debido a Stop-Loss.")
                    break
            elif side == "SELL":
                if current_price <= take_profit:
                    pnl = (entry_price - take_profit) * quantity
                    logger.info(f"Comprando {symbol} al precio {current_price:.10f} para obtener ganancias.")
                    await send_telegram_message(f"🟢 Comprando {symbol} al precio {current_price:.10f} para obtener ganancias.")
                    break
                elif current_price >= stop_loss:
                    pnl = (entry_price - stop_loss) * quantity
                    logger.info(f"Comprando {symbol} al precio {current_price:.10f} debido a Stop-Loss.")
                    await send_telegram_message(f"🔴 Comprando {symbol} al precio {current_price:.10f} debido a Stop-Loss.")
                    break
            await asyncio.sleep(5)

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        profit_loss_accumulated += pnl
        if pnl > 0:
            consecutive_wins += 1
            consecutive_losses = 0
        else:
            consecutive_wins = 0
            consecutive_losses += 1
        logger.info(f"Resultado: PnL: {pnl:.10f} USD, PnL Acumulado: {profit_loss_accumulated:.10f} USD")
        logger.info(f"Tiempo transcurrido: {elapsed_time}")
        await send_telegram_message(f"Resultado: PnL: {pnl:.10f} USD, PnL Acumulado: {profit_loss_accumulated:.10f} USD\n"
                                    f"Tiempo transcurrido: {elapsed_time}")
    except Exception as e:
        logger.error(f"Error al ejecutar la orden: {e}")

# ====================
# FUNCIÓN PRINCIPAL
# ====================
async def main():
    global current_pair_index, consecutive_losses, consecutive_wins, current_pair, last_signal_time
    try:
        while True:
            symbol = pairs[current_pair_index]
            current_pair = symbol
            logger.info(f"Obteniendo datos históricos para {symbol}...")
            df = await get_historical_data(symbol)
            if df is None or len(df) < WINDOW_SIZE:
                logger.warning(f"No se pudieron obtener datos para {symbol}. Saltando...")
                current_pair_index += 1
                await asyncio.sleep(SLEEP_INTERVAL)
                continue

            logger.info(f"Preparando datos para {symbol}...")
            X, y, _ = prepare_lstm_data(df, WINDOW_SIZE)
            logger.info(f"Dimensiones de X: {X.shape}")
            model = build_lstm_model((X.shape[1], X.shape[2])) if not os.path.exists(MODEL_SAVE_PATH) else load_model(MODEL_SAVE_PATH)
            early_stopping = EarlyStopping(monitor='loss', patience=3)
            model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stopping])
            model.save(MODEL_SAVE_PATH)

            prediction = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))[0][0]
            current_price = df['close'].iloc[-1]
            logger.info(f"Predicción del modelo: {prediction:.10f}, Precio actual: {current_price:.10f}")

            if prediction < PREDICTION_THRESHOLD:
                logger.info(f"Predicción demasiado baja ({prediction:.10f}). Esperando 5 minutos antes de cambiar de par.")
                await send_telegram_message(f"🔴 Predicción demasiado baja ({prediction:.10f}). Esperando 5 minutos antes de cambiar de par.")
                for _ in tqdm(range(MAX_WAIT_TIME_MINUTES * 60 // SLEEP_INTERVAL), desc="Esperando señales", ncols=80):
                    await asyncio.sleep(SLEEP_INTERVAL)
                current_pair_index += 1
                continue

            atr = df['atr'].iloc[-1]
            stop_loss = current_price - ATR_MULTIPLIER * atr if prediction > current_price else current_price + ATR_MULTIPLIER * atr
            take_profit = current_price + ATR_MULTIPLIER * atr if prediction > current_price else current_price - ATR_MULTIPLIER * atr

            side = "BUY" if prediction > current_price else "SELL"

            if side == "SELL" and not await check_balance(symbol):
                logger.info(f"No hay saldo disponible para {symbol}. Esperando orden de compra...")
                await send_telegram_message(f"🔴 No hay saldo disponible para {symbol}. Esperando orden de compra...")
                for _ in tqdm(range(MAX_WAIT_TIME_MINUTES * 60 // SLEEP_INTERVAL), desc="Esperando saldo", ncols=80):
                    await asyncio.sleep(SLEEP_INTERVAL)
                continue

            logger.info(f"Ejecutando orden {side} en {symbol}...")
            await execute_order(symbol, side, current_price, stop_loss, take_profit)

            if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                current_pair_index = (current_pair_index + 1) % len(pairs)
                logger.info(f"Cambiando de par a {pairs[current_pair_index]} después de {MAX_CONSECUTIVE_LOSSES} pérdidas consecutivas.")
                await send_telegram_message(f"🔴 Cambiando de par a {pairs[current_pair_index]} después de {MAX_CONSECUTIVE_LOSSES} pérdidas consecutivas.")
                consecutive_losses = 0
            elif consecutive_wins < MAX_CONSECUTIVE_WINS:
                current_pair_index += 1
            await asyncio.sleep(SLEEP_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Tarea asíncrona cancelada. Cerrando...")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")

if __name__ == "__main__":
    asyncio.run(main())