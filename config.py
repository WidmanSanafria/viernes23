# config.py

# ========================
# Configuración General
# ========================
ENVIRONMENT = "mainnet"  # Cambiar a "testnet" para pruebas, "mainnet" para producción

# ========================
# Claves API de Binance
# ========================
# Testnet
TESTNET_API_KEY = "SYAVHcqQzX7A7oavWN4mppN9w2Tzsz7B4LuJOE1IospNWa1LAjaJ2rJ0VYc1qVaS"
TESTNET_API_SECRET = "ifBGdBOzEE65wAkvjjOiprLP2yiMFU7MHyrgFh0WkvjT6WqjlbRfRdfvvZQQrOiK"

# Mainnet
MAINNET_API_KEY = "SYAVHcqQzX7A7oavWN4mppN9w2Tzsz7B4LuJOE1IospNWa1LAjaJ2rJ0VYc1qVaS"
MAINNET_API_SECRET = "ifBGdBOzEE65wAkvjjOiprLP2yiMFU7MHyrgFh0WkvjT6WqjlbRfRdfvvZQQrOiK"

# ========================
# Configuración de Telegram
# ========================
TELEGRAM_BOT_TOKEN = "8015944139:AAHF2F-aBugfnXaVMBLbLPQURWaHUpn-koc"  # Reemplaza con tu token de Telegram
TELEGRAM_USER_ID = "5517617300"  # Reemplaza con tu ID de usuario de Telegram

# ========================
# Selección Automática de Claves
# ========================
if ENVIRONMENT == "testnet":
    API_KEY = TESTNET_API_KEY
    API_SECRET = TESTNET_API_SECRET
    BINANCE_URL = "https://testnet.binance.vision"  # Endpoint de Testnet
else:
    API_KEY = MAINNET_API_KEY
    API_SECRET = MAINNET_API_SECRET
    BINANCE_URL = None  # URL Mainnet por defecto en ccxt