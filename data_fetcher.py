import requests
import pandas as pd
import websocket
import json

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="1h", limit=1000):
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Erreur lors de la récupération des données : {response.status_code} - {response.text}")
        return pd.DataFrame()
    data = response.json()
    if not isinstance(data, list):
        print(f"Réponse inattendue de l'API : {data}")
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    return df

def start_realtime_ws(on_message, on_open=None, on_error=None, on_close=None):
    ws_url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
    ws = websocket.WebSocketApp(ws_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

def install_and_import(package, import_name=None):
    import importlib
    import subprocess
    import sys
    try:
        if import_name is None:
            import_name = package
        importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        importlib.import_module(import_name)

# Automatisation des dépendances critiques
install_and_import('websocket-client', 'websocket')
install_and_import('requests')
install_and_import('pandas')
install_and_import('json')
