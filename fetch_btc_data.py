import requests
import pandas as pd
import time
import websocket
import json
import sqlite3

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="1h", start_time=None, end_time=None):
    url = f"https://api.binance.com/api/v3/klines"
    limit = 1000
    all_data = []
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if start_time:
        params["startTime"] = int(start_time)
    if end_time:
        params["endTime"] = int(end_time)
    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Erreur lors de la récupération des données : {response.status_code} - {response.text}")
            break
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        params["startTime"] = data[-1][0] + 1
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    return df

def on_message(ws, message):
    data = json.loads(message)
    kline = data['k']['c'] if 'k' in data and 'c' in data['k'] else None
    print(f"Dernier prix BTC/USDT en temps réel : {kline}")

def on_error(ws, error):
    print(f"Erreur WebSocket : {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket fermé")

def on_open(ws):
    print("Connexion WebSocket ouverte")

def save_to_sqlite(df, db_name="btc_data.db", table_name="ohlcv"):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Données sauvegardées dans la base SQLite : {db_name}, table : {table_name}")

if __name__ == "__main__":
    import argparse
    import os
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, help='Date de début (YYYY-MM-DD)', default=None)
    parser.add_argument('--end_date', type=str, help='Date de fin (YYYY-MM-DD)', default=None)
    parser.add_argument('--interval', type=str, help='Intervalle Binance (1m, 1h, 1d, etc.)', default='1h')
    args = parser.parse_args()

    # Gestion des dates
    def to_ms(dt):
        return int(pd.Timestamp(dt).timestamp() * 1000)

    start_ms = to_ms(args.start_date) if args.start_date else None
    end_ms = to_ms(args.end_date) if args.end_date else int(time.time() * 1000)
    interval = args.interval

    # Si pas de start_date, on update comme avant
    if not args.start_date:
        latest_time = None
        if os.path.exists("btc_ohlcv.csv"):
            old_df = pd.read_csv("btc_ohlcv.csv", parse_dates=["open_time", "close_time"])
            if not old_df.empty:
                latest_time = int(old_df["open_time"].max().timestamp() * 1000)
        else:
            old_df = pd.DataFrame()
        new_df = fetch_binance_ohlcv(interval=interval, start_time=latest_time, end_time=end_ms)
        if not new_df.empty:
            if not old_df.empty:
                df = pd.concat([old_df, new_df]).drop_duplicates(subset=["open_time"]).sort_values("open_time")
            else:
                df = new_df
            df.to_csv("btc_ohlcv.csv", index=False)
            print("Données BTC/USDT mises à jour jusqu'à aujourd'hui dans btc_ohlcv.csv")
            save_to_sqlite(df)
        elif not old_df.empty:
            print("Aucune nouvelle donnée, historique déjà à jour.")
            save_to_sqlite(old_df)
        else:
            print("Aucune donnée téléchargée.")
    else:
        # Téléchargement d'un historique complet sur la période demandée
        df = fetch_binance_ohlcv(interval=interval, start_time=start_ms, end_time=end_ms)
        if not df.empty:
            df.to_csv("btc_ohlcv.csv", index=False)
            print(f"Données BTC/USDT téléchargées ({args.start_date} -> {args.end_date}) dans btc_ohlcv.csv")
            save_to_sqlite(df)
        else:
            print("Aucune donnée téléchargée sur la période demandée.")

    # Optionnel : flux temps réel (inchangé)
    # ws_url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
    # ws = websocket.WebSocketApp(ws_url,
    #                             on_open=on_open,
    #                             on_message=on_message,
    #                             on_error=on_error,
    #                             on_close=on_close)
    # print("Connexion au flux temps réel Binance...")
    # ws.run_forever()
