import sqlite3
import importlib
import subprocess
import sys

def install_and_import(package, import_name=None):
    try:
        if import_name is None:
            import_name = package
        importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        importlib.import_module(import_name)

# Automatisation de pandas
install_and_import('pandas')
import pandas as pd

def save_to_sqlite(df, db_name="btc_data.db", table_name="ohlcv"):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Données sauvegardées dans la base SQLite : {db_name}, table : {table_name}")

def read_from_sqlite(db_name="btc_data.db", table_name="ohlcv"):
    try:
        conn = sqlite3.connect(db_name)
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame(), str(e)
