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

# Automatisation de pandas
install_and_import('pandas')

import pandas as pd

def compute_indicators(df):
    # Moyenne mobile simple 20 périodes
    df['SMA20'] = df['close'].astype(float).rolling(window=20).mean()
    # Moyenne mobile simple 50 périodes
    df['SMA50'] = df['close'].astype(float).rolling(window=50).mean()
    # RSI 14 périodes
    delta = df['close'].astype(float).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    # MACD (12, 26, 9)
    ema12 = df['close'].astype(float).ewm(span=12, adjust=False).mean()
    ema26 = df['close'].astype(float).ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Volume moyen 20 périodes
    df['VOL20'] = df['volume'].astype(float).rolling(window=20).mean()
    # Volume moyen 50 périodes
    df['VOL50'] = df['volume'].astype(float).rolling(window=50).mean()
    return df
