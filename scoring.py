import pandas as pd

def compute_score(df):
    # Score pondéré basé sur les indicateurs
    score = []
    for i, row in df.iterrows():
        # Correction : cast volume et VOL20 en float pour éviter les erreurs de comparaison
        try:
            volume = float(row['volume'])
        except Exception:
            volume = 0.0
        try:
            vol20 = float(row['VOL20'])
        except Exception:
            vol20 = 0.0
            
        s = 0
        # Pondération SMA20/SMA50 (croisement haussier/bassier)
        if row['SMA20'] > row['SMA50']:
            s += 1
        elif row['SMA20'] < row['SMA50']:
            s -= 1
        # RSI
        if row['RSI14'] < 30:
            s += 1  # Survente
        elif row['RSI14'] > 70:
            s -= 1  # Surachat
        # MACD
        if row['MACD'] > row['MACD_signal']:
            s += 1
        elif row['MACD'] < row['MACD_signal']:
            s -= 1
        # Volume (hausse du volume = +0.5)
        if volume > vol20:
            s += 0.5
        score.append(s)
    df['score'] = score
    return df

def detect_divergence(df):
    # Divergence avancée : MACD et prix ne vont pas dans le même sens, type haussier/baissier
    divergence = [0]
    divergence_type = [None]
    for i in range(1, len(df)):
        price_delta = float(df['close'].iloc[i]) - float(df['close'].iloc[i-1])
        macd_delta = df['MACD'].iloc[i] - df['MACD'].iloc[i-1]
        if price_delta * macd_delta < 0:
            divergence.append(1)
            if price_delta < 0 and macd_delta > 0:
                divergence_type.append('bullish')  # Divergence haussière
            elif price_delta > 0 and macd_delta < 0:
                divergence_type.append('bearish')  # Divergence baissière
            else:
                divergence_type.append(None)
        else:
            divergence.append(0)
            divergence_type.append(None)
    df['divergence'] = divergence
    df['divergence_type'] = divergence_type
    return df

def filter_signals(df, score_threshold=2):
    # Filtrage des faux signaux : on ne garde que les signaux forts
    df['signal'] = ((df['score'] >= score_threshold) & (df['divergence'] == 1)).astype(int)
    return df
