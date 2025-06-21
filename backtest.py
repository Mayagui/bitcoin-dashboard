import pandas as pd
import numpy as np
import random

def run_backtest(df, strategy='signal', score_threshold=2, initial_capital=10000, fee=0.001, invest_pct=1.0, take_profit=None, stop_loss=None, random_seed=42, **kwargs):
    """
    Backtest plusieurs stratégies :
    - 'signal' : entrée/sortie sur signal fort (full-in)
    - 'buy_and_hold' : achat au début, vente à la fin
    - 'random' : entrées/sorties aléatoires
    - 'sma_cross' : croisement SMA20/SMA50
    - 'rsi_extreme' : entrée sur survente/surachat RSI
    - 'advanced_risk' : gestion avancée du capital avec reinvestissement des pertes et retrait des gains
    Args:
        df: DataFrame avec colonnes ['open_time', 'close', 'score', 'divergence', 'SMA20', 'SMA50', 'RSI14']
        strategy: 'signal', 'buy_and_hold', 'random', 'sma_cross', 'rsi_extreme', 'advanced_risk'
        score_threshold: seuil pour valider un signal (pour 'signal')
        initial_capital: capital de départ
        fee: frais de transaction (ex: 0.001 = 0.1%)
        invest_pct: % du capital investi à chaque trade (0.0 à 1.0)
        take_profit: take profit en % (ex: 0.05 pour 5%)
        stop_loss: stop loss en % (ex: 0.02 pour 2%)
        random_seed: pour la reproductibilité du random
    Returns:
        equity_curve: DataFrame avec l'évolution du capital
        trades: DataFrame des trades
        stats: dict de performance
    """
    df = df.copy()
    df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
    df = df.sort_values('open_time').reset_index(drop=True)
    cash = initial_capital
    position = 0
    entry_price = 0
    entry_time = None
    position_size = 0
    trade_log = []
    equity = []  # Initialisation de la liste equity avant la boucle
    random.seed(random_seed)
    for i, row in df.iterrows():
        price = float(row['close'])
        # --- STRATEGY LOGIC ---
        if strategy == 'buy_and_hold':
            if i == 0:
                entry_price = price * (1 + fee)
                position = 1
                position_size = cash / entry_price
                cash = 0
                entry_time = row['open_time']
            elif i == len(df) - 1 and position == 1:
                exit_price = price * (1 - fee)
                cash = position_size * exit_price
                pnl = (exit_price - entry_price) / entry_price
                trade_log.append({'entry_time': entry_time, 'exit_time': row['open_time'], 'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl})
                position = 0
                position_size = 0
        elif strategy == 'random':
            if position == 0 and random.random() < 0.05 and cash > 0:
                entry_price = price * (1 + fee)
                invest_amount = cash * invest_pct
                position_size = invest_amount / entry_price
                cash -= invest_amount
                position = 1
                entry_time = row['open_time']
            elif position == 1 and (random.random() < 0.05 or i == len(df) - 1):
                exit_price = price * (1 - fee)
                cash += position_size * exit_price
                pnl = (exit_price - entry_price) / entry_price
                trade_log.append({'entry_time': entry_time, 'exit_time': row['open_time'], 'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl})
                position = 0
                position_size = 0
        elif strategy == 'signal':
            signal = (row['score'] >= score_threshold) and (row['divergence'] == 1) if 'score' in row and 'divergence' in row else False
            if position == 0 and signal and cash > 0:
                entry_price = price * (1 + fee)
                invest_amount = cash
                position_size = invest_amount / entry_price
                cash -= invest_amount
                position = 1
                entry_time = row['open_time']
            elif position == 1:
                pnl = (price - entry_price) / entry_price
                hit_tp = take_profit is not None and pnl >= take_profit
                hit_sl = stop_loss is not None and pnl <= -stop_loss
                exit_signal = not signal or hit_tp or hit_sl or i == len(df) - 1
                if exit_signal:
                    exit_price = price * (1 - fee)
                    cash += position_size * exit_price
                    pnl = (exit_price - entry_price) / entry_price
                    trade_log.append({'entry_time': entry_time, 'exit_time': row['open_time'], 'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl})
                    position = 0
                    position_size = 0
        elif strategy == 'sma_cross':
            cross_up = row.get('SMA20', 0) > row.get('SMA50', 0) and df['SMA20'].iloc[i-1] <= df['SMA50'].iloc[i-1] if i > 0 else False
            cross_down = row.get('SMA20', 0) < row.get('SMA50', 0) and df['SMA20'].iloc[i-1] >= df['SMA50'].iloc[i-1] if i > 0 else False
            if position == 0 and cross_up and cash > 0:
                entry_price = price * (1 + fee)
                invest_amount = cash
                position_size = invest_amount / entry_price
                cash -= invest_amount
                position = 1
                entry_time = row['open_time']
            elif position == 1 and (cross_down or i == len(df) - 1):
                exit_price = price * (1 - fee)
                cash += position_size * exit_price
                pnl = (exit_price - entry_price) / entry_price
                trade_log.append({'entry_time': entry_time, 'exit_time': row['open_time'], 'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl})
                position = 0
                position_size = 0
        elif strategy == 'rsi_extreme':
            rsi = row.get('RSI14', 50)
            if position == 0 and rsi < 30 and cash > 0:
                entry_price = price * (1 + fee)
                invest_amount = cash
                position_size = invest_amount / entry_price
                cash -= invest_amount
                position = 1
                entry_time = row['open_time']
            elif position == 1 and (rsi > 70 or i == len(df) - 1):
                exit_price = price * (1 - fee)
                cash += position_size * exit_price
                pnl = (exit_price - entry_price) / entry_price
                trade_log.append({'entry_time': entry_time, 'exit_time': row['open_time'], 'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl})
                position = 0
                position_size = 0
        elif strategy == 'advanced_risk':
            invest_pct = kwargs.get('invest_pct', 1.0)
            reinvest_loss_pct = kwargs.get('reinvest_loss_pct', 1.0)
            withdraw_gain_pct = kwargs.get('withdraw_gain_pct', 0.0)
            max_drawdown = kwargs.get('max_drawdown', 0.3)
            signal = (row['score'] >= score_threshold) and (row['divergence'] == 1) if 'score' in row and 'divergence' in row else False
            if position == 0 and signal and cash > 0:
                entry_price = price * (1 + fee)
                invest_amount = cash * invest_pct
                position_size = invest_amount / entry_price
                cash -= invest_amount
                position = 1
                entry_time = row['open_time']
            elif position == 1:
                pnl = (price - entry_price) / entry_price
                hit_tp = take_profit is not None and pnl >= take_profit
                hit_sl = stop_loss is not None and pnl <= -stop_loss
                exit_signal = not signal or hit_tp or hit_sl or i == len(df) - 1
                if exit_signal:
                    exit_price = price * (1 - fee)
                    position_value = position_size * exit_price
                    gain = position_value - (position_size * entry_price)
                    # Gestion du retrait ou réinvestissement
                    if gain > 0:
                        withdraw = gain * withdraw_gain_pct
                        cash += position_value - withdraw
                    else:
                        reinvest = abs(gain) * reinvest_loss_pct
                        cash += position_value + reinvest
                    pnl = (exit_price - entry_price) / entry_price
                    trade_log.append({'entry_time': entry_time, 'exit_time': row['open_time'], 'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl})
                    position = 0
                    position_size = 0
            # Drawdown sur l'equity totale (cash + position)
            current_equity = cash + (position_size * price if position == 1 else 0)
            if (current_equity / initial_capital - 1) <= -max_drawdown:
                break
        # Equity update
        if position == 0:
            equity.append(cash)
        else:
            equity.append(cash + position_size * price)
    # Correction : aligner equity et open_time
    if len(equity) > len(df['open_time']):
        equity = equity[:len(df['open_time'])]
    elif len(equity) < len(df['open_time']):
        equity += [equity[-1]] * (len(df['open_time']) - len(equity))
    equity_curve = pd.DataFrame({'open_time': df['open_time'], 'equity': equity})
    trades = pd.DataFrame(trade_log)
    # Correction : s'assurer que la colonne 'pnl' existe même si trade_log est vide
    if trades.empty:
        trades = pd.DataFrame({'pnl': []})
    # Stats
    total_return = equity[-1] / initial_capital - 1
    nb_trades = len(trades)
    win_rate = (trades['pnl'] > 0).mean() if nb_trades > 0 else 0
    max_drawdown = (1 - equity_curve['equity'] / equity_curve['equity'].cummax()).max()
    returns = equity_curve['equity'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    # Nouvelle métrique : profit factor
    profit_factor = trades['pnl'].sum() / abs(trades[trades['pnl'] < 0]['pnl'].sum()) if (trades['pnl'] < 0).any() else np.nan
    stats = {
        'total_return': total_return,
        'nb_trades': nb_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'profit_factor': profit_factor
    }
    return equity_curve, trades, stats

def optimize_parameters(df, score_range=[1,2,3], fee=0.001):
    results = []
    for score_threshold in score_range:
        equity_curve, trades, stats = run_backtest(df, strategy='signal', score_threshold=score_threshold, fee=fee)
        results.append({
            'score_threshold': score_threshold,
            'total_return': stats['total_return'],
            'sharpe': stats['sharpe'],
            'max_drawdown': stats['max_drawdown'],
            'nb_trades': stats['nb_trades'],
            'win_rate': stats['win_rate'],
            'profit_factor': stats['profit_factor']
        })
    return pd.DataFrame(results)
