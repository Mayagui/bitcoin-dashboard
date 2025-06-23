import streamlit as st
import pandas as pd
from db_utils import read_from_sqlite
import threading
import queue
import time
from data_fetcher import start_realtime_ws
import subprocess
from indicators import compute_indicators
from scoring import compute_score, detect_divergence, filter_signals
import plotly.graph_objects as go
import plotly.express as px
import feedparser
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import yfinance as yf

st.set_page_config(page_title="Dashboard Bitcoin", layout="wide")
st.title("Dashboard Bitcoin - Données historiques")

UPDATE_INTERVAL = 300  # 5 minutes

# --- FONCTIONS DE MISE À JOUR DES DONNÉES POUR STREAMLIT CLOUD ---
def update_binance_data():
    try:
        from fetch_btc_data import fetch_binance_ohlcv, save_to_sqlite
        import os
        import pandas as pd
        import time
        # On tente de charger l'ancien CSV s'il existe
        if os.path.exists("btc_ohlcv.csv"):
            old_df = pd.read_csv("btc_ohlcv.csv", parse_dates=["open_time", "close_time"])
            latest_time = int(old_df["open_time"].max().timestamp() * 1000) if not old_df.empty else None
        else:
            old_df = pd.DataFrame()
            latest_time = None
        new_df = fetch_binance_ohlcv(interval="1h", start_time=latest_time, end_time=int(time.time() * 1000))
        if not new_df.empty:
            if not old_df.empty:
                df = pd.concat([old_df, new_df]).drop_duplicates(subset=["open_time"]).sort_values("open_time")
            else:
                df = new_df
            df.to_csv("btc_ohlcv.csv", index=False)
            save_to_sqlite(df)
        elif not old_df.empty:
            save_to_sqlite(old_df)
    except Exception as e:
        print(f"Erreur update_binance_data: {e}")

def update_yahoo_data():
    try:
        import fetch_yahoo_btc
    except Exception as e:
        print(f"Erreur update_yahoo_data: {e}")

# --- MISE À JOUR DES DONNÉES AU DÉMARRAGE (remplace subprocess) ---
update_yahoo_data()
update_binance_data()

# Affichage des données historiques (toujours depuis le fichier local si l'API échoue)
df_result = read_from_sqlite() if 'read_from_sqlite' in globals() else (pd.DataFrame(), 'Fonction non trouvée')
if isinstance(df_result, tuple):
    df, error = df_result
else:
    df, error = df_result, None

# Nettoyage : suppression des lignes où 'close' n'est pas un nombre (CSV Yahoo)
if isinstance(df, pd.DataFrame) and not df.empty:
    df = df[pd.to_numeric(df['close'], errors='coerce').notnull()]
    df['close'] = df['close'].astype(float)

# Calcul des indicateurs AVANT l'affichage (corrige KeyError)
if isinstance(df, pd.DataFrame) and not df.empty:
    df = compute_indicators(df)
    df = compute_score(df)
    df = detect_divergence(df)
    df = filter_signals(df)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")
interval = st.sidebar.selectbox("Intervalle des données", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
score_threshold = st.sidebar.slider("Seuil du score pour signal fort", min_value=1, max_value=4, value=2)

# --- FEAR & GREED INDEX (sidebar) ---
def get_fear_greed_index():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            value = int(data['data'][0]['value'])
            label = data['data'][0]['value_classification']
            return value, label
    except Exception:
        pass
    return None, None

# --- FEAR & GREED INDEX GAUGE (Plotly) ---
def plot_fear_greed_gauge(index_value):
    if index_value <= 25:
        label = "Extreme Fear"
        color = "#d90429"
    elif index_value <= 45:
        label = "Fear"
        color = "#ffba08"
    elif index_value < 55:
        label = "Neutral"
        color = "#f7b801"
    elif index_value < 75:
        label = "Greed"
        color = "#43aa8b"
    else:
        label = "Extreme Greed"
        color = "#008000"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = index_value,
        number = {'suffix': f" ({label})", 'font': {'color': color, 'size': 28}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#888'},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': "#222831",
            'steps': [
                {'range': [0, 25], 'color': '#d90429'},
                {'range': [25, 45], 'color': '#ffba08'},
                {'range': [45, 55], 'color': '#f7b801'},
                {'range': [55, 75], 'color': '#43aa8b'},
                {'range': [75, 100], 'color': '#008000'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 6},
                'thickness': 0.8,
                'value': index_value
            }
        },
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': '<b>Fear & Greed Index</b>', 'font': {'size': 22, 'color': color}}
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="#222831", font=dict(color="#FFD700"))
    return fig

fear_value, fear_label = get_fear_greed_index()
if fear_value is not None:
    sidebar_gauge = plot_fear_greed_gauge(fear_value)
    st.sidebar.plotly_chart(sidebar_gauge, use_container_width=True)
else:
    st.sidebar.markdown(
        "<div style='background-color:#222831;padding:0.7em 1em;border-radius:8px;margin-bottom:1em;'>"
        "<span style='color:#888;font-size:1.1em;'>Fear & Greed Index : N/A</span></div>", unsafe_allow_html=True)

# --- OPTIONS D'AFFICHAGE DES INDICATEURS ---
show_sma = st.sidebar.checkbox("Afficher SMA20/SMA50", value=True)
show_rsi = st.sidebar.checkbox("Afficher RSI", value=True)
show_macd = st.sidebar.checkbox("Afficher MACD", value=True)
show_vol = st.sidebar.checkbox("Afficher Volumes", value=True)

# --- HISTORICAL PERIOD SELECTION ---
import datetime
# Toujours afficher par défaut les 5 dernières années (ou tout l'historique si moins de 5 ans)
if isinstance(df, pd.DataFrame) and not df.empty:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
        df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
    min_date = df['open_time'].min()
    max_date = df['open_time'].max()
    st.sidebar.info(f"Données de {min_date} à {max_date}")
    now = pd.Timestamp(datetime.datetime.now())
    five_years_ago = now - pd.DateOffset(years=5)
    # Par défaut, on prend 5 ans ou tout l'historique si moins de 5 ans
    default_start = max(min_date, five_years_ago)
    default_end = max_date
    # Sélecteur de plage de dates, navigation sur tout l'historique
    period = st.sidebar.date_input(
        "Période d'analyse (début et fin)",
        value=(default_start.date(), default_end.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key="period_date_input"
    )
    # Correction robuste : toujours obtenir deux dates scalaires
    if isinstance(period, (tuple, list)) and len(period) == 2:
        period_start, period_end = period
    elif isinstance(period, (tuple, list)) and len(period) == 1:
        period_start = period_end = period[0]
    else:
        period_start = period_end = period
    period_start = pd.to_datetime(period_start)
    period_end = pd.to_datetime(period_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    # Filtrage sur la période
    df_period = df[(df['open_time'] >= period_start) & (df['open_time'] <= period_end)].copy()
else:
    df_period = df

# --- PRIX EN TEMPS RÉEL BTC/USDT ---
def get_realtime_price():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return float(resp.json()["price"])
    except Exception as e:
        return None
    return None

realtime_price = get_realtime_price()
st.markdown("""
<div style='background-color:#222831;padding:1em;border-radius:8px;margin-bottom:1em;'>
    <span style='color:#FFD700;font-size:2em;font-weight:bold;'>BTC/USDT</span>
    <span style='color:#00FF00;font-size:2em;font-weight:bold;margin-left:1em;'>
        {}</span>
</div>""".format(f"{realtime_price:,.2f} $" if realtime_price else "N/A"), unsafe_allow_html=True)

# --- TABS FOR NAVIGATION ---
tabs = st.tabs([
    "📈 Indicateurs & Graphiques",
    "🚨 Signaux & Alertes",
    "📰 Actualités",
    "🧪 Backtesting",
    "⚡ Gestion avancée du capital",
    "🤖 Prédiction ML",
    "📊 Comparaison ML",
    "🔗 Corrélation"
])

# --- INDICATEURS & GRAPHIQUES ---
with tabs[0]:
    st.header("Visualisation des indicateurs techniques")
    if isinstance(df_period, pd.DataFrame) and not df_period.empty:
        if show_sma:
            st.subheader("Graphique Prix + SMA20/SMA50")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_period['open_time'],
                open=df_period['open'],
                high=df_period['high'],
                low=df_period['low'],
                close=df_period['close'],
                name='Prix',
                increasing_line_color='white',
                decreasing_line_color='white',
                increasing_fillcolor='white',
                decreasing_fillcolor='white',
                line=dict(width=1)
            ))
            fig.add_trace(go.Scatter(x=df_period['open_time'], y=df_period['SMA20'], line=dict(color='blue', width=1), name='SMA20'))
            fig.add_trace(go.Scatter(x=df_period['open_time'], y=df_period['SMA50'], line=dict(color='orange', width=1), name='SMA50'))
            # --- Ajout de la courbe de divergence (1/-1/0) ---
            if 'divergence' in df_period.columns:
                fig.add_trace(go.Scatter(
                    x=df_period['open_time'],
                    y=df_period['divergence'],
                    mode='lines',
                    line=dict(color='purple', width=2, dash='dot'),
                    name='Courbe de divergence',
                    yaxis='y2'
                ))
                # Ajout d'un axe secondaire pour la divergence
                fig.update_layout(
                    yaxis2=dict(
                        title='Divergence',
                        overlaying='y',
                        side='right',
                        range=[-1.2, 1.2],
                        showgrid=False,
                        tickvals=[-1, 0, 1],
                        ticktext=['Baissière', 'Aucune', 'Haussière']
                    )
                )
            # --- Ajout des signaux forts sur le graphique ---
            signals = df_period[(df_period['score'] >= score_threshold) & (df_period['divergence'] == 1)]
            max_signals = 30  # Limite d'affichage pour éviter surcharge
            if len(signals) > max_signals:
                signals = signals.tail(max_signals)
            for _, row in signals.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['open_time']],
                    y=[row['close']],
                    mode='markers+text',
                    marker=dict(symbol='arrow-bar-up', color='green', size=16),
                    text=["Signal"],
                    textposition="top center",
                    name='Signal fort',
                    showlegend=False
                ))
            fig.update_layout(xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
        if show_rsi:
            st.subheader("RSI 14 périodes")
            fig_rsi = px.line(df_period, x='open_time', y='RSI14', title='RSI 14 périodes')
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig_rsi, use_container_width=True)
        if show_macd:
            st.subheader("MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df_period['open_time'], y=df_period['MACD'], name='MACD', line=dict(color='purple')))
            fig_macd.add_trace(go.Scatter(x=df_period['open_time'], y=df_period['MACD_signal'], name='Signal', line=dict(color='orange')))
            st.plotly_chart(fig_macd, use_container_width=True)
        if show_vol:
            st.subheader("Volume moyen (20 et 50 périodes)")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=df_period['open_time'], y=df_period['volume'], name='Volume', marker_color='gray', opacity=0.5))
            fig_vol.add_trace(go.Scatter(x=df_period['open_time'], y=df_period['VOL20'], name='VOL20', line=dict(color='blue')))
            fig_vol.add_trace(go.Scatter(x=df_period['open_time'], y=df_period['VOL50'], name='VOL50', line=dict(color='orange')))
            fig_vol.update_layout(height=300)
            st.plotly_chart(fig_vol, use_container_width=True)
        # --- NOUVEAU : graphique séparé pour la divergence ---
        if 'divergence' in df_period.columns:
            st.subheader("Courbe de divergence (prix vs MACD)")
            import plotly.graph_objects as go
            fig_div = go.Figure()
            fig_div.add_trace(go.Scatter(
                x=df_period['open_time'],
                y=df_period['divergence'],
                mode='lines+markers',
                line=dict(color='purple', width=2, dash='dot'),
                marker=dict(size=6),
                name='Divergence'
            ))
            fig_div.update_layout(
                yaxis=dict(
                    title='Divergence',
                    range=[-1.2, 1.2],
                    tickvals=[-1, 0, 1],
                    ticktext=['Baissière', 'Aucune', 'Haussière']
                ),
                xaxis_title='Date',
                height=250,
                template='plotly_dark'
            )
            st.plotly_chart(fig_div, use_container_width=True)
    else:
        st.warning(f"Impossible de charger les données : {error if error else 'Aucune donnée trouvée.'}")

# --- SIGNES & ALERTES ---
with tabs[1]:
    st.header("Signaux filtrés (scoring + divergence)")
    if isinstance(df_period, pd.DataFrame) and not df_period.empty:
        filtered = df_period[(df_period['score'] >= score_threshold) & (df_period['divergence'] == 1)]
        if not filtered.empty:
            st.dataframe(filtered[['open_time', 'close', 'score', 'divergence']].tail(1))
            last_signal = filtered.iloc[-1]
            last_time = pd.to_datetime(last_signal['open_time'])
            # Suppression de toute logique d'alerte personnalisée (show_alerts, alert_types, ballons, sons)
            # Affichage simple du dernier signal fort détecté
            st.success(f"🚨 Dernier signal fort détecté : {last_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("Aucun signal fort détecté sur la période.")
    else:
        st.warning(f"Impossible de charger les données : {error if error else 'Aucune donnée trouvée.'}")

# --- ACTUALITÉS ---
with tabs[2]:
    st.header("Actualités Bitcoin (Cointelegraph)")
    try:
        news_feed = feedparser.parse('https://cointelegraph.com/rss/tag/bitcoin')
        if news_feed.entries:
            for entry in news_feed.entries[:5]:
                st.markdown(f"- [{entry.title}]({entry.link})")
        else:
            st.info("Aucune actualité trouvée.")
    except Exception as e:
        st.warning(f"Erreur lors de la récupération ou l'analyse des actualités : {e}")
    st.header("Événements géopolitiques et actualités mondiales (Reuters, filtré)")
    KEYWORDS = [
        "bitcoin", "btc", "crypto", "cryptocurrency", "blockchain", "fed", "banque centrale", "dollar", "binance", "coinbase", "ethereum", "etf", "inflation", "interest rate", "central bank", "regulation", "mining", "halving", "trading", "finance", "marché", "bourse", "nasdaq", "sec", "gouvernement", "politique", "sanction", "usa", "chine", "russie", "europe", "macro", "monétaire", "taux", "réserve", "banque mondiale", "imf", "g7", "g20"
    ]
    try:
        world_news = feedparser.parse('https://feeds.reuters.com/Reuters/worldNews')
        filtered = [entry for entry in world_news.entries[:20] if any(k.lower() in entry.title.lower() for k in KEYWORDS)]
        if filtered:
            for entry in filtered[:5]:
                st.markdown(f"- [{entry.title}]({entry.link})")
        else:
            st.info("Aucun événement géopolitique pertinent trouvé (bitcoin, fed, dollar, etc.).")
    except Exception as e:
        st.warning(f"Erreur lors de la récupération des actualités mondiales : {e}")

# --- BACKTESTING ---
with tabs[3]:
    st.header("Backtesting de la stratégie")
    from backtest import run_backtest, optimize_parameters
    if isinstance(df_period, pd.DataFrame) and not df_period.empty:
        st.subheader("Paramètres de backtest")
        capital = st.number_input("Capital initial", min_value=1000, max_value=1000000, value=10000, step=1000)
        fee = st.number_input("Frais de transaction (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100
        take_profit = st.number_input("Take profit (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.5) / 100 or None
        stop_loss = st.number_input("Stop loss (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.5) / 100 or None
        strategy = st.selectbox(
            "Stratégie",
            ["buy_and_hold", "signal", "sma_cross", "rsi_extreme", "random"],
            format_func=lambda x: {
                "buy_and_hold": "Buy & Hold (achat au début, vente à la fin)",
                "signal": "Signal fort (score + divergence)",
                "sma_cross": "Croisement SMA20/SMA50",
                "rsi_extreme": "RSI extrême (achat <30, vente >70)",
                "random": "Aléatoire (benchmark)"
            }[x],
            index=1
        )
        st.write("Score threshold utilisé :", score_threshold)
        if st.button("Lancer le backtest"):
            equity_curve, trades, stats = run_backtest(
                df_period, strategy=strategy, score_threshold=score_threshold, initial_capital=capital, fee=fee,
                take_profit=take_profit, stop_loss=stop_loss
            )
            st.subheader("Courbe de capital")
            st.line_chart(equity_curve.set_index('open_time')['equity'])
            st.subheader("Statistiques")
            st.write(stats)
            st.subheader("Trades")
            st.dataframe(trades)
            st.subheader("Distribution des PnL des trades")
            if not trades.empty:
                st.bar_chart(trades['pnl'])
        st.subheader("Optimisation du score_threshold")
        score_range = st.multiselect("Tester les valeurs de score_threshold", [1, 1.5, 2, 2.5, 3], default=[1,2,3])
        if st.button("Optimiser"):
            results = optimize_parameters(df_period, score_range=score_range, fee=fee)
            st.dataframe(results)
            st.line_chart(results.set_index('score_threshold')['total_return'])
    else:
        st.warning("Aucune donnée pour le backtest.")

# --- GESTION AVANCÉE DU CAPITAL ---
with tabs[4]:
    st.header("Gestion avancée du capital - Backtest personnalisé")
    from backtest import run_backtest
    if isinstance(df_period, pd.DataFrame) and not df_period.empty:
        st.subheader("Paramètres avancés")
        capital = st.number_input("Capital initial", min_value=1000, max_value=1000000, value=10000, step=1000, key="capital_advanced")
        fee = st.number_input("Frais de transaction (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="fee_advanced") / 100
        invest_pct = st.slider("% investi à chaque trade", min_value=0, max_value=100, value=50, step=5, key="invest_pct_advanced") / 100
        reinvest_loss_pct = st.slider("% à réinvestir après une perte", min_value=0, max_value=100, value=25, step=5, key="reinvest_loss_pct_advanced") / 100
        withdraw_gain_pct = st.slider("% des gains retirés à chaque gain", min_value=0, max_value=100, value=20, step=5, key="withdraw_gain_pct_advanced") / 100
        max_drawdown = st.slider("Seuil critique de perte globale (drawdown max, %)", min_value=5, max_value=90, value=30, step=5, key="max_drawdown_advanced") / 100
        take_profit = st.number_input("Take profit (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.5, key="tp_advanced") / 100 or None
        stop_loss = st.number_input("Stop loss (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.5, key="sl_advanced") / 100 or None
        strategy = st.selectbox(
            "Stratégie avancée",
            ["advanced_risk", "signal", "buy_and_hold", "sma_cross", "rsi_extreme", "random"],
            format_func=lambda x: {
                "advanced_risk": "Gestion avancée du capital (avec réinvestissement/retrait)",
                "signal": "Signal fort (score + divergence)",
                "buy_and_hold": "Buy & Hold (achat au début, vente à la fin)",
                "sma_cross": "Croisement SMA20/SMA50",
                "rsi_extreme": "RSI extrême (achat <30, vente >70)",
                "random": "Aléatoire (benchmark)"
            }[x],
            index=0
        )
        st.write("Score threshold utilisé :", score_threshold)
        if st.button("Lancer le backtest avancé"):
            kwargs = dict(
                invest_pct=invest_pct, reinvest_loss_pct=reinvest_loss_pct, withdraw_gain_pct=withdraw_gain_pct, max_drawdown=max_drawdown
            ) if strategy == 'advanced_risk' else {}
            equity_curve, trades, stats = run_backtest(
                df_period, strategy=strategy, score_threshold=score_threshold, initial_capital=capital, fee=fee,
                take_profit=take_profit, stop_loss=stop_loss, **kwargs
            )
            # --- Ajout : calcul de la courbe Buy & Hold pour comparaison ---
            if strategy != 'buy_and_hold':
                equity_bh, _, _ = run_backtest(
                    df_period, strategy='buy_and_hold', initial_capital=capital, fee=fee
                )
            else:
                equity_bh = None
            st.subheader("Courbe de capital")
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve['open_time'],
                y=equity_curve['equity'],
                mode='lines',
                name=f"Stratégie : {strategy}",
                line=dict(width=2, color='royalblue')
            ))
            if equity_bh is not None:
                fig.add_trace(go.Scatter(
                    x=equity_bh['open_time'],
                    y=equity_bh['equity'],
                    mode='lines',
                    name="Buy & Hold (marché)",
                    line=dict(width=2, color='orange', dash='dash')
                ))
            fig.update_layout(xaxis_title='Date', yaxis_title='Capital', template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Statistiques")
            st.write(stats)
            st.subheader("Trades")
            st.dataframe(trades)
            st.subheader("Distribution des PnL des trades")
            if not trades.empty:
                st.bar_chart(trades['pnl'])
    else:
        st.warning("Aucune donnée pour le backtest.")

# --- MACHINE LEARNING : Prédiction du prix de clôture à N jours ---
with tabs[5]:
    st.header("Prédiction du prix de clôture à N jours (RandomForest, XGBoost, LinearRegression)")
    horizon = 7
    # Sélecteur de période spécifique pour le ML
    if isinstance(df, pd.DataFrame) and not df.empty:
        min_date = df['open_time'].min()
        max_date = df['open_time'].max() - pd.Timedelta(days=horizon)
        st.markdown("**Sélectionnez la période d'analyse pour le Machine Learning**")
        ml_period = st.date_input(
            "Période ML (début et fin)",
            value=(max(min_date, max_date - pd.Timedelta(days=365)).date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="ml_period_date_input"
        )
        if isinstance(ml_period, (tuple, list)) and len(ml_period) == 2:
            ml_start, ml_end = ml_period
        elif isinstance(ml_period, (tuple, list)) and len(ml_period) == 1:
            ml_start = ml_end = ml_period[0]
        else:
            ml_start = ml_end = ml_period
        ml_start = pd.to_datetime(ml_start)
        ml_end = pd.to_datetime(ml_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df_ml = df[(df['open_time'] >= ml_start) & (df['open_time'] <= ml_end)].copy()
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'SMA20', 'SMA50', 'RSI14', 'MACD', 'MACD_signal', 'VOL20', 'VOL50'
        ]
        for col in features:
            df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
        df_ml = df_ml.dropna(subset=features)
        # Target : close dans N jours
        df_ml['target'] = df_ml['close'].shift(-horizon)
        df_ml = df_ml.dropna(subset=['target'])
        X = df_ml[features].reset_index(drop=True)
        y = df_ml['target'].reset_index(drop=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        df_test = df_ml.iloc[X_test.index].reset_index(drop=True)
        # RandomForest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        # XGBoost
        xgbr = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgbr.fit(X_train, y_train)
        y_pred_xgb = xgbr.predict(X_test)
        # LinearRegression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        # Affichage courbes
        st.subheader(f"Comparaison prix réel vs prédiction (test, horizon = {horizon} jours)")
        # Correction alignement index/longueur pour DataFrame de prédiction ML
        min_len = min(len(df_test), len(y_test), len(y_pred_rf), len(y_pred_xgb), len(y_pred_lr))
        df_test = df_test.iloc[:min_len].reset_index(drop=True)
        y_test = y_test.iloc[:min_len].reset_index(drop=True)
        y_pred_rf = y_pred_rf[:min_len]
        y_pred_xgb = y_pred_xgb[:min_len]
        y_pred_lr = y_pred_lr[:min_len]
        df_pred = pd.DataFrame({
            'Date': df_test['open_time'],
            'Prix réel': y_test,
            'RandomForest': y_pred_rf,
            'XGBoost': y_pred_xgb,
            'LinearRegression': y_pred_lr
        })
        st.line_chart(df_pred.set_index('Date'))
        # --- Affichage du vrai prix à la clôture pour la dernière date testée ---
        last_true = y_test.iloc[-1]
        last_X = df_ml[features].iloc[[-1]]
        st.subheader(f"Dernières prédictions (dans {horizon} jours)")
        st.write(f"Prix réel attendu : {last_true:,.2f} $")
        st.write(f"RandomForest : {rf.predict(last_X)[0]:,.2f} $")
        st.write(f"XGBoost : {xgbr.predict(last_X)[0]:,.2f} $")
        st.write(f"LinearRegression : {lr.predict(last_X)[0]:,.2f} $")
        # --- Tableau de classement des modèles par précision ---
        import numpy as np
        metrics = [
            {
                'Modèle': 'RandomForest',
                'MSE': mean_squared_error(y_test, y_pred_rf),
                'MAE': mean_absolute_error(y_test, y_pred_rf),
                'R2': r2_score(y_test, y_pred_rf)
            },
            {
                'Modèle': 'XGBoost',
                'MSE': mean_squared_error(y_test, y_pred_xgb),
                'MAE': mean_absolute_error(y_test, y_pred_xgb),
                'R2': r2_score(y_test, y_pred_xgb)
            },
            {
                'Modèle': 'LinearRegression',
                'MSE': mean_squared_error(y_test, y_pred_lr),
                'MAE': mean_absolute_error(y_test, y_pred_lr),
                'R2': r2_score(y_test, y_pred_lr)
            }
        ]
        df_metrics = pd.DataFrame(metrics)
        df_metrics = df_metrics.sort_values('MAE')
        st.subheader("Classement des modèles par précision (MAE croissant)")
        st.dataframe(df_metrics.reset_index(drop=True))
        # Stocker pour comparaison
        st.session_state['ml_results'] = {
            'y_test': y_test,
            'y_pred_rf': y_pred_rf,
            'y_pred_xgb': y_pred_xgb,
            'y_pred_lr': y_pred_lr
        }
        # --- Prédiction sur le futur réel (prochaine date disponible + horizon) ---
        st.markdown("---")
        st.subheader(f"Prédiction sur le futur réel (prochaine date + {horizon} jours)")
        last_row = df[features].iloc[[-1]].copy()
        for col in features:
            last_row[col] = pd.to_numeric(last_row[col], errors='coerce')
        last_row = last_row.astype(float)
        pred_rf = rf.predict(last_row)[0]
        pred_xgb = xgbr.predict(last_row)[0]
        pred_lr = lr.predict(last_row)[0]
        future_date = df['open_time'].max() + pd.Timedelta(days=horizon)
        st.write(f"Date prédite : {future_date.date()}")
        st.write(f"RandomForest : {pred_rf:,.2f} $")
        st.write(f"XGBoost : {pred_xgb:,.2f} $")
        st.write(f"LinearRegression : {pred_lr:,.2f} $")
    else:
        st.warning("Pas assez de données pour la prédiction ML sur la période sélectionnée.")

# --- COMPARAISON DES MODELES ML ---
with tabs[6]:
    st.header("Comparaison des modèles ML (prédiction à 7 jours)")
    results = st.session_state.get('ml_results', None)
    if results:
        y_test = results['y_test']
        y_pred_rf = results['y_pred_rf']
        y_pred_xgb = results['y_pred_xgb']
        y_pred_lr = results['y_pred_lr']
        st.write("**RandomForest**")
        st.write(f"MSE : {mean_squared_error(y_test, y_pred_rf):.2f}")
        st.write(f"MAE : {mean_absolute_error(y_test, y_pred_rf):.2f}")
        st.write(f"R² : {r2_score(y_test, y_pred_rf):.3f}")
        st.write("**XGBoost**")
        st.write(f"MSE : {mean_squared_error(y_test, y_pred_xgb):.2f}")
        st.write(f"MAE : {mean_absolute_error(y_test, y_pred_xgb):.2f}")
        st.write(f"R² : {r2_score(y_test, y_pred_xgb):.3f}")
        st.write("**LinearRegression**")
        st.write(f"MSE : {mean_squared_error(y_test, y_pred_lr):.2f}")
        st.write(f"MAE : {mean_absolute_error(y_test, y_pred_lr):.2f}")
        st.write(f"R² : {r2_score(y_test, y_pred_lr):.3f}")
        # Ajout du graphique d'erreur absolue par date
        st.subheader("Erreur absolue par date (test)")
        import numpy as np
        abs_err_rf = np.abs(y_test - y_pred_rf)
        abs_err_xgb = np.abs(y_test - y_pred_xgb)
        abs_err_lr = np.abs(y_test - y_pred_lr)
        df_err = pd.DataFrame({
            'Date': y_test.index,
            'Erreur RF': abs_err_rf,
            'Erreur XGB': abs_err_xgb,
            'Erreur LR': abs_err_lr
        })
        df_err['Date'] = df_err['Date'].astype(str)
        df_err = df_err.set_index('Date')
        st.line_chart(df_err, use_container_width=True)
    else:
        st.info("Lance d'abord une prédiction dans l'onglet ML.")

# --- ANALYSE AUTOMATISÉE DES SIGNAUX ET DIVERGENCES ---
if isinstance(df, pd.DataFrame) and not df.empty:
    import datetime
    now = pd.Timestamp(datetime.datetime.now())
    last_30d = now - pd.Timedelta(days=30)
    recent = df[df['open_time'] >= last_30d]
    n_signaux = recent[(recent['score'] >= score_threshold) & (recent['divergence'] == 1)].shape[0]
    n_divergences = recent[recent['divergence'] == 1].shape[0]
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Analyse 30 derniers jours :**")
    st.sidebar.markdown(f"- Signaux forts : {n_signaux}")
    st.sidebar.markdown(f"- Divergences détectées : {n_divergences}")
    if n_signaux > 0:
        last_signal = recent[(recent['score'] >= score_threshold) & (recent['divergence'] == 1)]['open_time'].max()
        st.sidebar.markdown(f"- Dernier signal fort : {last_signal}")
    else:
        st.sidebar.markdown("- Aucun signal fort sur 30j")
    # Tableau récapitulatif
    st.sidebar.markdown("**Tableau signaux/divergences (30j)**")
    st.sidebar.dataframe(recent[['open_time','score','divergence']].tail(30), use_container_width=True)

# --- ANALYSE AVANCÉE DES DIVERGENCES ---
if isinstance(df, pd.DataFrame) and not df.empty and 'divergence_type' in df.columns:
    recent = df[df['open_time'] >= (pd.Timestamp(datetime.datetime.now()) - pd.Timedelta(days=30))]
    n_div_bull = recent[recent['divergence_type'] == 'bullish'].shape[0]
    n_div_bear = recent[recent['divergence_type'] == 'bearish'].shape[0]
    last_bull = recent[recent['divergence_type'] == 'bullish']['open_time'].max() if n_div_bull > 0 else None
    last_bear = recent[recent['divergence_type'] == 'bearish']['open_time'].max() if n_div_bear > 0 else None
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Analyse divergences avancée (30j)**")
    st.sidebar.markdown(f"- Divergences haussières : {n_div_bull}")
    st.sidebar.markdown(f"- Divergences baissières : {n_div_bear}")
    if last_bull:
        st.sidebar.markdown(f"- Dernière haussière : {last_bull}")
    if last_bear:
        st.sidebar.markdown(f"- Dernière baissière : {last_bear}")
    st.sidebar.markdown("**Détails divergences (30j)**")
    st.sidebar.dataframe(recent[['open_time','score','divergence','divergence_type']].tail(30), use_container_width=True)

# --- CORRELATION AVEC D'AUTRES ACTIFS ---
with tabs[7]:
    st.header("Analyse de corrélation avec d'autres actifs")
    st.markdown("Comparez le Bitcoin à d'autres marchés (S&P500, or, ETH, etc.) sur différentes périodes.")
    import yfinance as yf
    default_tickers = 'BTC-USD,ETH-USD,^GSPC,GC=F'
    tickers_input = st.text_input("Tickers Yahoo Finance à comparer (séparés par virgule ou espace)", value=default_tickers)
    tickers = [t.strip() for t in tickers_input.replace(',', ' ').split() if t.strip()]
    period_corr = st.selectbox("Période de corrélation", ["30j", "90j", "1an", "Tout l'historique"], index=0)
    rolling_window = st.number_input("Fenêtre rolling (jours)", min_value=3, max_value=180, value=30, step=1)
    if isinstance(df, pd.DataFrame) and not df.empty and tickers:
        btc = df[['open_time', 'close']].copy()
        btc = btc.rename(columns={'close': 'BTC-USD'})
        btc = btc.set_index('open_time')
        data = btc.copy()
        for ticker in tickers:
            if ticker == 'BTC-USD':
                continue  # déjà présent
            try:
                yf_data = yf.download(ticker, start=btc.index.min(), end=btc.index.max())
                if not yf_data.empty:
                    yf_data = yf_data[['Close']].rename(columns={'Close': ticker})
                    yf_data.index = pd.to_datetime(yf_data.index)
                    data = data.join(yf_data, how='left')
            except Exception as e:
                st.warning(f"Erreur récupération {ticker}: {e}")
        # Nettoyage
        data = data.dropna()
        # Sélection période
        if period_corr == "30j":
            data = data[data.index >= (data.index.max() - pd.Timedelta(days=30))]
        elif period_corr == "90j":
            data = data[data.index >= (data.index.max() - pd.Timedelta(days=90))]
        elif period_corr == "1an":
            data = data[data.index >= (data.index.max() - pd.Timedelta(days=365))]
        # Calcul corrélation
        corr = data.corr()
        st.subheader("Matrice de corrélation")
        st.dataframe(corr, use_container_width=True)
        # Rolling correlation (fenêtre personnalisée)
        st.subheader(f"Corrélation glissante ({rolling_window} jours)")
        for ticker in tickers:
            if ticker == 'BTC-USD':
                continue
            if ticker not in data.columns:
                st.warning(f"Données manquantes pour {ticker}, impossible de calculer la corrélation glissante.")
                continue
            rolling_corr = data['BTC-USD'].rolling(window=rolling_window).corr(data[ticker])
            st.line_chart(rolling_corr.rename(f'BTC-USD vs {ticker}'))
    else:
        st.info("Saisissez au moins un ticker et assurez-vous que les données Bitcoin sont chargées.")
