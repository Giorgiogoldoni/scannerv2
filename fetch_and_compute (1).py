"""
RAPTOR SCANNER v2 — fetch_and_compute.py
Legge universe.csv, scarica OHLCV da Yahoo Finance,
calcola indicatori tecnici allineati al sistema Raptor,
genera signals.json e signals_history.json.

Segnali: BUY1 / BUY2 / BUY3 / SELL / WATCH / RANGING
Score: calibrato per categoria (Azionario, Obbligazionario, Leva, Monetario, Materie Prime)
History: ricostruita sui 200gg storici, poi incrementale ogni notte
"""

import os, json, time, logging, traceback, math, csv
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
import requests
import numpy as np

# ── CONFIGURAZIONE ─────────────────────────────────────────────────────────────
CONFIG = {
    "history_days":          200,
    "min_candles":           60,
    "max_retries":           3,
    "retry_delay":           2.0,
    "cache_dir":             "data/cache",
    "signals_file":          "data/signals.json",
    "history_file":          "data/signals_history.json",
    "log_file":              "logs/run.log",
    "cache_max_age_hours":   20,
    "max_workers":           10,
    "universe_file":         "universe.csv",

    # ER soglie per categoria
    "er_threshold": {
        "Leva Long":          0.25,
        "Short":              0.25,
        "default":            0.20,
    },
    "er_trending": {
        "Leva Long":          0.35,
        "Short":              0.35,
        "default":            0.28,
    },

    # BUY soglie score
    "buy1_score":            30,
    "buy2_score":            45,
    "buy3_score":            60,
    "sell_score":            25,

    # KAMA parametri
    "kama_n":                10,
    "kama_fast":             2,
    "kama_slow":             30,

    # Supertrend
    "st_period":             10,
    "st_multiplier":         3.0,

    # Bollinger
    "bb_period":             20,
    "bb_std":                2.0,

    # Volume
    "vol_period":            20,

    # AO periodi
    "ao_fast":               5,
    "ao_slow":               34,

    # RVI periodo
    "rvi_period":            10,

    # sys_rating soglie
    "rating_strong_buy":     60,
    "rating_buy":            45,
    "rating_neutral":        30,
}

CET = ZoneInfo("Europe/Rome")

# ── LOGGING ────────────────────────────────────────────────────────────────────
def setup_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(CONFIG["log_file"], encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

log = logging.getLogger(__name__)

# ── UNIVERSE ───────────────────────────────────────────────────────────────────
def load_universe() -> list:
    path = CONFIG["universe_file"]
    if not Path(path).exists():
        log.error(f"universe.csv non trovato: {path}")
        return []
    universe = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            universe.append((
                row['ticker'],
                row['name'],
                row['categoria'],
                row['is_leveraged'].strip().lower() == 'true',
                row['is_short'].strip().lower() == 'true',
                row['currency'],
            ))
    log.info(f"Universe caricato: {len(universe)} ticker")
    return universe

# ── YAHOO FINANCE ──────────────────────────────────────────────────────────────
def fetch_yahoo(ticker: str) -> dict | None:
    end   = datetime.utcnow()
    start = end - timedelta(days=CONFIG["history_days"])
    url   = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={int(start.timestamp())}&period2={int(end.timestamp())}"
        f"&events=history&includeAdjustedClose=true"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":     "application/json",
    }
    for attempt in range(1, CONFIG["max_retries"] + 1):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            data   = r.json()
            result = data.get("chart", {}).get("result")
            if not result:
                time.sleep(CONFIG["retry_delay"])
                continue
            ts        = result[0]["timestamp"]
            q         = result[0]["indicators"]["quote"][0]
            adj       = result[0]["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
            dates     = [datetime.utcfromtimestamp(t).strftime("%Y-%m-%d") for t in ts]
            closes    = [float(v) if v is not None else None for v in adj]
            opens     = [float(v) if v is not None else None for v in q["open"]]
            highs     = [float(v) if v is not None else None for v in q["high"]]
            lows      = [float(v) if v is not None else None for v in q["low"]]
            vols      = [int(v)   if v is not None else 0    for v in q["volume"]]
            valid = [(d,o,h,l,c,v) for d,o,h,l,c,v in
                     zip(dates,opens,highs,lows,closes,vols) if c is not None]
            if len(valid) < CONFIG["min_candles"]:
                return None
            d,o,h,l,c,v = zip(*valid)
            return {"dates":list(d),"opens":list(o),"highs":list(h),
                    "lows":list(l),"closes":list(c),"volumes":list(v)}
        except Exception as e:
            log.warning(f"{ticker}: attempt {attempt} — {e}")
            time.sleep(CONFIG["retry_delay"] * attempt)
    return None

# ── CACHE ──────────────────────────────────────────────────────────────────────
def cache_path(ticker: str) -> Path:
    return Path(CONFIG["cache_dir"]) / f"{ticker.replace('.','_')}.json"

def load_cache(ticker: str) -> dict | None:
    p = cache_path(ticker)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            cached = json.load(f)
        age_h = (datetime.utcnow() - datetime.fromisoformat(
            cached.get("cached_at","2000-01-01"))).total_seconds() / 3600
        if age_h < CONFIG["cache_max_age_hours"]:
            return cached.get("ohlcv")
    except Exception:
        pass
    return None

def save_cache(ticker: str, ohlcv: dict):
    p = cache_path(ticker)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"cached_at": datetime.utcnow().isoformat(), "ohlcv": ohlcv}, f)

def get_ohlcv(ticker: str) -> dict | None:
    cached = load_cache(ticker)
    if cached:
        return cached
    ohlcv = fetch_yahoo(ticker)
    if ohlcv:
        save_cache(ticker, ohlcv)
    return ohlcv

# ── INDICATORI ─────────────────────────────────────────────────────────────────
def safe_last(arr, default=None):
    valid = arr[~np.isnan(arr)]
    return float(valid[-1]) if len(valid) > 0 else default

def calc_ema(closes, period):
    ema = np.full(len(closes), np.nan)
    k   = 2.0 / (period + 1)
    s   = period - 1
    if s >= len(closes):
        return ema
    ema[s] = np.mean(closes[:period])
    for i in range(s+1, len(closes)):
        ema[i] = closes[i] * k + ema[i-1] * (1-k)
    return ema

def calc_sma(closes, period):
    sma = np.full(len(closes), np.nan)
    for i in range(period-1, len(closes)):
        sma[i] = np.mean(closes[i-period+1:i+1])
    return sma

def calc_er(closes, period=10):
    er = np.full(len(closes), np.nan)
    for i in range(period, len(closes)):
        direction  = abs(closes[i] - closes[i-period])
        volatility = np.sum(np.abs(np.diff(closes[i-period:i+1])))
        er[i] = direction / volatility if volatility != 0 else 0.0
    return er

def calc_kama(closes, n=10, fast=2, slow=30):
    er      = calc_er(closes, n)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    kama    = np.full(len(closes), np.nan)
    if n >= len(closes):
        return kama
    kama[n] = closes[n]
    for i in range(n+1, len(closes)):
        if not np.isnan(er[i]):
            sc      = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (closes[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    return kama

def calc_atr(highs, lows, closes, period=14):
    tr  = np.maximum(highs[1:]-lows[1:],
           np.maximum(np.abs(highs[1:]-closes[:-1]),
                      np.abs(lows[1:]-closes[:-1])))
    tr  = np.concatenate([[highs[0]-lows[0]], tr])
    atr = np.full(len(closes), np.nan)
    if period-1 >= len(closes):
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(closes)):
        atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
    return atr

def calc_sar(highs, lows, af_step=0.02, af_max=0.2):
    sar  = np.full(len(highs), np.nan)
    bull = np.full(len(highs), True)
    if len(highs) < 2:
        return sar, bull
    b  = True
    af = af_step
    ep = highs[0]
    sar[0] = lows[0]
    bull[0] = True
    for i in range(1, len(highs)):
        prev = sar[i-1]
        if b:
            sar[i] = prev + af * (ep - prev)
            sar[i] = min(sar[i], lows[i-1], lows[i-2] if i > 1 else lows[i-1])
            if lows[i] < sar[i]:
                b=False; af=af_step; ep=lows[i]; sar[i]=ep
            else:
                if highs[i] > ep:
                    ep=highs[i]; af=min(af+af_step, af_max)
        else:
            sar[i] = prev + af * (ep - prev)
            sar[i] = max(sar[i], highs[i-1], highs[i-2] if i > 1 else highs[i-1])
            if highs[i] > sar[i]:
                b=True; af=af_step; ep=highs[i]; sar[i]=ep
            else:
                if lows[i] < ep:
                    ep=lows[i]; af=min(af+af_step, af_max)
        bull[i] = b
    return sar, bull

def calc_ao(highs, lows):
    """Awesome Oscillator = SMA5(midpoint) - SMA34(midpoint)"""
    mid  = (highs + lows) / 2.0
    ao   = np.full(len(mid), np.nan)
    s5   = calc_sma(mid, CONFIG["ao_fast"])
    s34  = calc_sma(mid, CONFIG["ao_slow"])
    for i in range(len(ao)):
        if not np.isnan(s5[i]) and not np.isnan(s34[i]):
            ao[i] = s5[i] - s34[i]
    return ao

def calc_rvi(opens, highs, lows, closes, period=10):
    """Relative Vigor Index"""
    n   = len(closes)
    rvi = np.full(n, np.nan)
    sig = np.full(n, np.nan)
    if n < period + 4:
        return rvi, sig
    # numeratore = close-open pesato simmetrico
    num = np.full(n, np.nan)
    den = np.full(n, np.nan)
    for i in range(3, n):
        c0,o0,h0,l0 = closes[i],opens[i],highs[i],lows[i]
        c1,o1,h1,l1 = closes[i-1],opens[i-1],highs[i-1],lows[i-1]
        c2,o2,h2,l2 = closes[i-2],opens[i-2],highs[i-2],lows[i-2]
        c3,o3,h3,l3 = closes[i-3],opens[i-3],highs[i-3],lows[i-3]
        num[i] = ((c0-o0) + 2*(c1-o1) + 2*(c2-o2) + (c3-o3)) / 6.0
        den[i] = ((h0-l0) + 2*(h1-l1) + 2*(h2-l2) + (h3-l3)) / 6.0
    for i in range(period+3, n):
        s_num = np.sum(num[i-period+1:i+1])
        s_den = np.sum(den[i-period+1:i+1])
        rvi[i] = s_num / s_den if s_den != 0 else 0.0
    # Signal = symmetric MA(4) di RVI
    for i in range(3, n):
        if not np.isnan(rvi[i]) and not np.isnan(rvi[i-1]) and \
           not np.isnan(rvi[i-2]) and not np.isnan(rvi[i-3]):
            sig[i] = (rvi[i] + 2*rvi[i-1] + 2*rvi[i-2] + rvi[i-3]) / 6.0
    return rvi, sig

def calc_rsi(closes, period=14):
    delta    = np.diff(closes)
    gain     = np.where(delta > 0, delta, 0.0)
    loss     = np.where(delta < 0, -delta, 0.0)
    rsi      = np.full(len(closes), np.nan)
    if len(gain) < period:
        return rsi
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    for i in range(period, len(closes)-1):
        avg_gain = (avg_gain*(period-1) + gain[i]) / period
        avg_loss = (avg_loss*(period-1) + loss[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100.0
        rsi[i+1] = 100 - (100 / (1+rs))
    return rsi

def calc_adx(highs, lows, closes, period=14):
    atr  = calc_atr(highs, lows, closes, period)
    up   = highs[1:] - highs[:-1]
    down = lows[:-1] - lows[1:]
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)
    adx  = np.full(len(closes), np.nan)
    if len(closes) < period*2:
        return adx
    pdm14 = np.mean(pdm[:period])
    ndm14 = np.mean(ndm[:period])
    dx_arr = []
    for i in range(period, len(closes)-1):
        pdm14 = pdm14 - pdm14/period + pdm[i]
        ndm14 = ndm14 - ndm14/period + ndm[i]
        a14   = atr[i+1]
        pdi   = 100*pdm14/a14 if a14 != 0 else 0
        ndi   = 100*ndm14/a14 if a14 != 0 else 0
        dx    = 100*abs(pdi-ndi)/(pdi+ndi) if (pdi+ndi) != 0 else 0
        dx_arr.append(dx)
    if len(dx_arr) >= period:
        adx_val = np.mean(dx_arr[:period])
        idx     = period*2
        for i in range(period, len(dx_arr)):
            adx_val = (adx_val*(period-1) + dx_arr[i]) / period
            if idx < len(adx):
                adx[idx] = adx_val
            idx += 1
    return adx

def calc_bb(closes, period=20, n_std=2.0):
    mid   = np.full(len(closes), np.nan)
    upper = np.full(len(closes), np.nan)
    lower = np.full(len(closes), np.nan)
    width = np.full(len(closes), np.nan)
    for i in range(period-1, len(closes)):
        w      = closes[i-period+1:i+1]
        m,s    = np.mean(w), np.std(w, ddof=1)
        mid[i] = m
        upper[i] = m + n_std*s
        lower[i] = m - n_std*s
        width[i] = (upper[i]-lower[i])/m if m != 0 else 0
    return mid, upper, lower, width

def calc_vol_ratio(volumes, period=20):
    vr = np.full(len(volumes), np.nan)
    for i in range(period, len(volumes)):
        avg  = np.mean(volumes[i-period:i])
        vr[i] = volumes[i]/avg if avg != 0 else 1.0
    return vr

def momentum_pct(closes, bars):
    if len(closes) <= bars or closes[-bars-1] == 0:
        return 0.0
    return (closes[-1]/closes[-bars-1] - 1) * 100

# ── BAFF — bars above/below KAMA ──────────────────────────────────────────────
def calc_baff(closes, kama):
    """
    Conta quante barre consecutive il prezzo è sopra KAMA (positivo)
    o sotto KAMA (negativo). Ultimo valore.
    """
    n = len(closes)
    count = 0
    above = closes[-1] > kama[-1] if not np.isnan(kama[-1]) else False
    for i in range(n-1, -1, -1):
        if np.isnan(kama[i]):
            break
        if above:
            if closes[i] > kama[i]:
                count += 1
            else:
                break
        else:
            if closes[i] <= kama[i]:
                count -= 1
            else:
                break
    return count

# ── TREND KAMA — pendenza ─────────────────────────────────────────────────────
def kama_trend(kama, lookback=5):
    """
    VERDE: KAMA in salita da `lookback` barre
    ROSSO: KAMA in discesa da `lookback` barre
    GRIGIO: misto
    """
    valid = [(i,v) for i,v in enumerate(kama) if not np.isnan(v)]
    if len(valid) < lookback+1:
        return "GRIGIO"
    recent = [v for _,v in valid[-(lookback+1):]]
    rising  = all(recent[i] < recent[i+1] for i in range(len(recent)-1))
    falling = all(recent[i] > recent[i+1] for i in range(len(recent)-1))
    if rising:  return "VERDE"
    if falling: return "ROSSO"
    return "GRIGIO"

# ── SCORE PER CATEGORIA ───────────────────────────────────────────────────────
def compute_score(er, kama_above, sar_bull, ao_val, ao_rising,
                  rvi_bull, baff, k_pct, mom3m, adx, rsi,
                  trend, categoria):
    """
    Score calibrato per categoria:
    - Azionario/Tematico/Settoriale: score pieno 0-100
    - Leva Long: peso maggiore su ER e momentum, ignora RSI
    - Short: logica invertita (bear = positivo)
    - Obbligazionario: focus su momentum e SAR, penalizza ADX alto
    - Monetario: score fisso basso (non ha senso trend)
    - Materie Prime: standard ma ATR non considerato
    """
    cat = categoria

    # Monetari — score fisso
    if cat == "Monetario":
        return 10

    # Obbligazionario — score semplificato
    if cat == "Obbligazionario":
        s = 0.0
        s += min(30, max(0, (mom3m + 10) / 20 * 30))   # momentum 1m/3m pesato
        s += 20 if sar_bull else 0
        s += 15 if kama_above else 0
        s += 15 if ao_rising else 0
        s += 10 if rvi_bull else 0
        s += min(10, max(0, (er or 0) / 0.3 * 10))     # ER conta poco
        # penalità se ADX molto alto (obblig non amano trend forti)
        if adx and adx > 30:
            s *= 0.85
        if trend == "ROSSO":
            s *= 0.7
        return min(100, round(s))

    # Short — logica invertita (sar bear = bene, ao negativo = bene)
    if cat == "Short":
        s = 0.0
        s += min(25, max(0, (er or 0) / 0.5 * 25))
        s += 20 if not sar_bull else 0        # SAR bear = positivo per short
        s += 20 if (ao_val or 0) < 0 else 0  # AO negativo = positivo per short
        s += 15 if not kama_above else 0      # sotto KAMA = positivo per short
        s += 10 if not rvi_bull else 0
        s += min(10, max(0, abs(mom3m) / 20 * 10)) if mom3m < 0 else 0
        return min(100, round(s))

    # Leva Long — peso ER e momentum, penalità RSI overbought
    if cat == "Leva Long":
        s = 0.0
        s += min(35, max(0, (er or 0) / 0.5 * 35))   # ER peso alto
        s += 20 if kama_above else 0
        s += 15 if sar_bull else 0
        s += min(15, max(0, (mom3m + 30) / 60 * 15)) # momentum
        s += 10 if ao_rising else 0
        s += 5  if rvi_bull else 0
        if rsi and rsi > 75:                           # overbought penalità
            s *= 0.85
        if trend == "ROSSO":
            s *= 0.6
        return min(100, round(s))

    # Materie Prime
    if cat == "Materie Prime":
        s = 0.0
        s += min(25, max(0, (er or 0) / 0.5 * 25))
        s += 20 if kama_above else 0
        s += 15 if sar_bull else 0
        s += 15 if ao_rising else 0
        s += 10 if rvi_bull else 0
        s += min(10, max(0, (mom3m + 30) / 60 * 10))
        s += min(5,  max(0, baff * 1.0)) if baff > 0 else 0
        if trend == "ROSSO":
            s *= 0.6
        return min(100, round(s))

    # Azionario / Tematico / Settoriale — score pieno
    s = 0.0
    s += min(25, max(0, (er or 0) / 0.5 * 25))        # ER
    s += 20 if kama_above else 0                        # sopra KAMA
    s += 15 if sar_bull else 0                          # SAR
    s += 10 if ao_rising else 0                         # AO in miglioramento
    s += 10 if rvi_bull else 0                          # RVI
    s += min(10, max(0, (mom3m + 30) / 60 * 10))       # momentum 3m
    s += min(5,  max(0, (adx or 0) / 60 * 5))          # ADX
    s += min(5,  max(0, baff * 0.5)) if baff > 0 else 0 # baff
    if rsi and rsi > 70:
        s *= 0.9
    if trend == "ROSSO":
        s *= 0.6
    return min(100, round(s))

# ── SYS_RATING ────────────────────────────────────────────────────────────────
def compute_sys_rating(score, kama_above, sar_bull, ao_val,
                       rvi_bull, trend, categoria):
    """
    6 condizioni bull (Short: invertite):
    1. SAR bull
    2. AO > 0
    3. Prezzo > KAMA
    4. RVI bull
    5. Trend VERDE
    6. score >= 45
    """
    if categoria == "Monetario":
        return "NEUTRAL", 3, 3

    if categoria == "Short":
        # inverti tutto
        conds = [
            not sar_bull,
            (ao_val or 0) < 0,
            not kama_above,
            not rvi_bull,
            trend == "ROSSO",
            score >= 45,
        ]
    else:
        conds = [
            sar_bull,
            (ao_val or 0) > 0,
            kama_above,
            rvi_bull,
            trend == "VERDE",
            score >= 45,
        ]

    bull_count = sum(conds)
    sell_count = 6 - bull_count

    if score >= CONFIG["rating_strong_buy"] and bull_count >= 5:
        rating = "STRONG_BUY"
    elif score >= CONFIG["rating_buy"] and bull_count >= 4:
        rating = "BUY"
    elif score >= CONFIG["rating_neutral"] and kama_above:
        rating = "NEUTRAL"
    elif trend == "ROSSO" and score < 20:
        rating = "STRONG_SELL"
    else:
        rating = "SELL"

    return rating, bull_count, sell_count

# ── CLASSIFY SIGNAL ───────────────────────────────────────────────────────────
def classify_signal(closes, kama, sar_bull_arr, ao, er_series,
                    rvi_line, rvi_sig, categoria):
    """
    Calcola il segnale per ogni barra (per la history) e restituisce
    la lista dei segnali giornalieri.
    Segnali: BUY1 / BUY2 / BUY3 / SELL / WATCH / RANGING
    """
    n       = len(closes)
    signals = ['RANGING'] * n

    er_thresh = CONFIG["er_threshold"].get(categoria, CONFIG["er_threshold"]["default"])
    er_trend  = CONFIG["er_trending"].get(categoria, CONFIG["er_trending"]["default"])

    for i in range(CONFIG["kama_n"]+5, n):
        if np.isnan(kama[i]) or np.isnan(er_series[i]):
            continue

        price    = closes[i]
        er_val   = er_series[i]
        sar_bull = bool(sar_bull_arr[i])
        above    = price > kama[i]

        # AO in miglioramento: ultimi 2 valori crescenti
        ao_val   = ao[i] if not np.isnan(ao[i]) else None
        ao_rising = False
        if i >= 2 and not np.isnan(ao[i]) and not np.isnan(ao[i-1]):
            ao_rising = ao[i] > ao[i-1]

        # AO cala 3 barre consecutive
        ao_falling3 = False
        if i >= 3 and not any(np.isnan(ao[i-j]) for j in range(3)):
            ao_falling3 = ao[i] < ao[i-1] < ao[i-2]

        # RVI bull
        rvi_bull = False
        if not np.isnan(rvi_line[i]) and not np.isnan(rvi_sig[i]):
            rvi_bull = rvi_line[i] > rvi_sig[i]

        # KAMA trend (pendenza ultime 3 barre)
        k_rising = False
        if i >= 3 and not any(np.isnan(kama[i-j]) for j in range(3)):
            k_rising = kama[i] > kama[i-1] > kama[i-2]

        # baff a questa barra
        baff_i = 0
        for j in range(i, max(i-20,-1), -1):
            if np.isnan(kama[j]):
                break
            if closes[j] > kama[j]:
                baff_i += 1
            else:
                break

        # score a questa barra
        trend_i = "VERDE" if k_rising else ("ROSSO" if not k_rising and kama[i] < kama[i-1] else "GRIGIO")
        score_i = compute_score(
            er_val, above, sar_bull, ao_val, ao_rising,
            rvi_bull, baff_i,
            (price-kama[i])/kama[i]*100 if kama[i] != 0 else 0,
            0, None, None, trend_i, categoria
        )

        # ── Logica segnali ──────────────────────────────────────────────────
        if er_val < er_thresh:
            signals[i] = 'RANGING'
            continue

        # SELL: SAR bear O AO cala 3 O prezzo < KAMA
        if not above or (not sar_bull and ao_falling3):
            signals[i] = 'SELL'
            continue

        # Condizioni BUY base: prezzo > KAMA + AO migliorante + SAR bull
        buy_base = above and sar_bull and ao_rising

        if not buy_base:
            # WATCH: sopra KAMA ma SAR o AO non confermati
            if above and (sar_bull or ao_rising):
                signals[i] = 'WATCH'
            else:
                signals[i] = 'SELL' if not above else 'RANGING'
            continue

        # BUY1: condizioni base soddisfatte
        if score_i < CONFIG["buy2_score"] or baff_i < 3 or er_val < er_trend:
            signals[i] = 'BUY1'
        # BUY2: score >= 45, baff >= 3, er >= trending
        elif score_i < CONFIG["buy3_score"] or baff_i < 5 or not rvi_bull:
            signals[i] = 'BUY2'
        # BUY3: score >= 60, baff >= 5, rvi bull
        else:
            signals[i] = 'BUY3'

    return signals

# ── HISTORY — ricostruisce da dati storici ────────────────────────────────────
def build_history(ticker, dates, signals_arr, closes):
    """
    Prende la serie completa di segnali giornalieri e restituisce
    solo i cambi di segnale (transizioni).
    """
    history = []
    prev    = None
    for i, (date, sig, price) in enumerate(zip(dates, signals_arr, closes)):
        if sig != prev and sig not in ('RANGING',):
            history.append({
                "date":   date,
                "signal": sig,
                "price":  round(float(price), 4),
            })
            prev = sig
    return history

# ── PROCESS SINGLE ETF ────────────────────────────────────────────────────────
def process_etf(etf_tuple: tuple) -> dict | None:
    ticker, nome, categoria, is_lev, is_short, currency = etf_tuple

    ohlcv = get_ohlcv(ticker)
    if not ohlcv:
        return None

    closes  = np.array(ohlcv["closes"],  dtype=float)
    highs   = np.array(ohlcv["highs"],   dtype=float)
    lows    = np.array(ohlcv["lows"],    dtype=float)
    opens   = np.array(ohlcv["opens"],   dtype=float)
    volumes = np.array(ohlcv["volumes"], dtype=float)
    dates   = ohlcv["dates"]

    try:
        # ── Indicatori ──────────────────────────────────────────────────────
        kama      = calc_kama(closes, CONFIG["kama_n"], CONFIG["kama_fast"], CONFIG["kama_slow"])
        er_series = calc_er(closes, CONFIG["kama_n"])
        sar, sar_bull_arr = calc_sar(highs, lows)
        ao        = calc_ao(highs, lows)
        rvi, rvi_sig = calc_rvi(opens, highs, lows, closes, CONFIG["rvi_period"])
        rsi       = calc_rsi(closes, 14)
        adx       = calc_adx(highs, lows, closes, 14)
        atr       = calc_atr(highs, lows, closes, 14)
        bb_mid, bb_upper, bb_lower, bb_width = calc_bb(closes, CONFIG["bb_period"], CONFIG["bb_std"])
        vol_ratio = calc_vol_ratio(volumes, CONFIG["vol_period"])
        ema20     = calc_ema(closes, 20)
        ema50     = calc_ema(closes, 50)
        sma200    = calc_sma(closes, 200)

        # ── Valori correnti ─────────────────────────────────────────────────
        last_close   = float(closes[-1])
        last_kama    = safe_last(kama)
        last_er      = safe_last(er_series)
        last_sar     = safe_last(sar)
        last_sar_bull= bool(sar_bull_arr[-1])
        last_ao      = safe_last(ao)
        last_rvi     = safe_last(rvi)
        last_rvi_sig = safe_last(rvi_sig)
        last_rsi     = safe_last(rsi)
        last_adx     = safe_last(adx)
        last_atr     = safe_last(atr)
        last_vr      = safe_last(vol_ratio)
        last_ema20   = safe_last(ema20)
        last_ema50   = safe_last(ema50)
        last_sma200  = safe_last(sma200)
        last_bb_w    = safe_last(bb_width)
        last_bb_up   = safe_last(bb_upper)
        last_bb_lo   = safe_last(bb_lower)

        kama_above = last_close > last_kama if last_kama else False
        k_pct      = (last_close - last_kama) / last_kama * 100 if last_kama else 0

        # AO rising (ultime 2 barre)
        ao_valid = ao[~np.isnan(ao)]
        ao_rising = len(ao_valid) >= 2 and ao_valid[-1] > ao_valid[-2]
        ao_falling3 = len(ao_valid) >= 3 and ao_valid[-1] < ao_valid[-2] < ao_valid[-3]

        # RVI bull
        rvi_bull = (last_rvi is not None and last_rvi_sig is not None
                    and last_rvi > last_rvi_sig)

        # KAMA trend
        trend = kama_trend(kama, 5)

        # baff
        baff = calc_baff(closes, kama)

        # momentum
        mom1m = momentum_pct(closes, 21)
        mom3m = momentum_pct(closes, 63)
        mom6m = momentum_pct(closes, 126)

        # variazione giornaliera
        chg_pct = round((closes[-1]/closes[-2]-1)*100, 2) if len(closes) >= 2 else 0

        # ── Score e rating ───────────────────────────────────────────────────
        score = compute_score(
            last_er, kama_above, last_sar_bull, last_ao, ao_rising,
            rvi_bull, baff, k_pct, mom3m, last_adx, last_rsi, trend, categoria
        )

        tv_rating, tv_buy, tv_sell = compute_sys_rating(
            score, kama_above, last_sar_bull, last_ao, rvi_bull, trend, categoria
        )

        # ── Segnali su tutta la serie (per history) ──────────────────────────
        signals_arr = classify_signal(
            closes, kama, sar_bull_arr, ao, er_series, rvi, rvi_sig, categoria
        )

        # Segnale corrente
        current_signal = signals_arr[-1]

        # ── Trova data segnale corrente ──────────────────────────────────────
        # cerca l'ultima transizione verso il segnale attuale
        signal_date = dates[-1]
        signal_bars = 0
        for i in range(len(signals_arr)-1, -1, -1):
            if signals_arr[i] != current_signal:
                signal_date = dates[i+1] if i+1 < len(dates) else dates[-1]
                signal_bars = len(signals_arr) - 1 - i
                break

        # ── History (solo transizioni non-RANGING) ───────────────────────────
        history = build_history(ticker, dates, signals_arr, closes)

        # ── Chart data — file separato per ticker ────────────────────────────
        n90 = min(90, len(dates))
        def to_list(arr, decimals=4):
            return [round(float(v), decimals) if not math.isnan(v) else None
                    for v in arr[-n90:].tolist()]

        chart = {
            "dates":   dates[-n90:],
            "closes":  [round(v,4) for v in closes[-n90:].tolist()],
            "kama":    to_list(kama),
            "ao":      to_list(ao),
            "rsi":     [round(float(v),2) if not math.isnan(v) else None for v in rsi[-n90:].tolist()],
            "signals": signals_arr[-n90:],
        }
        chart_dir = Path("data/charts")
        chart_dir.mkdir(parents=True, exist_ok=True)
        with open(chart_dir / f"{ticker.replace('.','_')}.json", "w") as cf:
            json.dump(chart, cf, separators=(',',':'))

        return {
            # Identificativo
            "ticker":       ticker,
            "nome":         nome,
            "categoria":    categoria,
            "is_leveraged": is_lev,
            "is_short":     is_short,
            "currency":     currency,
            # Prezzi
            "price":        round(last_close, 4),
            "chg_pct":      chg_pct,
            "atr":          round(last_atr, 4) if last_atr else None,
            # Segnale
            "signal":       current_signal,
            "signal_date":  signal_date,
            "signal_bars":  signal_bars,
            # Score e rating
            "score":        score,
            "tv_rating":    tv_rating,
            "tv_buy":       tv_buy,
            "tv_sell":      tv_sell,
            # Indicatori chiave
            "er":           round(last_er, 4)      if last_er    else None,
            "kama":         round(last_kama, 4)    if last_kama  else None,
            "k_pct":        round(k_pct, 2),
            "kama_above":   kama_above,
            "kama_trend":   trend,
            "baff":         baff,
            "sar_bull":     last_sar_bull,
            "ao":           round(last_ao, 4)      if last_ao    else None,
            "ao_rising":    ao_rising,
            "rvi_bull":     rvi_bull,
            "rsi":          round(last_rsi, 1)     if last_rsi   else None,
            "adx":          round(last_adx, 1)     if last_adx   else None,
            "vol_ratio":    round(last_vr, 2)      if last_vr    else None,
            "bb_width":     round(last_bb_w, 4)    if last_bb_w  else None,
            "mom1m":        round(mom1m, 2),
            "mom3m":        round(mom3m, 2),
            "mom6m":        round(mom6m, 2),
            "ema20":        round(last_ema20, 4)   if last_ema20 else None,
            "ema50":        round(last_ema50, 4)   if last_ema50 else None,
            "sma200":       round(last_sma200, 4)  if last_sma200 else None,
            "above_sma200": bool(last_close > last_sma200) if last_sma200 else None,
            # History locale (ultima settimana di transizioni)
            "recent_history": history[-5:] if history else [],
        }

    except Exception as e:
        log.error(f"{ticker}: errore — {e}\n{traceback.format_exc()}")
        return None

# ── HISTORY MANAGER ───────────────────────────────────────────────────────────
def update_history(history_data: dict, results: list) -> dict:
    """
    Aggiorna signals_history.json:
    - Se il ticker non esiste → aggiunge tutta la history ricostruita
    - Se il ticker esiste → aggiunge solo nuove transizioni dall'ultima data
    """
    for r in results:
        ticker  = r["ticker"]
        new_h   = r.get("recent_history", [])
        if not new_h:
            continue

        if ticker not in history_data:
            # Prima volta — ricostruisci da tutto il disponibile
            history_data[ticker] = new_h
        else:
            existing  = history_data[ticker]
            last_date = existing[-1]["date"] if existing else "2000-01-01"
            # Aggiungi solo entry più recenti dell'ultima salvata
            to_add = [h for h in new_h if h["date"] > last_date]
            # Controlla che il segnale sia cambiato
            if to_add:
                last_sig = existing[-1]["signal"] if existing else None
                filtered = [h for h in to_add if h["signal"] != last_sig]
                if filtered:
                    history_data[ticker].extend(filtered)
                    # Tieni solo ultimi 200 eventi per ticker
                    history_data[ticker] = history_data[ticker][-200:]

    return history_data

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    setup_logging()
    Path("data").mkdir(exist_ok=True)
    Path(CONFIG["cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    run_start = datetime.now(CET)
    log.info(f"=== RAPTOR SCANNER v2 START {run_start.strftime('%Y-%m-%d %H:%M CET')} ===")

    universe = load_universe()
    if not universe:
        log.error("Universe vuoto — abort")
        return

    log.info(f"Avvio scansione {len(universe)} ticker con {CONFIG['max_workers']} thread...")

    results  = []
    errors   = []
    skipped  = []

    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = {executor.submit(process_etf, etf): etf[0] for etf in universe}
        done = 0
        for future in as_completed(futures):
            ticker = futures[future]
            done  += 1
            try:
                result = future.result()
                if result:
                    results.append(result)
                    log.info(f"[{done}/{len(universe)}] {ticker} → {result['signal']} score:{result['score']} {result['tv_rating']}")
                else:
                    skipped.append(ticker)
                    log.warning(f"[{done}/{len(universe)}] {ticker} → SKIP")
            except Exception as e:
                errors.append(ticker)
                log.error(f"[{done}/{len(universe)}] {ticker} → ERROR: {e}")

    # ── Carica history esistente ─────────────────────────────────────────────
    history_path = Path(CONFIG["history_file"])
    history_data = {}
    if history_path.exists():
        try:
            with open(history_path) as f:
                history_data = json.load(f)
            log.info(f"History caricata: {len(history_data)} ticker")
        except Exception as e:
            log.warning(f"History non leggibile, si reimposta: {e}")

    # ── Aggiorna history ─────────────────────────────────────────────────────
    history_data = update_history(history_data, results)

    # ── Statistiche segnali ──────────────────────────────────────────────────
    from collections import Counter
    sig_counts = Counter(r["signal"] for r in results)
    rat_counts = Counter(r["tv_rating"] for r in results)

    # ── Scrivi signals.json ──────────────────────────────────────────────────
    output = {
        "meta": {
            "generated_at":    run_start.isoformat(),
            "generated_display": run_start.strftime("%d/%m/%Y %H:%M CET"),
            "total":           len(results),
            "errors":          errors,
            "skipped":         skipped,
            "signal_counts":   dict(sig_counts),
            "rating_counts":   dict(rat_counts),
        },
        "signals": results,
    }

    # Rimuovi chart dalla history (troppo pesante)
    for r in output["signals"]:
        r.pop("recent_history", None)

    with open(CONFIG["signals_file"], "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))
    log.info(f"signals.json scritto: {len(results)} ETF")

    # ── Scrivi signals_history.json ──────────────────────────────────────────
    with open(CONFIG["history_file"], "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, separators=(',', ':'))
    log.info(f"signals_history.json scritto: {len(history_data)} ticker")

    run_end = datetime.now(CET)
    elapsed = (run_end - run_start).total_seconds()
    log.info(f"=== FINE in {elapsed:.0f}s · OK:{len(results)} SKIP:{len(skipped)} ERR:{len(errors)} ===")

if __name__ == "__main__":
    main()
