import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime

# ============================================================
# Streamlit 設定
# ============================================================

st.set_page_config(
    page_title="台股當沖選股 v14.6",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# 設定
# ============================================================

STOCK_POOL_FILE = "完整台股股票池.xlsx"

BATCH_SIZE = 80
PERIOD = "3mo"

MIN_AMOUNT = 50_000_000
MIN_VOLUME = 3_000
MIN_AVG_AMOUNT = 30_000_000
MIN_AVG_VOLUME = 3_000

TOP_N = 10


# ============================================================
# RSI
# ============================================================

def calc_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# ATR
# ============================================================

def calc_atr(df, period=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(period).mean()

    return atr


# ============================================================
# MACD
# ============================================================

def calc_macd(
    close,
    fast=12,
    slow=26,
    signal=9
):

    ema_fast = close.ewm(
        span=fast,
        adjust=False
    ).mean()

    ema_slow = close.ewm(
        span=slow,
        adjust=False
    ).mean()

    macd = ema_fast - ema_slow

    signal_line = macd.ewm(
        span=signal,
        adjust=False
    ).mean()

    histogram = macd - signal_line

    return macd, signal_line, histogram


# ============================================================
# 讀取股票池
# ============================================================

@st.cache_data
def load_stock_pool():

    df = pd.read_excel(
        STOCK_POOL_FILE
    )

    required = [
        "股票代號",
        "股票名稱",
        "市場",
        "Yahoo代號"
    ]

    for col in required:

        if col not in df.columns:
            raise RuntimeError(
                f"股票池缺少欄位：{col}"
            )

    df = df.dropna(
        subset=["Yahoo代號"]
    ).copy()

    df["Yahoo代號"] = (
        df["Yahoo代號"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["Yahoo代號"] != ""
    ]

    return df


# ============================================================
# 批次下載
# ============================================================

def download_batch(tickers):

    try:

        data = yf.download(
            tickers=tickers,
            period=PERIOD,
            interval="1d",
            auto_adjust=False,
            group_by="column",
            threads=True,
            progress=False
        )

        if data is None or data.empty:
            return None

        return data

    except Exception:

        return None


# ============================================================
# 整批下載
# ============================================================

@st.cache_data(ttl=900)
def download_all_data(tickers):

    all_data = {}

    batches = [
        tickers[i:i + BATCH_SIZE]
        for i in range(
            0,
            len(tickers),
            BATCH_SIZE
        )
    ]

    progress = st.progress(0)

    status = st.empty()

    for i, batch in enumerate(
        batches,
        1
    ):

        status.text(
            f"正在下載第 {i}/{len(batches)} 批，共 {len(batch)} 檔"
        )

        data = download_batch(
            batch
        )

        if data is not None:

            # yfinance 多股票格式
            if isinstance(
                data.columns,
                pd.MultiIndex
            ):

                for ticker in batch:

                    try:

                        if ticker not in data.columns.get_level_values(1):
                            continue

                        df = data.xs(
                            ticker,
                            axis=1,
                            level=1
                        ).copy()

                        if not df.empty:
                            all_data[ticker] = df

                    except Exception:
                        continue

            else:

                if len(batch) == 1:

                    all_data[batch[0]] = data.copy()

        progress.progress(
            i / len(batches)
        )

        time.sleep(0.2)

    status.text(
        f"資料下載完成，共取得 {len(all_data)} 檔"
    )

    return all_data


# ============================================================
# 股票代號
# ============================================================

def get_stock_code(ticker):

    return (
        str(ticker)
        .replace(".TW", "")
        .replace(".TWO", "")
    )


# ============================================================
# 單一股票計算
# ============================================================

def process_stock(
    ticker,
    df,
    stock_info
):

    if df is None or df.empty:
        return None

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in required:

        if col not in df.columns:
            return None

    df = df.copy()

    df = df.dropna(
        subset=required
    )

    if len(df) < 30:
        return None

    close = df["Close"]
    volume = df["Volume"]

    price = float(
        close.iloc[-1]
    )

    previous_close = float(
        close.iloc[-2]
    )

    change_pct = (
        (price / previous_close - 1)
        * 100
    )

    today_volume = float(
        volume.iloc[-1]
    )

    today_lots = (
        today_volume / 1000
    )

    amount = (
        price * today_volume
    )

    avg5_volume = (
        volume.iloc[-5:].mean()
        / 1000
    )

    avg20_volume = (
        volume.iloc[-20:].mean()
        / 1000
    )

    avg5_amount = (
        (
            close.iloc[-5:]
            * volume.iloc[-5:]
        ).mean()
    )

    volume_ratio = (
        today_lots / avg5_volume
        if avg5_volume > 0
        else 0
    )

    volume_strength = (
        avg5_volume / avg20_volume
        if avg20_volume > 0
        else 0
    )

    # --------------------------------------------------------
    # 流動性篩選
    # --------------------------------------------------------

    if amount < MIN_AMOUNT:
        return None

    if today_lots < MIN_VOLUME:
        return None

    if avg5_amount < MIN_AVG_AMOUNT:
        return None

    if avg5_volume < MIN_AVG_VOLUME:
        return None

    # --------------------------------------------------------
    # 均線
    # --------------------------------------------------------

    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi_series = calc_rsi(
        close,
        14
    )

    rsi14 = float(
        rsi_series.iloc[-1]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr_series = calc_atr(
        df,
        14
    )

    atr = float(
        atr_series.iloc[-1]
    )

    atr_pct = (
        atr / price * 100
        if price > 0
        else 0
    )

    # --------------------------------------------------------
    # 當日振幅
    # --------------------------------------------------------

    today_high = float(
        df["High"].iloc[-1]
    )

    today_low = float(
        df["Low"].iloc[-1]
    )

    amplitude_pct = (
        (today_high - today_low)
        / price
        * 100
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd, macd_signal, macd_hist = calc_macd(
        close
    )

    macd_value = float(
        macd.iloc[-1]
    )

    macd_signal_value = float(
        macd_signal.iloc[-1]
    )

    macd_hist_value = float(
        macd_hist.iloc[-1]
    )

    return {

        "股票代號": stock_info["股票代號"],
        "股票名稱": stock_info["股票名稱"],
        "市場": stock_info["市場"],

        "Yahoo代號": ticker,

        "收盤價": price,

        "漲跌幅%": change_pct,

        "成交量(張)": today_lots,

        "5日平均成交量(張)": avg5_volume,

        "成交金額(億元)": amount / 100_000_000,

        "5日平均成交金額(億元)": (
            avg5_amount / 100_000_000
        ),

        "量比": volume_ratio,

        "量能強度": volume_strength,

        "MA5": ma5,
        "MA10": ma10,
        "MA20": ma20,

        "RSI14": rsi14,

        "ATR": atr,

        "ATR%": atr_pct,

        "當日振幅%": amplitude_pct,

        "MACD": macd_value,

        "MACD訊號": macd_signal_value,

        "MACD柱": macd_hist_value
    }


# ============================================================
# 100分制
# ============================================================

def calculate_score(row):

    score = 0

    # --------------------------------------------------------
    # 成交金額
    # --------------------------------------------------------

    amount = row["成交金額(億元)"]

    if amount >= 20:
        score += 20

    elif amount >= 10:
        score += 17

    elif amount >= 5:
        score += 14

    else:
        score += 10

    # --------------------------------------------------------
    # 量比
    # --------------------------------------------------------

    vr = row["量比"]

    if vr >= 3:
        score += 20

    elif vr >= 2:
        score += 17

    elif vr >= 1.5:
        score += 14

    elif vr >= 1:
        score += 10

    else:
        score += 5

    # --------------------------------------------------------
    # 波動
    # --------------------------------------------------------

    atr = row["ATR%"]

    if 2 <= atr <= 7:
        score += 15

    elif atr <= 10:
        score += 11

    elif atr <= 15:
        score += 7

    else:
        score += 3

    # --------------------------------------------------------
    # 趨勢
    # --------------------------------------------------------

    price = row["收盤價"]

    if (
        price > row["MA5"]
        and row["MA5"] > row["MA10"]
        and row["MA10"] > row["MA20"]
    ):

        score += 20

    elif price > row["MA20"]:

        score += 13

    else:

        score += 6

    # --------------------------------------------------------
    # 動能
    # --------------------------------------------------------

    if row["MACD柱"] > 0:
        score += 10

    else:
        score += 5

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = row["RSI14"]

    if 50 <= rsi <= 70:

        score += 15

    elif 45 <= rsi < 50:

        score += 10

    elif 70 < rsi <= 75:

        score += 8

    elif 75 < rsi <= 80:

        score += 4

    elif rsi > 80:

        score += 0

    else:

        score += 5

    return min(
        score,
        100
    )



# ============================================================
# v9.4 三分數制
# ============================================================

def calculate_activity_score(row):
    """只衡量是否適合列入當沖觀察，不代表做多或做空。"""
    score = 0

    amount = row["成交金額(億元)"]
    lots = row["成交量(張)"]
    vr = row["量比"]
    atr = row["ATR%"]

    # 成交金額 25
    if amount >= 50:
        score += 25
    elif amount >= 20:
        score += 22
    elif amount >= 10:
        score += 18
    elif amount >= 5:
        score += 14
    else:
        score += 10

    # 成交量 20
    if lots >= 50000:
        score += 20
    elif lots >= 20000:
        score += 17
    elif lots >= 10000:
        score += 14
    elif lots >= 5000:
        score += 11
    else:
        score += 8

    # 量比 25
    if vr >= 3:
        score += 25
    elif vr >= 2:
        score += 22
    elif vr >= 1.5:
        score += 18
    elif vr >= 1:
        score += 13
    else:
        score += 7

    # ATR / 可交易波動 20
    if 2 <= atr <= 6:
        score += 20
    elif 1.2 <= atr < 2 or 6 < atr <= 8:
        score += 16
    elif 0.8 <= atr < 1.2 or 8 < atr <= 10:
        score += 11
    else:
        score += 6

    # 近期量能穩定度 10
    volume_strength = row["量能強度"]
    if 0.8 <= volume_strength <= 2.5:
        score += 10
    elif 0.5 <= volume_strength < 0.8 or 2.5 < volume_strength <= 4:
        score += 7
    else:
        score += 4

    return min(score, 100)


def calculate_long_score(row):
    """多方結構強度；不是立即買進訊號。"""
    score = 0
    price = row["收盤價"]
    ma5, ma10, ma20 = row["MA5"], row["MA10"], row["MA20"]
    macd = row["MACD柱"]
    rsi = row["RSI14"]
    vr = row["量比"]
    change = row["漲跌幅%"]

    # 趨勢 35
    if price > ma5 > ma10 > ma20:
        score += 35
    elif price > ma10 > ma20:
        score += 28
    elif price > ma20:
        score += 20
    elif price > ma5:
        score += 12
    else:
        score += 4

    # MACD 20
    if macd > 0:
        score += 20
    else:
        score += 4

    # RSI 20
    if 52 <= rsi <= 68:
        score += 20
    elif 45 <= rsi < 52 or 68 < rsi <= 75:
        score += 14
    elif 40 <= rsi < 45 or 75 < rsi <= 80:
        score += 8
    else:
        score += 3

    # 量價確認 15
    if vr >= 2 and 0 < change <= 7:
        score += 15
    elif vr >= 1.2 and change > 0:
        score += 11
    elif change > -2:
        score += 7
    else:
        score += 2

    # 避免追高 / 大跌中硬判多 10
    if 0 <= change <= 5:
        score += 10
    elif -2 < change < 0 or 5 < change <= 7:
        score += 7
    elif -4 < change <= -2:
        score += 3

    return min(score, 100)


def calculate_short_score(row):
    """空方結構強度；不是立即放空訊號。"""
    score = 0
    price = row["收盤價"]
    ma5, ma10, ma20 = row["MA5"], row["MA10"], row["MA20"]
    macd = row["MACD柱"]
    rsi = row["RSI14"]
    vr = row["量比"]
    change = row["漲跌幅%"]

    # 趨勢 35
    if price < ma5 < ma10 < ma20:
        score += 35
    elif price < ma10 < ma20:
        score += 28
    elif price < ma20:
        score += 20
    elif price < ma5:
        score += 12
    else:
        score += 4

    # MACD 20
    if macd < 0:
        score += 20
    else:
        score += 4

    # RSI 20
    if 32 <= rsi <= 48:
        score += 20
    elif 25 <= rsi < 32 or 48 < rsi <= 55:
        score += 14
    elif 20 <= rsi < 25 or 55 < rsi <= 60:
        score += 8
    else:
        score += 3

    # 量價確認 15
    if vr >= 2 and -7 <= change < 0:
        score += 15
    elif vr >= 1.2 and change < 0:
        score += 11
    elif change < 2:
        score += 7
    else:
        score += 2

    # 避免追空 / 大漲中硬判空 10
    if -5 <= change <= 0:
        score += 10
    elif -7 <= change < -5 or 0 < change < 2:
        score += 7
    elif 2 <= change < 4:
        score += 3

    return min(score, 100)


def score_direction(row):
    """用多空分數差判斷觀察方向；差距不足則中性。"""
    long_score = row["多方強度"]
    short_score = row["空方強度"]

    if long_score >= 65 and long_score - short_score >= 12:
        return "做多"
    if short_score >= 65 and short_score - long_score >= 12:
        return "做空"
    return "中性"

# ============================================================
# 當沖訊號
# ============================================================

def judgement(row):

    rsi = row["RSI14"]
    change = row["漲跌幅%"]
    volume_ratio = row["量比"]
    atr = row["ATR%"]
    price = row["收盤價"]
    ma5 = row["MA5"]
    ma10 = row["MA10"]
    ma20 = row["MA20"]
    macd_hist = row["MACD柱"]

    # 極端波動優先警示
    if atr >= 10:
        return ("中性", "極端波動", "ATR 過高，先控制部位")

    # 多方：突破
    if (
        price > ma5
        and ma5 > ma10
        and ma10 > ma20
        and volume_ratio >= 2
        and macd_hist > 0
        and 50 <= rsi < 75
        and 0 < change <= 7
    ):
        return ("做多", "突破做多觀察",
                "均線多頭＋量能放大＋MACD偏多＋當日上漲確認")

    # 空方：跌破
    if (
        price < ma5
        and ma5 < ma10
        and ma10 < ma20
        and volume_ratio >= 1.5
        and macd_hist < 0
        and 25 < rsi <= 55
        and -7 <= change < 0
    ):
        return ("做空", "跌破做空觀察",
                "均線空頭＋量能放大＋MACD偏空＋當日下跌確認")

    # 多方：回檔
    if (
        price >= ma20
        and price <= ma5 * 1.02
        and ma5 >= ma10
        and macd_hist >= 0
        and 45 <= rsi <= 70
        and change > -3
    ):
        return ("做多", "回檔做多觀察",
                "多頭架構中的短線回檔，等待止穩")

    # 空方：反彈
    if (
        price <= ma20
        and price >= ma5 * 0.98
        and ma5 <= ma10
        and macd_hist <= 0
        and 30 <= rsi <= 55
        and change < 3
    ):
        return ("做空", "反彈做空觀察",
                "空頭架構中的短線反彈，等待轉弱")

    # 多方趨勢
    if price > ma20 and macd_hist > 0 and rsi >= 50:
        return ("做多", "偏多觀察", "股價站上月線且動能偏多")

    # 空方趨勢
    if price < ma20 and macd_hist < 0 and rsi <= 50:
        return ("做空", "偏空觀察", "股價跌破月線且動能偏空")

    if rsi >= 80:
        return ("中性", "過熱不追", "RSI ≥ 80，避免追多")

    if rsi <= 20:
        return ("中性", "超跌不追空", "RSI ≤ 20，避免追空")

    if volume_ratio < 0.8:
        return ("中性", "量能不足", "今日成交量低於近期平均")

    if atr >= 7:
        return ("中性", "高波動控制部位", "波動較大，需控制交易部位")

    if volume_ratio >= 1.5:
        return ("中性", "等待確認", "量能增加但方向尚未完全確認")

    return ("中性", "觀望", "目前訊號不足")


# ============================================================
# 交易計畫
# ============================================================

def trade_levels(row):

    price = row["收盤價"]
    atr = row["ATR"]
    direction = row["交易方向"]
    signal = row["當沖訊號"]

    if direction == "做空":
        entry = price
        stop = price + atr * 0.8
        target1 = price - atr * 1.2
        target2 = price - atr * 2.0
        pullback = price + atr * 0.3

    elif direction == "做多":
        entry = price
        stop = price - atr * 0.8
        target1 = price + atr * 1.2
        target2 = price + atr * 2.0
        pullback = price - atr * 0.3

    else:
        entry = price
        stop = price - atr
        target1 = price + atr
        target2 = price + atr * 1.5
        pullback = price

    risk = abs(entry - stop)
    reward1 = abs(target1 - entry)
    reward2 = abs(target2 - entry)

    rr1 = reward1 / risk if risk > 0 else 0
    rr2 = reward2 / risk if risk > 0 else 0

    return pd.Series({
        "進場參考價": entry,
        "回檔/反彈參考價": pullback,
        "停損參考價": stop,
        "第一目標價": target1,
        "第二目標價": target2,
        "風險報酬比1": rr1,
        "風險報酬比2": rr2
    })


# ============================================================
# 主分析
# ============================================================

def run_analysis():

    stock_pool = load_stock_pool()

    tickers = (
        stock_pool["Yahoo代號"]
        .tolist()
    )

    data = download_all_data(
        tuple(tickers)
    )

    results = []

    info_map = (
        stock_pool
        .set_index("Yahoo代號")
        .to_dict("index")
    )

    for ticker, df in data.items():

        if ticker not in info_map:
            continue

        result = process_stock(
            ticker,
            df,
            info_map[ticker]
        )

        if result is not None:

            results.append(
                result
            )

    if not results:

        return pd.DataFrame()

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # 分數
    # --------------------------------------------------------

    result_df["當沖活躍度"] = result_df.apply(
        calculate_activity_score, axis=1
    )

    result_df["多方強度"] = result_df.apply(
        calculate_long_score, axis=1
    )

    result_df["空方強度"] = result_df.apply(
        calculate_short_score, axis=1
    )

    # --------------------------------------------------------
    # 訊號
    # --------------------------------------------------------

    signals = result_df.apply(
        judgement,
        axis=1
    )

    result_df["交易方向"] = (
        signals.apply(
            lambda x: x[0]
        )
    )

    result_df["當沖訊號"] = (
        signals.apply(
            lambda x: x[1]
        )
    )

    result_df["訊號理由"] = (
        signals.apply(
            lambda x: x[2]
        )
    )

    # v9.4：方向由多空雙分數決定，訊號名稱仍保留作型態參考
    result_df["交易方向"] = result_df.apply(
        score_direction,
        axis=1
    )

    # --------------------------------------------------------
    # 交易計畫
    # --------------------------------------------------------

    levels = result_df.apply(
        trade_levels,
        axis=1
    )

    result_df = pd.concat(
        [
            result_df,
            levels
        ],
        axis=1
    )

    # --------------------------------------------------------
    # 適合度
    # --------------------------------------------------------

    def suitability(score):

        if score >= 90:
            return "★★★★★ 極佳"

        elif score >= 80:
            return "★★★★☆ 很佳"

        elif score >= 70:
            return "★★★☆☆ 良好"

        elif score >= 60:
            return "★★☆☆☆ 普通"

        else:
            return "★☆☆☆☆ 不建議"

    result_df["當沖適合度"] = (
        result_df["當沖活躍度"]
        .apply(suitability)
    )

    # --------------------------------------------------------
    # 風險
    # --------------------------------------------------------

    def risk(row):

        if row["ATR%"] >= 10:
            return "極高風險"

        elif row["ATR%"] >= 7:
            return "高風險"

        elif row["ATR%"] >= 4:
            return "中風險"

        else:
            return "低風險"

    result_df["當沖風險"] = (
        result_df.apply(
            risk,
            axis=1
        )
    )

    # --------------------------------------------------------
    # 操作建議
    # --------------------------------------------------------

    def advice(signal):

        mapping = {
            "突破做多觀察": "等待突破確認後再考慮做多",
            "回檔做多觀察": "等待回檔止穩後再考慮做多",
            "偏多觀察": "偏多觀察",
            "跌破做空觀察": "等待跌破確認後再考慮做空",
            "反彈做空觀察": "等待反彈轉弱後再考慮做空",
            "偏空觀察": "偏空觀察",
            "量能不足": "暫不進場",
            "過熱不追": "避免追多",
            "超跌不追空": "避免追空",
            "極端波動": "降低部位",
            "高波動控制部位": "控制交易部位",
            "等待確認": "等待方向確認",
            "觀望": "觀察"
        }

        return mapping.get(signal, "觀察")

    result_df["操作建議"] = (
        result_df["當沖訊號"]
        .apply(advice)
    )

    # --------------------------------------------------------
    # 排序
    # --------------------------------------------------------

    result_df = result_df.sort_values(
        [
            "當沖活躍度",
            "量比"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    return result_df




# ============================================================
# v14 正式 v13.6 ORB 做多 + 做空研究觀察
# ============================================================

V14_COST_PCT = 0.20
V14_BENCHMARK = "0050.TW"

def _v14_flatten(df, ticker=None):
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        try:
            if ticker is not None and ticker in x.columns.get_level_values(-1):
                x = x.xs(ticker, axis=1, level=-1).copy()
            else:
                x.columns = x.columns.get_level_values(0)
        except Exception:
            x.columns = x.columns.get_level_values(0)
    return x

@st.cache_data(ttl=900)
def v14_download_daily(ticker, period="1y"):
    try:
        return _v14_flatten(yf.download(
            ticker, period=period, interval="1d", auto_adjust=False,
            progress=False, threads=False
        ), ticker)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def v14_download_daily_batch(tickers, period="3mo"):
    """批次下載個股日線，供 D-1 收盤使用。"""
    tickers = list(tickers)
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers=tickers, period=period, interval="1d",
            auto_adjust=False, group_by="column",
            threads=True, progress=False
        )
    except Exception:
        return {}
    if raw is None or raw.empty:
        return {}
    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        last = raw.columns.get_level_values(-1)
        for t in tickers:
            try:
                if t in last:
                    x = raw.xs(t, axis=1, level=-1).copy()
                    if not x.empty:
                        out[t] = x
            except Exception:
                pass
    elif len(tickers) == 1:
        out[tickers[0]] = raw.copy()
    return out


@st.cache_data(ttl=60)
def v14_download_5m_batch(tickers):
    """批次下載完整股票池 5分K；回傳 {ticker: DataFrame}。"""
    tickers = list(tickers)
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers=tickers,
            period="5d",
            interval="5m",
            auto_adjust=False,
            group_by="column",
            threads=True,
            progress=False
        )
    except Exception:
        return {}

    if raw is None or raw.empty:
        return {}

    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        level_last = raw.columns.get_level_values(-1)
        for t in tickers:
            try:
                if t not in level_last:
                    continue
                x = raw.xs(t, axis=1, level=-1).copy()
                if not x.empty:
                    out[t] = x
            except Exception:
                continue
    elif len(tickers) == 1:
        out[tickers[0]] = raw.copy()
    return out


@st.cache_data(ttl=60)
def v14_download_5m(ticker):
    try:
        return _v14_flatten(yf.download(
            ticker, period="5d", interval="5m", auto_adjust=False,
            progress=False, threads=False
        ), ticker)
    except Exception:
        return pd.DataFrame()

def v14_market_regime(trade_date):
    """Regime 嚴格使用交易日 D 之前的最後完成 0050 日線。"""
    m = v14_download_daily(V14_BENCHMARK, "1y")
    req = {"Close"}
    if m.empty or not req.issubset(m.columns):
        return {"ok":False, "lowvol":False, "reason":"0050 日線不足"}
    z = pd.DataFrame(index=m.index)
    z["close"] = pd.to_numeric(m["Close"], errors="coerce")
    z["ret"] = z["close"].pct_change()
    z["vol20"] = z["ret"].rolling(20, min_periods=20).std()
    z["thr60"] = z["vol20"].shift(1).rolling(60, min_periods=60).median()
    z = z.dropna(subset=["vol20","thr60"])
    if z.empty:
        return {"ok":False, "lowvol":False, "reason":"0050 LowVol 歷史不足"}
    dates = pd.Index([pd.Timestamp(i).date() for i in z.index])
    completed = z[dates < trade_date]
    if completed.empty:
        return {"ok":False, "lowvol":False, "reason":"找不到 D-1 完成日線"}
    r = completed.iloc[-1]
    return {
        "ok": True,
        "lowvol": bool(r["vol20"] < r["thr60"]),
        "date": str(pd.Timestamp(completed.index[-1]).date()),
        "vol20": float(r["vol20"]),
        "thr60": float(r["thr60"]),
    }

def v14_prev_daily_close_from_df(d, trade_date):
    if d is None or d.empty or "Close" not in d.columns:
        return None, None
    dates = np.array([pd.Timestamp(i).date() for i in d.index], dtype=object)
    idx = np.where(dates < trade_date)[0]
    if len(idx) == 0:
        return None, None
    i = idx[-1]
    v = pd.to_numeric(pd.Series([d.iloc[i]["Close"]]), errors="coerce").iloc[0]
    if not np.isfinite(v):
        return None, None
    return float(v), str(dates[i])


def v14_prev_daily_close(ticker, trade_date):
    return v14_prev_daily_close_from_df(
        v14_download_daily(ticker, "3mo"), trade_date
    )

def v14_prepare_intraday(df, trade_date=None):
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    required = ["Open","High","Low","Close","Volume"]
    if not all(c in d.columns for c in required):
        return pd.DataFrame()
    for c in required:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    try:
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC").tz_convert("Asia/Taipei")
        else:
            d.index = d.index.tz_convert("Asia/Taipei")
    except Exception:
        pass
    d = d.dropna(subset=required)
    if d.empty:
        return d
    target_date = trade_date if trade_date is not None else d.index[-1].date()
    return d[d.index.date == target_date].between_time("09:00","13:20").copy()

def v14_orb_snapshot(ticker, research_short=False, trade_date=None, raw_5m=None, raw_daily=None):
    raw = raw_5m if raw_5m is not None else v14_download_5m(ticker)
    d = v14_prepare_intraday(raw, trade_date=trade_date)
    if len(d) < 6:
        return None
    day = d.index[-1].date()
    if raw_daily is not None:
        prev_close, prev_date = v14_prev_daily_close_from_df(raw_daily, day)
    else:
        prev_close, prev_date = v14_prev_daily_close(ticker, day)
    if prev_close is None:
        return None

    orb = d.between_time("09:00","09:25")
    if len(orb) < 6:
        latest_t = d.index[-1].time()
        before_0930 = latest_t < pd.Timestamp("09:30").time()
        status_text = (
            "等待09:30 ORB完成"
            if before_0930
            else f"ORB資料不完整，本日排除（{len(orb)}/6根）"
        )
        return {
            "Yahoo代號":ticker, "日期":str(day), "狀態":status_text,
            "最新5分K":str(d.index[-1]), "前日收盤":prev_close,
            "ORB完整":False, "ORB根數":len(orb)
        }
    orb = orb.iloc[:6]
    oh = float(orb["High"].max()); ol = float(orb["Low"].min())
    op = float(orb.iloc[0]["Open"])
    gap = (op / prev_close - 1) * 100
    width = (oh / ol - 1) * 100 if ol > 0 else np.nan

    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    v = d["Volume"].astype(float)
    d["VWAP"] = (tp * v).cumsum() / v.cumsum().replace(0, np.nan)
    d["VR"] = v / v.shift(1).rolling(12).mean()

    direction = "做空研究" if research_short else "正式做多"
    base_ok = (gap <= -1 if research_short else gap >= 1) and np.isfinite(width) and width < 3
    signal_i = None
    for i in range(6, len(d)):
        ts = d.index[i]
        if ts.time() >= pd.Timestamp("12:00").time():
            break
        b = d.iloc[i]
        vr = b["VR"]
        if not np.isfinite(vr) or vr < 1.2:
            continue
        if research_short:
            hit = b["Close"] < ol and b["Close"] < b["VWAP"] and b["Close"] < b["Open"]
        else:
            hit = b["Close"] > oh and b["Close"] > b["VWAP"] and b["Close"] > b["Open"]
        if base_ok and hit:
            signal_i = i
            break

    nowbar = d.iloc[-1]
    out = {
        "Yahoo代號": ticker, "日期":str(day), "方向":direction,
        "ORB完整":True, "ORB根數":6,
        "最新5分K":str(d.index[-1]), "前日資料日":prev_date, "前日收盤":prev_close,
        "Gap%":gap, "ORB High":oh, "ORB Low":ol, "ORB寬度%":width,
        "最新價":float(nowbar["Close"]), "VWAP":float(nowbar["VWAP"]) if np.isfinite(nowbar["VWAP"]) else np.nan,
        "最新VR":float(nowbar["VR"]) if np.isfinite(nowbar["VR"]) else np.nan,
    }

    latest_time = d.index[-1].time()
    if latest_time >= pd.Timestamp("13:00").time():
        out["狀態"] = "今日策略已結束（13:00後不持倉）"
        return out
    if latest_time >= pd.Timestamp("12:00").time() and signal_i is None:
        out["狀態"] = "12:00後禁止新進場"
        return out
    if not base_ok:
        out["狀態"] = "未符合 Gap / ORB 基本條件"
        return out
    if signal_i is None:
        out["狀態"] = "等待 ORB / VWAP / VR 訊號"
        return out

    out["訊號時間"] = str(d.index[signal_i])
    next_i = signal_i + 1
    if next_i >= len(d):
        out["狀態"] = "訊號成立，等待下一根5分K開盤"
        return out
    if d.index[next_i].time() >= pd.Timestamp("12:00").time():
        out["狀態"] = "訊號過晚，不進場"
        return out

    entry = float(d.iloc[next_i]["Open"])
    stop = oh if research_short else ol
    R = (stop-entry) if research_short else (entry-stop)
    if not np.isfinite(R) or R <= 0:
        out["狀態"] = "R無效，不進場"
        return out
    target = entry - 2*R if research_short else entry + 2*R
    out.update({
        "狀態":"研究訊號成立" if research_short else "正式做多訊號成立",
        "進場時間":str(d.index[next_i]), "進場價":entry,
        "停損價":stop, "2R目標":target, "R":R
    })
    return out

def v14_latest_trade_date():
    """以 0050 最新可用 5 分 K 決定本次分析交易日 D。"""
    b = v14_prepare_intraday(v14_download_5m(V14_BENCHMARK))
    if b.empty:
        return None
    return b.index[-1].date()

def v14_build_daily_cache(stock_pool, trade_date):
    """v14.5：新交易日只建立一次 D-1 日線快取。"""
    info = stock_pool[["Yahoo代號","股票代號","股票名稱","市場"]].drop_duplicates("Yahoo代號").copy()
    tickers = info["Yahoo代號"].astype(str).tolist()
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    daily_map = {}
    p = st.progress(0)
    s = st.empty()
    for bi, batch in enumerate(batches, start=1):
        s.text(f"建立 D-1 日線快取：第 {bi}/{len(batches)} 批")
        got = v14_download_daily_batch(tuple(batch), "3mo")
        daily_map.update(got)
        p.progress(bi / len(batches))

    st.session_state["v14_daily_map"] = daily_map
    st.session_state["v14_daily_trade_date"] = str(trade_date)
    s.text(f"D-1 日線快取完成：{len(daily_map)}/{len(tickers)} 檔")
    return daily_map


def v14_scan(result_df):
    """v14.6：第一次全市場建池；同交易日後續只刷新固定候選池。"""
    stock_pool = load_stock_pool()
    if stock_pool is None or stock_pool.empty:
        return pd.DataFrame(), pd.DataFrame(), {"ok":False, "lowvol":False, "reason":"股票池為空"}

    trade_date = v14_latest_trade_date()
    if trade_date is None:
        return pd.DataFrame(), pd.DataFrame(), {"ok":False, "lowvol":False, "reason":"0050 5分K不足"}

    regime = v14_market_regime(trade_date)
    regime["trade_date"] = str(trade_date)

    info = stock_pool[["Yahoo代號","股票代號","股票名稱","市場"]].drop_duplicates("Yahoo代號").copy()
    all_tickers = info["Yahoo代號"].astype(str).tolist()
    info_map = info.set_index("Yahoo代號").to_dict("index")

    cached_date = st.session_state.get("v14_daily_trade_date")
    daily_map = st.session_state.get("v14_daily_map")
    if cached_date != str(trade_date) or not isinstance(daily_map, dict) or not daily_map:
        daily_map = v14_build_daily_cache(stock_pool, trade_date)

    pool_date = st.session_state.get("v14_pool_trade_date")
    long_pool = st.session_state.get("v14_long_pool")
    short_pool = st.session_state.get("v14_short_pool")
    pool_ready = pool_date == str(trade_date) and isinstance(long_pool, list) and isinstance(short_pool, list)

    if pool_ready:
        scan_tickers = list(dict.fromkeys(long_pool + short_pool))
        scan_mode = "候選池快速刷新"
    else:
        scan_tickers = all_tickers
        scan_mode = "首次全市場建池"

    total = len(scan_tickers)
    if total == 0:
        regime.update({"pool_ready":True, "long_pool_n":0, "short_pool_n":0, "union_pool_n":0, "scan_n":0, "scan_mode":"候選池為空"})
        return pd.DataFrame(), pd.DataFrame(), regime

    batches = [scan_tickers[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    progress = st.progress(0)
    status = st.empty()
    processed = missing_5m = missing_daily = 0
    long_rows, short_rows = [], []
    new_long_pool, new_short_pool = [], []

    for bi, batch in enumerate(batches, start=1):
        status.text(f"v14.6 {scan_mode}：第 {bi}/{len(batches)} 批（{processed}/{total}）")
        batch_5m = v14_download_5m_batch(tuple(batch))

        for t in batch:
            raw5 = batch_5m.get(t)
            rawd = daily_map.get(t)
            if raw5 is None or raw5.empty:
                missing_5m += 1; processed += 1; continue
            if rawd is None or rawd.empty:
                missing_daily += 1; processed += 1; continue

            meta = info_map.get(t, {})

            if regime.get("ok") and regime.get("lowvol"):
                x = v14_orb_snapshot(t, False, trade_date, raw5, rawd)
                if x:
                    x.update(meta)
                    if (x.get("ORB完整") is True and pd.notna(x.get("Gap%")) and
                        pd.notna(x.get("ORB寬度%")) and float(x["Gap%"]) >= 1.0 and
                        float(x["ORB寬度%"]) < 3.0):
                        new_long_pool.append(t)
                    if (not pool_ready) or (t in long_pool):
                        long_rows.append(x)

            y = v14_orb_snapshot(t, True, trade_date, raw5, rawd)
            if y:
                y.update(meta)
                if (y.get("ORB完整") is True and pd.notna(y.get("Gap%")) and
                    pd.notna(y.get("ORB寬度%")) and float(y["Gap%"]) <= -1.0 and
                    float(y["ORB寬度%"]) < 3.0):
                    new_short_pool.append(t)
                if (not pool_ready) or (t in short_pool):
                    short_rows.append(y)

            processed += 1
        progress.progress(min(processed / total, 1.0))

    if not pool_ready:
        st.session_state["v14_long_pool"] = list(dict.fromkeys(new_long_pool))
        st.session_state["v14_short_pool"] = list(dict.fromkeys(new_short_pool))
        st.session_state["v14_pool_trade_date"] = str(trade_date)
        long_pool = st.session_state["v14_long_pool"]
        short_pool = st.session_state["v14_short_pool"]

    union_n = len(set((long_pool or []) + (short_pool or [])))
    regime.update({
        "missing_5m":missing_5m, "missing_daily":missing_daily,
        "processed":processed, "total":total, "daily_cache":True,
        "pool_ready":True, "long_pool_n":len(long_pool or []),
        "short_pool_n":len(short_pool or []), "union_pool_n":union_n,
        "scan_n":total, "scan_mode":scan_mode
    })
    status.text(
        f"v14.6 {scan_mode}完成：掃描 {processed}/{total}｜"
        f"正式做多池 {len(long_pool or [])}｜做空研究池 {len(short_pool or [])}｜"
        f"聯集 {union_n}｜5分K缺資料 {missing_5m}"
    )
    return pd.DataFrame(long_rows), pd.DataFrame(short_rows), regime

# ============================================================
# v10.0 盤中 5 分 K 訊號
# ============================================================

@st.cache_data(ttl=60)
def download_intraday_data(tickers):
    """只抓候選股的當日/近期 5 分 K，避免對全市場抓盤中資料。"""
    intraday = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                period="5d",
                interval="5m",
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if df is None or df.empty:
                continue

            # 單一 ticker 有時仍可能回傳 MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    if ticker in df.columns.get_level_values(-1):
                        df = df.xs(ticker, axis=1, level=-1).copy()
                    else:
                        df.columns = df.columns.get_level_values(0)
                except Exception:
                    df.columns = df.columns.get_level_values(0)

            required = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required):
                continue

            df = df.dropna(subset=required).copy()
            if len(df) >= 6:
                intraday[ticker] = df

        except Exception:
            continue

    return intraday


def analyze_intraday(ticker, df):
    """以最新 5 分 K 判斷盤中結構。不是保證成交或獲利的交易指令。"""
    if df is None or len(df) < 6:
        return None

    d = df.copy()

    # 只使用最後一個交易日
    try:
        last_date = d.index[-1].date()
        d = d[d.index.date == last_date].copy()
    except Exception:
        pass

    if len(d) < 3:
        return None

    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    volume = d["Volume"].astype(float)

    typical = (high + low + close) / 3
    cumulative_volume = volume.cumsum().replace(0, np.nan)
    vwap_series = (typical * volume).cumsum() / cumulative_volume

    ema5 = close.ewm(span=5, adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()

    price = float(close.iloc[-1])
    vwap = float(vwap_series.iloc[-1])
    e5 = float(ema5.iloc[-1])
    e10 = float(ema10.iloc[-1])

    # 不把當前 K 棒算進「先前高低點」，避免自己突破自己
    prior = d.iloc[:-1]
    prior_high = float(prior["High"].max())
    prior_low = float(prior["Low"].min())

    recent_vol = float(volume.iloc[-1])
    base_vol = float(volume.iloc[:-1].tail(12).mean()) if len(volume) > 1 else 0
    intraday_vr = recent_vol / base_vol if base_vol > 0 else 0

    # 5分K ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr5 = float(tr.rolling(5).mean().iloc[-1]) if len(tr) >= 5 else float(tr.mean())
    if not np.isfinite(atr5) or atr5 <= 0:
        atr5 = max(price * 0.005, 0.01)

    long_trigger = (
        price > vwap
        and e5 > e10
        and price > prior_high
        and intraday_vr >= 1.2
    )

    short_trigger = (
        price < vwap
        and e5 < e10
        and price < prior_low
        and intraday_vr >= 1.2
    )

    if long_trigger:
        signal = "🟢 做多觸發"
        reason = "站上 VWAP＋5分K短均線偏多＋突破盤中前高＋量能確認"
        entry = price
        stop = min(vwap, price - atr5 * 0.8)
        target1 = price + atr5 * 1.2
        target2 = price + atr5 * 2.0

    elif short_trigger:
        signal = "🔴 做空觸發"
        reason = "跌破 VWAP＋5分K短均線偏空＋跌破盤中前低＋量能確認"
        entry = price
        stop = max(vwap, price + atr5 * 0.8)
        target1 = price - atr5 * 1.2
        target2 = price - atr5 * 2.0

    else:
        signal = "⚪ 尚未觸發"
        if price >= vwap and e5 >= e10:
            reason = "盤中偏多，但尚未同時突破前高並取得量能確認"
        elif price <= vwap and e5 <= e10:
            reason = "盤中偏空，但尚未同時跌破前低並取得量能確認"
        else:
            reason = "VWAP、短均線與突破/跌破條件尚未形成一致方向"
        entry = np.nan
        stop = np.nan
        target1 = np.nan
        target2 = np.nan

    return {
        "Yahoo代號": ticker,
        "盤中價格": price,
        "VWAP": vwap,
        "EMA5(5分)": e5,
        "EMA10(5分)": e10,
        "盤中前高": prior_high,
        "盤中前低": prior_low,
        "5分量比": intraday_vr,
        "盤中訊號": signal,
        "盤中理由": reason,
        "盤中進場參考": entry,
        "盤中停損參考": stop,
        "盤中目標1": target1,
        "盤中目標2": target2
    }


def build_intraday_candidates(result_df, n_each=10):
    """從日線選股結果取高活躍、多方與空方候選的聯集。"""
    if result_df is None or result_df.empty:
        return []

    active = result_df.nlargest(n_each, "當沖活躍度")
    longs = result_df.nlargest(n_each, "多方強度")
    shorts = result_df.nlargest(n_each, "空方強度")

    tickers = pd.concat([
        active["Yahoo代號"],
        longs["Yahoo代號"],
        shorts["Yahoo代號"]
    ]).drop_duplicates().tolist()

    return tickers


def run_intraday_analysis(result_df):
    tickers = build_intraday_candidates(result_df, n_each=10)
    data = download_intraday_data(tuple(tickers))

    rows = []
    for ticker in tickers:
        if ticker not in data:
            continue
        item = analyze_intraday(ticker, data[ticker])
        if item is not None:
            rows.append(item)

    if not rows:
        return pd.DataFrame()

    intraday_df = pd.DataFrame(rows)

    info = result_df[
        ["Yahoo代號", "股票代號", "股票名稱",
         "當沖活躍度", "多方強度", "空方強度"]
    ].drop_duplicates("Yahoo代號")

    intraday_df = intraday_df.merge(info, on="Yahoo代號", how="left")

    order = {
        "🟢 做多觸發": 0,
        "🔴 做空觸發": 0,
        "⚪ 尚未觸發": 1
    }
    intraday_df["_order"] = intraday_df["盤中訊號"].map(order).fillna(2)

    return intraday_df.sort_values(
        ["_order", "當沖活躍度"],
        ascending=[True, False]
    ).drop(columns="_order").reset_index(drop=True)

# ============================================================
# Streamlit UI
# ============================================================

st.title("📈 台股當沖選股 v14.6")

st.caption(
    "雲端版｜正式 v13.6 ORB 做多＋做空研究觀察｜手機可使用"
)

# ------------------------------------------------------------
# 更新按鈕
# ------------------------------------------------------------

if st.button(
    "🔄 更新資料並開始選股",
    type="primary",
    width="stretch"
):

    with st.spinner(
        "正在下載台股資料並分析，請稍候..."
    ):

        try:

            result_df = run_analysis()

            st.session_state[
                "result_df"
            ] = result_df

            st.session_state[
                "update_time"
            ] = datetime.now()

        except Exception as e:

            st.error(
                f"程式發生錯誤：{e}"
            )

# ------------------------------------------------------------
# 顯示結果
# ------------------------------------------------------------

if "result_df" in st.session_state:

    result_df = st.session_state[
        "result_df"
    ]

    update_time = st.session_state.get(
        "update_time"
    )

    if update_time:

        st.info(
            "最後更新："
            + update_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    if result_df.empty:

        st.warning(
            "目前沒有符合條件的股票。"
        )

    else:

        st.success(
            f"符合條件：{len(result_df)} 檔"
        )

        # ====================================================
        # v14 固定策略 / 研究觀察
        # ====================================================
        st.subheader("🎯 v14 ORB 策略")
        st.caption(
            "正式做多＝鎖定 v13.6 規則並掃描完整股票池；做空＝v13.7 研究觀察，不屬於正式策略。"
            " Yahoo 盤中資料可能延遲，請以「最新5分K」時間為準。"
        )

        c_v14a, c_v14b = st.columns([3, 1])
        with c_v14a:
            run_v14 = st.button("⚡ 更新 v14 ORB 訊號", width="stretch")
        with c_v14b:
            reset_v14 = st.button("♻️ 重建 D-1 快取", width="stretch")

        if reset_v14:
            st.session_state.pop("v14_daily_map", None)
            st.session_state.pop("v14_daily_trade_date", None)
            st.session_state.pop("v14_long_pool", None)
            st.session_state.pop("v14_short_pool", None)
            st.session_state.pop("v14_pool_trade_date", None)
            v14_download_daily_batch.clear()
            st.success("D-1 與當日 ORB 候選池快取已清除；下次更新會重新建立。")

        if run_v14:
            with st.spinner("第一次建立 D-1＋全市場 ORB 候選池；之後只刷新候選池 5 分 K..."):
                long_orb, short_orb, regime = v14_scan(result_df)
                st.session_state["v14_long_orb"] = long_orb
                st.session_state["v14_short_orb"] = short_orb
                st.session_state["v14_regime"] = regime
                st.session_state["v14_time"] = datetime.now()

        if "v14_regime" in st.session_state:
            regime = st.session_state["v14_regime"]
            if not regime.get("ok"):
                st.warning("0050 Regime 無法判定：" + regime.get("reason","資料不足"))
            elif regime.get("lowvol"):
                st.success(
                    f"🟢 正式做多策略 ON｜交易日 D={regime.get('trade_date')}｜0050 D-1={regime.get('date')}｜"
                    f"vol20={regime.get('vol20'):.4f} < threshold={regime.get('thr60'):.4f}"
                )
            else:
                st.warning(
                    f"⚪ 正式做多策略 OFF｜交易日 D={regime.get('trade_date')}｜0050 D-1={regime.get('date')}｜"
                    f"vol20={regime.get('vol20'):.4f} ≥ threshold={regime.get('thr60'):.4f}"
                )

            if st.session_state.get("v14_time"):
                st.caption("v14 更新：" + st.session_state["v14_time"].strftime("%Y-%m-%d %H:%M:%S"))

            long_orb = st.session_state.get("v14_long_orb", pd.DataFrame())
            short_orb = st.session_state.get("v14_short_orb", pd.DataFrame())

            st.markdown("#### 🟢 正式策略：v13.6 ORB 做多")
            if long_orb.empty:
                st.info("目前沒有正式做多候選，或今日 LowVol Regime 為 OFF。")
            else:
                long_show = [c for c in [
                    "股票代號","股票名稱","市場","Yahoo代號","狀態","最新5分K","ORB完整","ORB根數","Gap%","ORB寬度%","最新VR",
                    "ORB High","ORB Low","訊號時間","進場時間","進場價","停損價","2R目標"
                ] if c in long_orb.columns]
                st.dataframe(long_orb[long_show], width="stretch", hide_index=True)

            st.markdown("#### 🔴 做空研究觀察")
            st.caption("v13.7 歷史結果高度集中於單一交易日，因此這裡只做研究觀察，不視為正式交易策略。")
            if short_orb.empty:
                st.info("目前沒有做空研究觀察資料。")
            else:
                short_show = [c for c in [
                    "股票代號","股票名稱","市場","Yahoo代號","狀態","最新5分K","ORB完整","ORB根數","Gap%","ORB寬度%","最新VR",
                    "ORB High","ORB Low","訊號時間","進場時間","進場價","停損價","2R目標"
                ] if c in short_orb.columns]
                st.dataframe(short_orb[short_show], width="stretch", hide_index=True)

        with st.expander("🧪 舊 v10 盤中訊號（研究參考）"):
            st.caption("以下舊 EMA/VWAP/前高前低訊號保留作研究參考，不屬於 v13.6 正式策略。")

            # ====================================================
            # v10 盤中 5 分 K
            # ====================================================

            st.subheader("⚡ v10 盤中 5分K 訊號")
            st.caption(
                "先由日線 v9.4 選出高優先候選，再抓 5 分 K。"
                "盤中觸發需同時參考 VWAP、5分K短均線、前高/前低與量能。"
            )

            if st.button(
                "⚡ 更新盤中訊號",
                width="stretch"
            ):
                with st.spinner("正在抓取候選股 5 分 K 並計算 VWAP..."):
                    intraday_df = run_intraday_analysis(result_df)
                    st.session_state["intraday_df"] = intraday_df
                    st.session_state["intraday_time"] = datetime.now()

            if "intraday_df" in st.session_state:
                intraday_df = st.session_state["intraday_df"]
                intraday_time = st.session_state.get("intraday_time")

                if intraday_time:
                    st.info(
                        "盤中訊號更新："
                        + intraday_time.strftime("%Y-%m-%d %H:%M:%S")
                    )

                if intraday_df.empty:
                    st.warning("目前沒有取得可用的 5 分 K 資料。")
                else:
                    trigger_df = intraday_df[
                        intraday_df["盤中訊號"] != "⚪ 尚未觸發"
                    ]

                    if trigger_df.empty:
                        st.info("目前候選股尚未出現完整的多空觸發條件。")
                    else:
                        st.success(f"目前盤中觸發：{len(trigger_df)} 檔")

                    intraday_cols = [
                        "股票代號", "股票名稱",
                        "當沖活躍度", "多方強度", "空方強度",
                        "盤中價格", "VWAP", "5分量比",
                        "盤中訊號", "盤中理由",
                        "盤中進場參考", "盤中停損參考",
                        "盤中目標1", "盤中目標2"
                    ]

                    st.dataframe(
                        intraday_df[intraday_cols],
                        width="stretch",
                        hide_index=True
                    )

                    with st.expander("📐 查看盤中技術細節"):
                        detail_cols = [
                            "股票代號", "股票名稱",
                            "盤中價格", "VWAP",
                            "EMA5(5分)", "EMA10(5分)",
                            "盤中前高", "盤中前低",
                            "5分量比", "盤中訊號"
                        ]
                        st.dataframe(
                            intraday_df[detail_cols],
                            width="stretch",
                            hide_index=True
                        )

        # ====================================================
        # TOP 10
        # ====================================================

        st.subheader(
            "🏆 台股當沖 TOP 10"
        )

        top10 = result_df.head(
            TOP_N
        ).copy()

        display_cols = [

            "股票代號",
            "股票名稱",
            "收盤價",
            "交易方向",
            "漲跌幅%",
            "成交量(張)",
            "5日平均成交量(張)",
            "成交金額(億元)",
            "量比",
            "RSI14",
            "ATR%",
            "當沖活躍度",
            "多方強度",
            "空方強度",
            "當沖訊號",
            "當沖適合度",
            "當沖風險"
        ]

        # 手機版 TOP 10 卡片
        st.caption("📱 手機版重點資訊｜點開卡片查看交易計畫")

        for rank, (_, row) in enumerate(top10.iterrows(), start=1):
            code = str(row["股票代號"])
            name = str(row["股票名稱"])
            change = float(row["漲跌幅%"] or 0)
            arrow = "▲" if change > 0 else ("▼" if change < 0 else "－")

            with st.expander(
                f"#{rank}  {code} {name}｜{row['收盤價']:.2f}｜{arrow} {change:+.2f}%｜🔥{int(row['當沖活躍度'])} 🟢{int(row['多方強度'])} 🔴{int(row['空方強度'])}",
                expanded=(rank <= 3)
            ):
                s1, s2, s3 = st.columns(3)
                s1.metric("🔥 當沖活躍度", int(row["當沖活躍度"]))
                s2.metric("🟢 多方強度", int(row["多方強度"]))
                s3.metric("🔴 空方強度", int(row["空方強度"]))

                c1, c2, c3 = st.columns(3)
                c1.metric("量比", f"{row['量比']:.2f}")
                c2.metric("RSI14", f"{row['RSI14']:.1f}")
                c3.metric("ATR%", f"{row['ATR%']:.2f}%")

                st.markdown(f"**🎯 訊號：{row['當沖訊號']}**")
                st.caption(str(row["訊號理由"]))
                st.write(f"適合度：{row['當沖適合度']}　｜　風險：{row['當沖風險']}")

                p1, p2 = st.columns(2)
                p1.metric("進場參考", f"{row['進場參考價']:.2f}")
                p2.metric("回檔/反彈", f"{row['回檔/反彈參考價']:.2f}")

                p3, p4, p5 = st.columns(3)
                p3.metric("停損", f"{row['停損參考價']:.2f}")
                p4.metric("目標 1", f"{row['第一目標價']:.2f}")
                p5.metric("目標 2", f"{row['第二目標價']:.2f}")

                st.write(
                    f"風報比：1:{row['風險報酬比1']:.2f} / 1:{row['風險報酬比2']:.2f}　｜　"
                    f"操作：{row['操作建議']}"
                )

        with st.expander("📋 TOP 10 完整表格"):
            st.dataframe(
                top10[display_cols],
                width="stretch",
                hide_index=True
            )

        # ====================================================
        # 多空雙向候選
        # ====================================================

        long_candidates = (
            result_df[result_df["當沖活躍度"] >= 60]
            .sort_values(["多方強度", "當沖活躍度"], ascending=False)
            .head(10)
        )

        short_candidates = (
            result_df[result_df["當沖活躍度"] >= 60]
            .sort_values(["空方強度", "當沖活躍度"], ascending=False)
            .head(10)
        )

        breakout = result_df[
            result_df["當沖訊號"] == "突破做多觀察"
        ]

        breakdown = result_df[
            result_df["當沖訊號"] == "跌破做空觀察"
        ]

        pullback = result_df[
            result_df["當沖訊號"] == "回檔做多觀察"
        ]

        rebound_short = result_df[
            result_df["當沖訊號"] == "反彈做空觀察"
        ]

        st.caption("🔥 活躍度＝值不值得盯；🟢/🔴 強度＝多空結構完整度，不等於立即進場訊號。")

        st.subheader("🟢 多方強度 TOP 10")
        if long_candidates.empty:
            st.write("目前沒有符合條件的做多候選。")
        else:
            st.dataframe(
                long_candidates[display_cols],
                width="stretch",
                hide_index=True
            )

        st.subheader("🔴 空方強度 TOP 10")
        if short_candidates.empty:
            st.write("目前沒有符合條件的做空候選。")
        else:
            st.dataframe(
                short_candidates[display_cols],
                width="stretch",
                hide_index=True
            )

        with st.expander("🚀 突破做多觀察"):
            if breakout.empty:
                st.write("目前沒有符合條件的股票。")
            else:
                st.dataframe(breakout[display_cols], width="stretch", hide_index=True)

        with st.expander("📉 跌破做空觀察"):
            if breakdown.empty:
                st.write("目前沒有符合條件的股票。")
            else:
                st.dataframe(breakdown[display_cols], width="stretch", hide_index=True)

        with st.expander("↩️ 回檔做多觀察"):
            if pullback.empty:
                st.write("目前沒有符合條件的股票。")
            else:
                st.dataframe(pullback[display_cols], width="stretch", hide_index=True)

        with st.expander("↗️ 反彈做空觀察"):
            if rebound_short.empty:
                st.write("目前沒有符合條件的股票。")
            else:
                st.dataframe(rebound_short[display_cols], width="stretch", hide_index=True)

        # ====================================================
        # 詳細交易計畫
        # ====================================================

        st.subheader(
            "🎯 TOP 10 交易計畫"
        )

        plan_cols = [

            "股票代號",
            "股票名稱",
            "收盤價",
            "交易方向",
            "當沖訊號",
            "訊號理由",
            "進場參考價",
            "回檔/反彈參考價",
            "停損參考價",
            "第一目標價",
            "第二目標價",
            "風險報酬比1",
            "風險報酬比2",
            "操作建議"
        ]

        st.dataframe(
            top10[
                plan_cols
            ],
            width="stretch",
            hide_index=True
        )

        # ====================================================
        # 完整資料
        # ====================================================

        with st.expander(
            "📊 查看全部符合條件股票"
        ):

            st.dataframe(
                result_df,
                width="stretch",
                hide_index=True
            )

else:

    st.info(
        "👆 按上面的「更新資料並開始選股」開始分析。"
    )