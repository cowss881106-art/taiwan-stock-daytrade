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
    page_title="台股當沖選股 v10.2",
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
    """
    v10.1：
    1. 突破做多
    2. 回踩做多
    3. 跌破做空
    4. 反彈做空

    使用最近 6 根已完成/最新 5 分K的局部結構，
    不再要求突破整日最高或跌破整日最低。
    """
    if df is None or len(df) < 8:
        return None

    d = df.copy()

    # 只取最新交易日
    try:
        last_date = d.index[-1].date()
        d = d[d.index.date == last_date].copy()
    except Exception:
        pass

    if len(d) < 8:
        return None

    close = d["Close"].astype(float)
    open_ = d["Open"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    volume = d["Volume"].astype(float)

    typical = (high + low + close) / 3
    cumulative_volume = volume.cumsum().replace(0, np.nan)
    vwap_series = (typical * volume).cumsum() / cumulative_volume

    ema5 = close.ewm(span=5, adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()

    price = float(close.iloc[-1])
    last_open = float(open_.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    vwap = float(vwap_series.iloc[-1])
    e5 = float(ema5.iloc[-1])
    e10 = float(ema10.iloc[-1])

    # 最近 6 根之前的局部高低點：排除目前 K 棒
    lookback = min(6, len(d) - 1)
    prior = d.iloc[-(lookback + 1):-1]
    local_high = float(prior["High"].max())
    local_low = float(prior["Low"].min())

    # 量比：目前 5 分K vs 前 12 根平均
    recent_vol = float(volume.iloc[-1])
    base_vol = float(volume.iloc[:-1].tail(12).mean())
    intraday_vr = recent_vol / base_vol if base_vol > 0 else 0

    # 5分 ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr5 = float(tr.rolling(5).mean().iloc[-1])
    if not np.isfinite(atr5) or atr5 <= 0:
        atr5 = max(price * 0.005, 0.01)

    # 前一根狀態，用來辨識回踩/反彈
    prev_price = float(close.iloc[-2])
    prev_vwap = float(vwap_series.iloc[-2])
    prev_e5 = float(ema5.iloc[-2])

    bullish_candle = price > last_open
    bearish_candle = price < last_open

    # 1) 突破做多：局部突破 + VWAP + EMA + 放量
    breakout_long = (
        price > vwap
        and e5 > e10
        and price > local_high
        and intraday_vr >= 1.2
        and bullish_candle
    )

    # 2) 回踩做多：前一根靠近/跌到 VWAP 或 EMA5，本根重新站回並轉強
    pullback_long = (
        price > vwap
        and e5 >= e10
        and bullish_candle
        and (
            prev_price <= prev_vwap * 1.003
            or prev_price <= prev_e5
            or last_low <= vwap * 1.002
        )
        and price > prev_price
        and intraday_vr >= 0.8
    )

    # 3) 跌破做空
    breakdown_short = (
        price < vwap
        and e5 < e10
        and price < local_low
        and intraday_vr >= 1.2
        and bearish_candle
    )

    # 4) 反彈做空：前一根靠近/站上 VWAP 或 EMA5，本根重新跌回並轉弱
    rebound_short = (
        price < vwap
        and e5 <= e10
        and bearish_candle
        and (
            prev_price >= prev_vwap * 0.997
            or prev_price >= prev_e5
            or last_high >= vwap * 0.998
        )
        and price < prev_price
        and intraday_vr >= 0.8
    )

    if breakout_long:
        signal = "🟢 突破做多"
        reason = "站上VWAP＋5分K多頭＋突破近6根高點＋量能確認"
        entry = price
        stop = min(vwap, local_high, price - atr5 * 0.8)
        target1 = price + atr5 * 1.2
        target2 = price + atr5 * 2.0

    elif pullback_long:
        signal = "🟢 回踩做多"
        reason = "VWAP/EMA回踩後重新轉強＋5分K維持多頭"
        entry = price
        stop = min(vwap, last_low, price - atr5 * 0.8)
        target1 = price + atr5 * 1.2
        target2 = price + atr5 * 2.0

    elif breakdown_short:
        signal = "🔴 跌破做空"
        reason = "跌破VWAP＋5分K空頭＋跌破近6根低點＋量能確認"
        entry = price
        stop = max(vwap, local_low, price + atr5 * 0.8)
        target1 = price - atr5 * 1.2
        target2 = price - atr5 * 2.0

    elif rebound_short:
        signal = "🔴 反彈做空"
        reason = "反彈VWAP/EMA失敗後重新轉弱＋5分K維持空頭"
        entry = price
        stop = max(vwap, last_high, price + atr5 * 0.8)
        target1 = price - atr5 * 1.2
        target2 = price - atr5 * 2.0

    else:
        signal = "⚪ 尚未觸發"
        if price > vwap and e5 > e10:
            reason = "盤中偏多，等待局部突破或VWAP/EMA回踩轉強"
        elif price < vwap and e5 < e10:
            reason = "盤中偏空，等待局部跌破或VWAP/EMA反彈轉弱"
        else:
            reason = "VWAP與5分K短均線方向尚未一致"
        entry = np.nan
        stop = np.nan
        target1 = np.nan
        target2 = np.nan

    # 訊號時間使用資料本身最後一根 K 棒時間，不用伺服器現在時間
    try:
        signal_time = d.index[-1].strftime("%Y-%m-%d %H:%M")
    except Exception:
        signal_time = str(d.index[-1])

    return {
        "Yahoo代號": ticker,
        "訊號時間": signal_time,
        "盤中價格": price,
        "VWAP": vwap,
        "EMA5(5分)": e5,
        "EMA10(5分)": e10,
        "近6根高點": local_high,
        "近6根低點": local_low,
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

st.title("📈 台股當沖選股 v10.2")

st.caption(
    "雲端版｜5分K四型態＋VWAP＋訊號時間｜手機可使用"
)

# ------------------------------------------------------------
# 更新按鈕
# ------------------------------------------------------------

if st.button(
    "🔄 更新資料並開始選股",
    type="primary",
    use_container_width=True
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
        # v10 盤中 5 分 K
        # ====================================================

        st.subheader("⚡ v10.2 盤中即時觸發")
        st.caption(
            "先由日線 v9.4 選出高優先候選，再抓 5 分 K。"
            "盤中判斷包含：突破做多、回踩做多、跌破做空、反彈做空；並顯示5分K訊號時間。"
        )

        if st.button(
            "⚡ 更新盤中訊號",
            use_container_width=True
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
                ].copy()

                waiting_df = intraday_df[
                    intraday_df["盤中訊號"] == "⚪ 尚未觸發"
                ].copy()

                trigger_cols = [
                    "股票代號", "股票名稱", "盤中訊號", "訊號時間",
                    "盤中價格", "VWAP", "5分量比",
                    "當沖活躍度", "多方強度", "空方強度",
                    "盤中進場參考", "盤中停損參考",
                    "盤中目標1", "盤中目標2", "盤中理由"
                ]

                if trigger_df.empty:
                    st.info("目前沒有即時觸發訊號。")
                else:
                    st.success(f"🚨 目前即時觸發：{len(trigger_df)} 檔")
                    st.dataframe(
                        trigger_df[trigger_cols],
                        use_container_width=True,
                        hide_index=True
                    )

                    # 手機重點卡片：真正觸發的股票直接展開
                    for _, row in trigger_df.iterrows():
                        with st.expander(
                            f"{row['盤中訊號']}｜{row['股票代號']} {row['股票名稱']}｜"
                            f"{row['盤中價格']:.2f}｜{row['訊號時間']}",
                            expanded=True
                        ):
                            a, b, c = st.columns(3)
                            a.metric("🔥 活躍度", int(row["當沖活躍度"]))
                            b.metric("🟢 多方", int(row["多方強度"]))
                            c.metric("🔴 空方", int(row["空方強度"]))

                            d, e, f = st.columns(3)
                            d.metric("盤中價", f"{row['盤中價格']:.2f}")
                            e.metric("VWAP", f"{row['VWAP']:.2f}")
                            f.metric("5分量比", f"{row['5分量比']:.2f}")

                            p1, p2 = st.columns(2)
                            p1.metric("進場參考", f"{row['盤中進場參考']:.2f}")
                            p2.metric("停損參考", f"{row['盤中停損參考']:.2f}")

                            t1, t2 = st.columns(2)
                            t1.metric("目標1", f"{row['盤中目標1']:.2f}")
                            t2.metric("目標2", f"{row['盤中目標2']:.2f}")

                            st.caption(row["盤中理由"])

                with st.expander(
                    f"👀 尚未觸發候選（{len(waiting_df)} 檔）",
                    expanded=False
                ):
                    if waiting_df.empty:
                        st.write("目前沒有尚未觸發候選。")
                    else:
                        waiting_cols = [
                            "股票代號", "股票名稱", "訊號時間",
                            "當沖活躍度", "多方強度", "空方強度",
                            "盤中價格", "VWAP", "5分量比",
                            "盤中理由"
                        ]
                        st.dataframe(
                            waiting_df[waiting_cols],
                            use_container_width=True,
                            hide_index=True
                        )

                with st.expander("📐 查看盤中技術細節"):
                    detail_cols = [
                        "股票代號", "股票名稱", "訊號時間",
                        "盤中價格", "VWAP",
                        "EMA5(5分)", "EMA10(5分)",
                        "近6根高點", "近6根低點",
                        "5分量比", "盤中訊號"
                    ]
                    st.dataframe(
                        intraday_df[detail_cols],
                        use_container_width=True,
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
                use_container_width=True,
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
                use_container_width=True,
                hide_index=True
            )

        st.subheader("🔴 空方強度 TOP 10")
        if short_candidates.empty:
            st.write("目前沒有符合條件的做空候選。")
        else:
            st.dataframe(
                short_candidates[display_cols],
                use_container_width=True,
                hide_index=True
            )

        with st.expander("🚀 突破做多觀察"):
            if breakout.empty:
                st.write("目前沒有符合條件的股票。")
            else:
                st.dataframe(breakout[display_cols], use_container_width=True, hide_index=True)

        with st.expander("📉 跌破做空觀察"):
            if breakdown.empty:
                st.write("目前沒有符合條件的股票。")
            else:
                st.dataframe(breakdown[display_cols], use_container_width=True, hide_index=True)

        with st.expander("↩️ 回檔做多觀察"):
            if pullback.empty:
                st.write("目前沒有符合條件的股票。")
            else:
                st.dataframe(pullback[display_cols], use_container_width=True, hide_index=True)

        with st.expander("↗️ 反彈做空觀察"):
            if rebound_short.empty:
                st.write("目前沒有符合條件的股票。")
            else:
                st.dataframe(rebound_short[display_cols], use_container_width=True, hide_index=True)

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
            use_container_width=True,
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
                use_container_width=True,
                hide_index=True
            )

else:

    st.info(
        "👆 按上面的「更新資料並開始選股」開始分析。"
    )