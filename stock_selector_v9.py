# -*- coding: utf-8 -*-

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from datetime import datetime
import pickle
import warnings
import numpy as np
import pandas as pd

from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule

warnings.filterwarnings("ignore")

CACHE_DIR = "cache_stock_data"
OUTPUT_FILE = "台股當沖選股_TOP10_v9.xlsx"

MIN_AMOUNT = 50_000_000
MIN_VOLUME = 3_000
MIN_AVG_AMOUNT = 30_000_000
MIN_AVG_VOLUME = 3_000

TOP_N = 10


def get_cache_latest_date():

    latest_date = None

    if not os.path.exists(CACHE_DIR):
        return None

    files = [
        f
        for f in os.listdir(CACHE_DIR)
        if f.endswith(".pkl")
    ]

    for filename in files:

        path = os.path.join(
            CACHE_DIR,
            filename
        )

        try:

            with open(path, "rb") as f:
                data = pickle.load(f)

            if data is None or data.empty:
                continue

            current_date = data.index.max()

            if latest_date is None or current_date > latest_date:
                latest_date = current_date

        except Exception:
            continue

    return latest_date


def cache_needs_update():

    latest_date = get_cache_latest_date()

    if latest_date is None:
        return True

    today = pd.Timestamp(
        datetime.now().date()
    )

    latest_date = pd.Timestamp(
        latest_date
    ).normalize()

    return latest_date < today
def ensure_cache_is_updated():

    latest_date = get_cache_latest_date()

    print()
    print("=" * 70)
    print("檢查台股 Cache")
    print("=" * 70)

    if latest_date is None:

        print("目前沒有可用 Cache")
        print("開始下載最新資料...")

    elif not cache_needs_update():

        print(
            f"Cache 最新日期：{latest_date}"
        )
        print("✓ Cache 已是最新資料")
        print("✓ 不需要重新下載")

        return

    else:

        print(
            f"Cache 最新日期：{latest_date}"
        )
        print("⚠️ Cache 不是今天資料")
        print("開始更新最新台股資料...")

    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "update_stock_cache.py"
        ],
        check=False
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Cache 更新失敗，停止執行選股程式"
        )

    new_date = get_cache_latest_date()

    if new_date is None:

        raise RuntimeError(
            "Cache 更新後找不到有效資料"
        )

    print(
        f"✓ Cache 更新完成：{new_date}"
    )

# ============================================================
# ============================================================
# 技術指標
# ============================================================








# ============================================================
# 載入 batch cache
# ============================================================
# ============================================================
# 載入 batch cache
# ============================================================

def load_all_cache():

    all_data = {}

    files = sorted(
        [
            f for f in os.listdir(CACHE_DIR)
            if f.endswith(".pkl")
        ],
        key=lambda x: int(
            x.replace("batch_", "").replace(".pkl", "")
        )
        if x.replace("batch_", "").replace(".pkl", "").isdigit()
        else 9999
    )

    print(f"找到快取批次：{len(files)} 個")

    for filename in files:

        path = os.path.join(CACHE_DIR, filename)

        try:

            with open(path, "rb") as f:
                data = pickle.load(f)

            print(
                f"  {filename}："
                f"{len(data) if hasattr(data, '__len__') else '未知'} 筆"
            )

            # ==================================================
            # 情況 1：dict
            # ==================================================

            if isinstance(data, dict):

                for code, df in data.items():

                    if isinstance(df, pd.DataFrame):
                        all_data[str(code)] = df

            # ==================================================
            # 情況 2：DataFrame
            # ==================================================

            elif isinstance(data, pd.DataFrame):

                df = data

                # ----------------------------------------------
                # MultiIndex：yfinance 批次下載格式
                # ----------------------------------------------

                if isinstance(df.columns, pd.MultiIndex):

                    level0 = list(
                        df.columns.get_level_values(0)
                    )

                    level1 = list(
                        df.columns.get_level_values(1)
                    )

                    price_fields = {
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Adj Close",
                        "Volume"
                    }

                    # 找出 OHLCV 所在的 level
                    if len(
                        price_fields.intersection(level0)
                    ) >= 3:

                        price_level = 0
                        ticker_level = 1

                    elif len(
                        price_fields.intersection(level1)
                    ) >= 3:

                        price_level = 1
                        ticker_level = 0

                    else:

                        print(
                            f"  無法辨識 {filename} 的 MultiIndex 格式"
                        )

                        continue

                    tickers = list(
                        dict.fromkeys(
                            df.columns.get_level_values(
                                ticker_level
                            )
                        )
                    )

                    count = 0

                    for ticker in tickers:

                        try:

                            stock_df = df.xs(
                                ticker,
                                axis=1,
                                level=ticker_level
                            )

                            # 某些格式 xs 後仍可能是 MultiIndex
                            if isinstance(
                                stock_df.columns,
                                pd.MultiIndex
                            ):

                                stock_df.columns = (
                                    stock_df.columns
                                    .get_level_values(price_level)
                                )

                            stock_df.columns = [
                                str(c)
                                for c in stock_df.columns
                            ]

                            # 只保留真正需要的欄位
                            required = [
                                "Open",
                                "High",
                                "Low",
                                "Close",
                                "Volume"
                            ]

                            if all(
                                c in stock_df.columns
                                for c in required
                            ):

                                all_data[str(ticker)] = stock_df
                                count += 1

                        except Exception:
                            continue

                    print(
                        f"    已拆分股票：{count} 檔"
                    )

                # ----------------------------------------------
                # 普通 DataFrame
                # ----------------------------------------------

                else:

                    required = [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume"
                    ]

                    if all(
                        c in df.columns
                        for c in required
                    ):

                        all_data[filename] = df

        except Exception as e:

            print(
                f"  讀取失敗 {filename}: {e}"
            )

    print()
    print(
        f"快取股票資料總數：{len(all_data)}"
    )
    print()

    return all_data


# ============================================================
# 找股票代號
# ============================================================

def get_stock_code(key):

    text = str(key)

    # Yahoo 格式
    if text.endswith(".TW"):
        return text[:-3]

    if text.endswith(".TWO"):
        return text[:-4]

    return text


# ============================================================
# 單檔計算
# ============================================================
# ============================================================
# 技術指標計算
# ============================================================

def calc_rsi(close, period=14):
    """
    RSI14
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))

    rsi = 100 - (100 / (1 + rs))

    # 如果沒有下跌資料，RSI 視為 100
    rsi = rsi.where(avg_loss != 0, 100)

    return rsi


def calc_atr(df, period=14):
    """
    ATR14
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(period).mean()

    return atr


def calc_macd(
    close,
    fast_period=12,
    slow_period=26,
    signal_period=9
):
    """
    MACD
    """
    ema_fast = close.ewm(
        span=fast_period,
        adjust=False
    ).mean()

    ema_slow = close.ewm(
        span=slow_period,
        adjust=False
    ).mean()

    macd = ema_fast - ema_slow

    signal = macd.ewm(
        span=signal_period,
        adjust=False
    ).mean()

    hist = macd - signal

    return macd, signal, hist
def process_stock(stock_code, df):

    try:

        if not isinstance(df, pd.DataFrame):
            return None

        if len(df) < 30:
            return None

        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(c in df.columns for c in required):
            return None

        df = df.dropna(subset=required)

        if len(df) < 30:
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        last = df.iloc[-1]

        price = float(last["Close"])

        prev_close = float(df["Close"].iloc[-2])

        change = (
            (price / prev_close - 1) * 100
            if prev_close != 0
            else 0
        )

        today_volume = float(last["Volume"])

        today_lots = today_volume / 1000

        amount = price * today_volume

        avg5_volume = volume.tail(5).mean() / 1000
        avg20_volume = volume.tail(20).mean() / 1000

        avg5_amount = (
            df["Close"] * df["Volume"]
        ).tail(5).mean()

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

        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]

        rsi = calc_rsi(close).iloc[-1]

        atr = calc_atr(df).iloc[-1]

        atr_pct = (
            atr / price * 100
            if price > 0
            else 0
        )

        macd, macd_signal, macd_hist = calc_macd(close)

        macd_value = macd.iloc[-1]
        signal_value = macd_signal.iloc[-1]
        hist_value = macd_hist.iloc[-1]

        daily_amplitude = (
            (float(last["High"]) - float(last["Low"]))
            / price * 100
            if price > 0
            else 0
        )

        amp_series = (
            (df["High"] - df["Low"])
            / df["Close"] * 100
        )

        avg_amplitude = amp_series.tail(5).mean()

        result = {

            "股票代號": get_stock_code(stock_code),

            "收盤價": round(price, 2),

            "漲跌幅": round(change, 2),

            "成交量": int(today_lots),

            "5日平均成交量": int(avg5_volume),

            "20日平均成交量": int(avg20_volume),

            "成交金額": amount,

            "5日平均成交金額": avg5_amount,

            "量比": round(volume_ratio, 2),

            "5日/20日量能": round(volume_strength, 2),

            "RSI14": round(float(rsi), 2),

            "ATR%": round(float(atr_pct), 2),

            "當日振幅": round(float(daily_amplitude), 2),

            "5日平均振幅": round(float(avg_amplitude), 2),

            "MA5": round(float(ma5), 2),

            "MA10": round(float(ma10), 2),

            "MA20": round(float(ma20), 2),

            "MACD": round(float(macd_value), 4),

            "MACD訊號": round(float(signal_value), 4),

            "MACD柱": round(float(hist_value), 4)
        }

        # ====================================================
        # 流動性篩選
        # ====================================================

        if amount < MIN_AMOUNT:
            return None

        if today_lots < MIN_VOLUME:
            return None

        if avg5_amount < MIN_AVG_AMOUNT:
            return None

        if avg5_volume < MIN_AVG_VOLUME:
            return None

        return result

    except Exception:

        return None


# ============================================================
# v6 專業評分
# ============================================================

def calculate_score(row):

    score = 0

    amount = row["成交金額"]
    avg_amount = row["5日平均成交金額"]

    volume = row["成交量"]
    avg_volume = row["5日平均成交量"]

    ratio = row["量比"]
    strength = row["5日/20日量能"]

    amplitude = row["當日振幅"]
    atr = row["ATR%"]
    avg_amp = row["5日平均振幅"]

    change = row["漲跌幅"]
    rsi = row["RSI14"]

    # 流動性

    if amount >= 10_000_000_000:
        score += 8
    elif amount >= 5_000_000_000:
        score += 7
    elif amount >= 2_000_000_000:
        score += 5
    elif amount >= 1_000_000_000:
        score += 3
    else:
        score += 1

    if avg_amount >= 5_000_000_000:
        score += 6
    elif avg_amount >= 2_000_000_000:
        score += 5
    elif avg_amount >= 1_000_000_000:
        score += 3
    else:
        score += 1

    if volume >= 50_000:
        score += 3
    elif volume >= 20_000:
        score += 2
    else:
        score += 1

    if avg_volume >= 20_000:
        score += 3
    elif avg_volume >= 10_000:
        score += 2
    else:
        score += 1

    # 量能

    if ratio >= 3:
        score += 12
    elif ratio >= 2:
        score += 10
    elif ratio >= 1.5:
        score += 8
    elif ratio >= 1.2:
        score += 6
    elif ratio >= 1:
        score += 4

    if strength >= 1.5:
        score += 8
    elif strength >= 1.3:
        score += 7
    elif strength >= 1.15:
        score += 5
    elif strength >= 1:
        score += 3
    else:
        score += 1

    # 波動

    if 4 <= amplitude <= 10:
        score += 8
    elif 3 <= amplitude < 4 or 10 < amplitude <= 15:
        score += 6
    elif 2 <= amplitude < 3:
        score += 4
    elif amplitude > 15:
        score += 3
    else:
        score += 1

    if 3 <= atr <= 8:
        score += 7
    elif 2 <= atr < 3 or 8 < atr <= 12:
        score += 5
    elif atr > 12:
        score += 3
    else:
        score += 1

    if 3 <= avg_amp <= 8:
        score += 5
    elif 2 <= avg_amp < 3 or 8 < avg_amp <= 12:
        score += 4
    elif avg_amp > 12:
        score += 2
    else:
        score += 1

    # 趨勢

    close = row["收盤價"]
    ma5 = row["MA5"]
    ma10 = row["MA10"]
    ma20 = row["MA20"]

    if close > ma5 > ma10 > ma20:
        score += 15
    elif close > ma5 and ma5 > ma10:
        score += 12
    elif close > ma20:
        score += 8
    elif close > ma10:
        score += 5

    # 動能

    if 2 <= change <= 7:
        score += 7
    elif 1 <= change < 2:
        score += 5
    elif 7 < change <= 10:
        score += 5
    elif 0 <= change < 1:
        score += 3
    elif -2 <= change < 0:
        score += 1

    if row["MACD"] > row["MACD訊號"] and row["MACD柱"] > 0:
        score += 8
    elif row["MACD"] > row["MACD訊號"]:
        score += 5
    elif row["MACD柱"] > 0:
        score += 3

    # RSI

    if 50 <= rsi <= 65:
        score += 10
    elif 65 < rsi <= 70:
        score += 8
    elif 45 <= rsi < 50:
        score += 7
    elif 40 <= rsi < 45:
        score += 5
    elif 70 < rsi <= 75:
        score += 5
    elif 75 < rsi <= 80:
        score += 2
    elif rsi > 80:
        score += 0
    else:
        score += 3

    # RSI penalty

    if rsi > 80:
        score -= 10
    elif rsi > 75:
        score -= 5
    elif rsi > 70:
        score -= 2

    return max(0, min(100, score))


# ============================================================
# 判斷
# ============================================================

def judgement(row):

    score = row["當沖強度"]
    rsi = row["RSI14"]
    ratio = row["量比"]
    change = row["漲跌幅"]
    amplitude = row["當日振幅"]
    atr = row["ATR%"]

    if rsi >= 80:
        return "過熱不追", "RSI 過熱，避免追價"

    if change <= -3 and amplitude >= 8:
        return "偏空高風險", "跌幅較大且波動偏高"

    if amplitude >= 15 or atr >= 12:
        return "極端波動", "波動極大，控制部位"

    if (
        score >= 85
        and ratio >= 2
        and 50 <= rsi <= 70
        and change > 0
    ):
        return "突破買進觀察", "量能強、趨勢偏多，等待突破"

    if (
        score >= 80
        and change > 0
        and ratio >= 1.5
        and rsi < 75
    ):
        return "偏多觀察", "趨勢與量能偏多"

    if (
        score >= 70
        and ratio >= 1.2
        and 45 <= rsi <= 65
        and change <= 2
    ):
        return "回檔買進觀察", "量能仍在，等待回檔企穩"

    if ratio >= 2 and score >= 70:
        return "量能異常", "成交量明顯放大，等待價格確認"

    if amplitude >= 10:
        return "高波動控制部位", "振幅偏高，降低持倉風險"

    if ratio < 1:
        return "量能不足", "量比低於 1"

    if score >= 70:
        return "等待確認", "條件尚可，等待價格確認"

    return "觀望", "當沖條件不足"


# ============================================================
# 交易價格
# ============================================================

def trade_levels(row):

    close = float(row["收盤價"])
    atr_pct = float(row["ATR%"])

    atr_value = close * atr_pct / 100

    signal = str(row["當沖訊號"])

    if signal == "突破買進觀察":

        entry = close + atr_value * 0.20
        stop = entry - atr_value * 0.80
        target1 = entry + atr_value * 1.20
        target2 = entry + atr_value * 2.00

    elif signal == "回檔買進觀察":

        entry = close - atr_value * 0.30
        stop = entry - atr_value * 0.80
        target1 = entry + atr_value * 1.20
        target2 = entry + atr_value * 2.00

    elif signal == "多方觀察":

        entry = close
        stop = entry - atr_value * 0.80
        target1 = entry + atr_value * 1.20
        target2 = entry + atr_value * 2.00

    elif signal == "高波動控制部位":

        entry = close
        stop = entry - atr_value * 0.70
        target1 = entry + atr_value * 1.00
        target2 = entry + atr_value * 1.60

    elif signal == "偏空高風險":

        entry = close
        stop = close + atr_value * 0.80
        target1 = close - atr_value * 1.00
        target2 = close - atr_value * 1.60

    else:

        entry = close
        stop = close - atr_value * 0.80
        target1 = entry + atr_value * 1.00
        target2 = entry + atr_value * 1.50

    risk = abs(entry - stop)

    if risk <= 0:
        risk = close * 0.005

    reward1 = abs(target1 - entry)
    reward2 = abs(target2 - entry)

    rr1 = reward1 / risk
    rr2 = reward2 / risk

    return (
        round(entry, 2),
        round(stop, 2),
        round(target1, 2),
        round(target2, 2),
        round(rr1, 2),
        round(rr2, 2)
    )
# ============================================================
# Excel
# ============================================================

def format_sheet(ws):

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    fill = PatternFill(
        "solid",
        fgColor="1F4E78"
    )

    font = Font(
        color="FFFFFF",
        bold=True
    )

    for cell in ws[1]:

        cell.fill = fill
        cell.font = font

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center"
        )

    for column in ws.columns:

        max_len = 0

        for cell in column:

            try:
                max_len = max(
                    max_len,
                    len(str(cell.value))
                )
            except:
                pass

        letter = column[0].column_letter

        ws.column_dimensions[letter].width = min(
            max(max_len + 2, 10),
            30
        )

    headers = {
        cell.value: cell.column
        for cell in ws[1]
    }

    for name in [
        "收盤價",
        "進場參考價",
        "停損參考價",
        "第一目標價",
        "第二目標價"
    ]:

        if name in headers:

            col = headers[name]

            for r in range(2, ws.max_row + 1):
                ws.cell(
                    r,
                    col
                ).number_format = "0.00"

    if "當沖強度" in headers:

        col = headers["當沖強度"]

        letter = ws.cell(
            1,
            col
        ).column_letter

        if ws.max_row < 2:
            return

        ws.conditional_formatting.add(
            f"{letter}2:{letter}{ws.max_row}",
            ColorScaleRule(
                start_type="num",
                start_value=50,
                mid_type="num",
                mid_value=70,
                end_type="num",
                end_value=90
            )
        )


# ============================================================
# 主程式
# ============================================================
# ============================================================
# v8.0 交易計畫計算
# ============================================================

def calculate_trade_plan(row):

    close = float(row["收盤價"])
    atr_pct = float(row["ATR%"])

    signal = str(row["當沖訊號"])

    # ATR 轉成價格
    atr = close * atr_pct / 100

    # --------------------------------------------------------
    # 突破買進觀察
    # --------------------------------------------------------

    if signal == "突破買進觀察":

        entry = close
        breakout = close + atr * 0.20
        stop = close - atr * 0.80

        target1 = close + atr * 1.20
        target2 = close + atr * 2.00

        advice = "突破確認後進場，不追高"

    # --------------------------------------------------------
    # 回檔買進觀察
    # --------------------------------------------------------

    elif signal == "回檔買進觀察":

        entry = close - atr * 0.30
        breakout = close
        stop = entry - atr * 0.80

        target1 = entry + atr * 1.20
        target2 = entry + atr * 2.00

        advice = "等待拉回支撐區再進場"

    # --------------------------------------------------------
    # 多方觀察
    # --------------------------------------------------------

    elif signal == "多方觀察":

        entry = close
        breakout = close + atr * 0.20
        stop = close - atr

        target1 = close + atr * 1.20
        target2 = close + atr * 2.00

        advice = "偏多觀察，等待量價確認"

    # --------------------------------------------------------
    # 高波動
    # --------------------------------------------------------

    elif signal == "高波動控制部位":

        entry = close
        breakout = close + atr * 0.20
        stop = close - atr * 0.70

        target1 = close + atr * 1.00
        target2 = close + atr * 1.60

        advice = "波動過大，降低部位"

    # --------------------------------------------------------
    # 偏空
    # --------------------------------------------------------

    elif signal == "偏空高風險":

        entry = close
        breakout = close
        stop = close + atr * 0.80

        target1 = close - atr * 1.00
        target2 = close - atr * 1.60

        advice = "偏空高風險，不建議追多"

    # --------------------------------------------------------
    # 其他
    # --------------------------------------------------------

    else:

        entry = close
        breakout = close
        stop = close - atr
        target1 = close + atr
        target2 = close + atr * 1.50

        advice = "等待進一步確認"

    # --------------------------------------------------------
    # 風險報酬比
    # --------------------------------------------------------

    risk = abs(entry - stop)

    if risk > 0:

        rr1 = abs(target1 - entry) / risk
        rr2 = abs(target2 - entry) / risk

    else:

        rr1 = 0
        rr2 = 0

    return pd.Series({

        "進場參考價": round(entry, 2),
        "突破確認價": round(breakout, 2),
        "回檔買進價": round(entry, 2),
        "停損參考價": round(stop, 2),
        "第一目標價": round(target1, 2),
        "第二目標價": round(target2, 2),
        "風險報酬比1": round(rr1, 2),
        "風險報酬比2": round(rr2, 2),
        "操作建議": advice
    })

def main():

    print()
    print("=" * 70)
    print("台股當沖選股 v8.0")
    print("=" * 70)
    print()

    ensure_cache_is_updated()

    data = load_all_cache()

    results = []

    print("開始分析股票...")
    print()

    for i, (code, df) in enumerate(
        data.items(),
        1
    ):

        result = process_stock(
            code,
            df
        )

        if result is not None:
            results.append(result)

        if i % 200 == 0:
            print(
                f"分析進度：{i} / {len(data)}"
            )

    if not results:

        print()
        print("仍然沒有符合條件的股票。")
        print()
        print("這表示 cache 的資料格式需要再確認。")
        return

    df = pd.DataFrame(results)

    df["當沖強度"] = df.apply(
        calculate_score,
        axis=1
    )

    df["當沖訊號"] = None
    df["訊號理由"] = None

    df["操作建議"] = None

    signals = df.apply(
        judgement,
        axis=1,
        result_type="expand"
    )

    df["當沖訊號"] = signals[0]
    df["訊號理由"] = signals[1]

    levels = df.apply(
        trade_levels,
        axis=1,
        result_type="expand"
    )

    levels.columns = [
        "進場參考價",
        "停損參考價",
        "第一目標價",
        "第二目標價",
        "風險報酬比1",
        "風險報酬比2"
    ]

    for col in levels.columns:
        df[col] = levels[col]

    df["操作建議"] = df["當沖訊號"].apply(
        lambda x: {
            "突破買進觀察": "突破確認後進場，不追高",
            "回檔買進觀察": "等待拉回支撐區再進場",
            "多方觀察": "偏多觀察，等待量價確認",
            "高波動控制部位": "波動過大，降低部位",
            "偏空高風險": "偏空高風險，不建議追多",
            "量能異常": "量能異常，等待方向確認",
            "量能不足": "量能不足，暫不追價",
            "過熱不追": "RSI偏高，不建議追價",
            "等待確認": "等待量價確認",
            "觀望": "暫時觀望"
        }.get(x, "等待進一步確認")
    )
    # 適合度

    df["當沖適合度"] = df[
        "當沖強度"
    ].apply(
        lambda x:
            "★★★★★ 極佳" if x >= 90 else
            "★★★★☆ 很佳" if x >= 80 else
            "★★★☆☆ 良好" if x >= 70 else
            "★★☆☆☆ 普通" if x >= 60 else
            "★☆☆☆☆ 不建議"
    )

    # 風險

    df["當沖風險"] = df.apply(
        lambda r:
            "極端風險"
            if r["當日振幅"] >= 15 or r["ATR%"] >= 12
            else "高風險"
            if r["當日振幅"] >= 10 or r["ATR%"] >= 8
            else "中風險"
            if r["當日振幅"] >= 5 or r["ATR%"] >= 4
            else "低風險",
        axis=1
    )

    df["操作建議"] = df[
        "當沖訊號"
    ].map({

        "突破買進觀察":
            "可列入盤中觀察",

        "偏多觀察":
            "可列入盤中觀察",

        "回檔買進觀察":
            "等待回檔",

        "過熱不追":
            "不追價",

        "偏空高風險":
            "避免做多",

        "極端波動":
            "降低部位",

        "高波動控制部位":
            "降低部位",

        "量能不足":
            "等待量能",

        "量能異常":
            "等待價格確認",

        "等待確認":
            "等待確認",

        "觀望":
            "暫不操作"
    }).fillna("等待確認")

    df = df.sort_values(
        ["當沖強度", "量比"],
        ascending=[False, False]
    ).reset_index(drop=True)

    df.insert(
        0,
        "排名",
        range(1, len(df) + 1)
    )

    top10 = df.head(10).copy()

    breakthrough = df[
        df["當沖訊號"] == "突破買進觀察"
    ].copy()

    pullback = df[
        df["當沖訊號"] == "回檔買進觀察"
    ].copy()

    bullish = df[
        df["當沖訊號"].isin([
            "突破買進觀察",
            "偏多觀察",
            "回檔買進觀察"
        ])
    ].copy()

    high_risk = df[
        df["當沖風險"].isin([
            "高風險",
            "極端風險"
        ])
    ].copy()

    columns = [
        "排名",
        "股票代號",
        "收盤價",
        "漲跌幅",
        "成交量",
        "5日平均成交量",
        "20日平均成交量",
        "成交金額",
        "5日平均成交金額",
        "量比",
        "5日/20日量能",
        "RSI14",
        "ATR%",
        "當日振幅",
        "5日平均振幅",
        "MA5",
        "MA10",
        "MA20",
        "MACD",
        "MACD訊號",
        "MACD柱",
        "當沖強度",
        "當沖適合度",
        "當沖風險",
        "當沖訊號",
        "訊號理由",
        "進場參考價",
        "停損參考價",
        "第一目標價",
        "第二目標價",
        "風險報酬比1",
        "風險報酬比2",
        "操作建議"
    ]

    columns = [
        c for c in columns
        if c in df.columns
    ]

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        top10[columns].to_excel(
            writer,
            sheet_name="TOP10",
            index=False
        )

        df[columns].to_excel(
            writer,
            sheet_name="全部候選股",
            index=False
        )

        breakthrough[columns].to_excel(
            writer,
            sheet_name="突破買進",
            index=False
        )

        pullback[columns].to_excel(
            writer,
            sheet_name="回檔觀察",
            index=False
        )

        bullish[columns].to_excel(
            writer,
            sheet_name="多方觀察",
            index=False
        )

        high_risk[columns].to_excel(
            writer,
            sheet_name="高風險",
            index=False
        )

    wb = load_workbook(
        OUTPUT_FILE
    )

    for ws in wb.worksheets:
        format_sheet(ws)

    wb.save(
        OUTPUT_FILE
    )

    # ========================================================
    # 終端輸出
    # ========================================================

    print()
    print("=" * 70)
    print("台股當沖 TOP 10")
    print("=" * 70)

    show = [
        "排名",
        "股票代號",
        "收盤價",
        "漲跌幅",
        "量比",
        "RSI14",
        "ATR%",
        "當沖強度",
        "當沖訊號",
        "當沖風險"
    ]

    print(
        top10[show].to_string(
            index=False
        )
    )

    print("=" * 70)
    print("Excel 已產生")
    print(f"輸出：{OUTPUT_FILE}")
    print("完成！")
    print("=" * 70)

    print(
        f"符合條件：{len(df)} 檔"
    )

    print(
        f"突破買進觀察："
        f"{len(breakthrough)} 檔"
    )

    print(
        f"回檔買進觀察："
        f"{len(pullback)} 檔"
    )

    print(
        f"多方觀察："
        f"{len(bullish)} 檔"
    )

    print(
        f"高風險："
        f"{len(high_risk)} 檔"
    )

    print()
    print(
        f"輸出：{OUTPUT_FILE}"
    )
    print()


if __name__ == "__main__":
    main()