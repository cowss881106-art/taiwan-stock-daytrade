import io
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

STOCK_POOL_FILE = "完整台股股票池.xlsx"
MIN_AVG_VOLUME_5D = 6000
MIN_AVG_RANGE_5D = 2.0

TWSE_T86 = "https://www.twse.com.tw/rwd/zh/fund/T86"
TPEX_3INSTI = "https://www.tpex.org.tw/www/zh-tw/3insti/dailyTrade/3itrade_hedge"

HEADERS = {"User-Agent": "Mozilla/5.0"}

def normalize_code(x):
    s = str(x).strip().replace("'", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(4) if s.isdigit() and len(s) < 4 else s

def load_stock_pool():
    df = pd.read_excel(STOCK_POOL_FILE, dtype={"股票代號": str})
    df["股票代號"] = df["股票代號"].map(normalize_code)
    return df.drop_duplicates("股票代號").reset_index(drop=True)

def yahoo_symbol(code, market):
    market = str(market)
    return f"{code}.TWO" if ("上櫃" in market or "OTC" in market) else f"{code}.TW"

def fetch_daily_metrics(pool):
    symbols = {yahoo_symbol(r["股票代號"], r["市場"]): r["股票代號"]
               for _, r in pool.iterrows()}
    tickers = list(symbols)
    out = []
    batch_size = 100

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start+batch_size]
        data = yf.download(
            batch, period="12d", interval="1d",
            group_by="ticker", auto_adjust=False,
            progress=False, threads=True
        )
        for ticker in batch:
            try:
                x = data[ticker].dropna(how="all") if len(batch) > 1 else data.dropna(how="all")
                if len(x) < 5:
                    continue
                x = x.tail(5)
                vol = pd.to_numeric(x["Volume"], errors="coerce") / 1000.0
                high = pd.to_numeric(x["High"], errors="coerce")
                low = pd.to_numeric(x["Low"], errors="coerce")
                close = pd.to_numeric(x["Close"], errors="coerce")
                rng = (high - low) / close * 100
                out.append({
                    "股票代號": symbols[ticker],
                    "5日平均成交量(張)": round(float(vol.mean()), 1),
                    "5日平均振幅%": round(float(rng.mean()), 2),
                })
            except Exception:
                continue
        time.sleep(0.2)

    return pd.DataFrame(out)

def fetch_json(url, params):
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_twse(date):
    ds = date.strftime("%Y%m%d")
    j = fetch_json(TWSE_T86, {"date": ds, "selectType": "ALL", "response": "json"})
    fields, data = j.get("fields", []), j.get("data", [])
    if not data:
        return pd.DataFrame()
    d = pd.DataFrame(data, columns=fields)
    code_col = next((c for c in d.columns if "證券代號" in c), None)
    foreign_col = next((c for c in d.columns if "外陸資買賣超股數" in c and "不含" not in c), None)
    if not code_col or not foreign_col:
        return pd.DataFrame()
    z = pd.DataFrame({
        "股票代號": d[code_col].map(normalize_code),
        "外資買賣超(張)": pd.to_numeric(
            d[foreign_col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        ) / 1000.0
    })
    return z.dropna()

def fetch_tpex(date):
    roc = f"{date.year-1911}/{date.month:02d}/{date.day:02d}"
    j = fetch_json(TPEX_3INSTI, {"date": roc, "response": "json"})
    tables = j.get("tables", [])
    if not tables:
        return pd.DataFrame()
    fields, data = tables[0].get("fields", []), tables[0].get("data", [])
    if not data:
        return pd.DataFrame()
    d = pd.DataFrame(data, columns=fields)
    code_col = next((c for c in d.columns if "代號" in c), None)
    foreign_col = next((c for c in d.columns if "外資及陸資" in c and "買賣超" in c), None)
    if not code_col or not foreign_col:
        return pd.DataFrame()
    z = pd.DataFrame({
        "股票代號": d[code_col].map(normalize_code),
        "外資買賣超(張)": pd.to_numeric(
            d[foreign_col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        ) / 1000.0
    })
    return z.dropna()

def get_recent_institutional_days(n=2):
    found = []
    d = datetime.now().date() - timedelta(days=1)
    attempts = 0
    while len(found) < n and attempts < 12:
        if d.weekday() < 5:
            try:
                tw = fetch_twse(d)
                ot = fetch_tpex(d)
                merged = pd.concat([tw, ot], ignore_index=True)
                if len(merged) > 100:
                    found.append((d, merged.drop_duplicates("股票代號")))
            except Exception:
                pass
        d -= timedelta(days=1)
        attempts += 1
    if len(found) < n:
        raise RuntimeError("找不到最近兩個完整法人交易日")
    return found

def build_watchlist():
    pool = load_stock_pool()
    metrics = fetch_daily_metrics(pool)
    base = pool.merge(metrics, on="股票代號", how="inner")
    base = base[
        (base["5日平均成交量(張)"] >= MIN_AVG_VOLUME_5D) &
        (base["5日平均振幅%"] >= MIN_AVG_RANGE_5D)
    ].copy()

    days = get_recent_institutional_days(2)
    d1, inst1 = days[0]
    d2, inst2 = days[1]
    inst1 = inst1.rename(columns={"外資買賣超(張)": "最近日外資買賣超(張)"})
    inst2 = inst2.rename(columns={"外資買賣超(張)": "前一日外資買賣超(張)"})

    x = base.merge(inst1, on="股票代號", how="inner").merge(inst2, on="股票代號", how="inner")
    x["盤前方向"] = np.where(
        (x["最近日外資買賣超(張)"] > 0) & (x["前一日外資買賣超(張)"] > 0), "多方候選",
        np.where(
            (x["最近日外資買賣超(張)"] < 0) & (x["前一日外資買賣超(張)"] < 0), "空方候選", ""
        )
    )
    x = x[x["盤前方向"] != ""].copy()
    x["法人最近日"] = str(d1)
    x["法人前一日"] = str(d2)
    return x.sort_values(["盤前方向", "5日平均成交量(張)"], ascending=[True, False]).reset_index(drop=True)

def to_excel_bytes(df):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="今日監控池")
    bio.seek(0)
    return bio.getvalue()
