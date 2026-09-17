import os
import pickle
import shutil
import time

import pandas as pd
import yfinance as yf


# ============================================================
# 設定
# ============================================================

STOCK_POOL_FILE = "完整台股股票池.xlsx"
CACHE_DIR = "cache_stock_data"

BATCH_SIZE = 80
PERIOD = "6mo"

DOWNLOAD_TIMEOUT = 30


# ============================================================
# 讀取股票池
# ============================================================

def load_stock_pool():

    df = pd.read_excel(STOCK_POOL_FILE)

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

    print(
        f"股票池：{len(df)} 檔"
    )

    return df


# ============================================================
# 下載單一批次
# ============================================================

def download_batch(tickers):

    print(
        f"開始下載：{len(tickers)} 檔"
    )

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

        raise RuntimeError(
            "Yahoo Finance 沒有回傳資料"
        )

    return data


# ============================================================
# 儲存 Batch
# ============================================================

def save_batch(data, batch_number):

    os.makedirs(
        CACHE_DIR,
        exist_ok=True
    )

    path = os.path.join(
        CACHE_DIR,
        f"batch_{batch_number}.pkl"
    )

    with open(path, "wb") as f:

        pickle.dump(
            data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(
        f"已儲存：{path}"
    )


# ============================================================
# 更新 Cache
# ============================================================

def update_cache():

    print()
    print("=" * 70)
    print("台股資料 Cache 更新程式")
    print("=" * 70)
    print()

    stock_pool = load_stock_pool()

    tickers = (
        stock_pool["Yahoo代號"]
        .dropna()
        .astype(str)
        .tolist()
    )

    total = len(tickers)

    batches = [
        tickers[i:i + BATCH_SIZE]
        for i in range(
            0,
            total,
            BATCH_SIZE
        )
    ]

    print(
        f"總股票數：{total}"
    )

    print(
        f"批次數：{len(batches)}"
    )

    print()

    # --------------------------------------------------------
    # 建立暫存資料夾
    # --------------------------------------------------------

    temp_dir = CACHE_DIR + "_new"

    if os.path.exists(temp_dir):

        shutil.rmtree(
            temp_dir
        )

    os.makedirs(
        temp_dir
    )

    # --------------------------------------------------------
    # 逐批下載
    # --------------------------------------------------------

    for i, batch in enumerate(
        batches,
        1
    ):

        print()
        print(
            f"[{i}/{len(batches)}] "
            f"下載 {len(batch)} 檔"
        )

        try:

            data = download_batch(
                batch
            )

            if data.empty:

                print(
                    "  ⚠️ 沒有資料，跳過"
                )

                continue

            path = os.path.join(
                temp_dir,
                f"batch_{i}.pkl"
            )

            with open(
                path,
                "wb"
            ) as f:

                pickle.dump(
                    data,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL
                )

            print(
                f"  ✓ 儲存 {path}"
            )

        except Exception as e:

            print(
                f"  ✗ 下載失敗：{e}"
            )

        time.sleep(1)

    # --------------------------------------------------------
    # 確認至少有資料
    # --------------------------------------------------------

    files = [
        f
        for f in os.listdir(temp_dir)
        if f.endswith(".pkl")
    ]

    if not files:

        shutil.rmtree(
            temp_dir
        )

        raise RuntimeError(
            "更新失敗：沒有產生任何 Cache"
        )

    # --------------------------------------------------------
    # 替換舊 Cache
    # --------------------------------------------------------

        # --------------------------------------------------------
    # 安全替換正式 Cache
    # --------------------------------------------------------

    expected_batches = len(batches)

    if len(files) != expected_batches:

        shutil.rmtree(
            temp_dir
        )

        raise RuntimeError(
            f"Cache 更新失敗：預期 {expected_batches} 批，"
            f"實際只有 {len(files)} 批"
        )

    print()
    print("正在驗證新 Cache...")

    latest_dates = []

    for filename in files:

        path = os.path.join(
            temp_dir,
            filename
        )

        with open(path, "rb") as f:
            batch_data = pickle.load(f)

        if batch_data is None or batch_data.empty:

            shutil.rmtree(
                temp_dir
            )

            raise RuntimeError(
                f"Cache 驗證失敗：{filename} 沒有資料"
            )

        latest_dates.append(
            batch_data.index.max()
        )

    latest_date = max(latest_dates)
    earliest_date = min(latest_dates)

    print(
        f"Cache 日期範圍：{earliest_date} ~ {latest_date}"
    )

    if latest_date != earliest_date:

        shutil.rmtree(
            temp_dir
        )

        raise RuntimeError(
            "Cache 更新失敗：不同批次的最新日期不一致"
        )

    print()
    print(
        f"✓ {len(files)}/{expected_batches} 個批次驗證成功"
    )
    print(
        f"✓ 最新交易日：{latest_date}"
    )

    old_dir = CACHE_DIR + "_old"

    if os.path.exists(old_dir):

        shutil.rmtree(
            old_dir
        )

    if os.path.exists(CACHE_DIR):

        os.rename(
            CACHE_DIR,
            old_dir
        )

    try:

        os.rename(
            temp_dir,
            CACHE_DIR
        )

    except Exception:

        if os.path.exists(old_dir):

            os.rename(
                old_dir,
                CACHE_DIR
            )

        raise

    if os.path.exists(old_dir):

        shutil.rmtree(
            old_dir
        )

    print()
    print("✓ 正式 Cache 已更新")
    print(
        f"✓ 最新交易日：{latest_date}"
    )

    print()
    print("=" * 70)
    print("Cache 更新完成")
    print("=" * 70)
    print(
        f"產生批次：{len(files)}"
    )
    print()


# ============================================================
# 主程式
# ============================================================

if __name__ == "__main__":

    update_cache()