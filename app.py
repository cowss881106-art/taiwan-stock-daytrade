import os
import subprocess
import sys
import pandas as pd
import streamlit as st


OUTPUT_FILE = "台股當沖選股_TOP10_v9.xlsx"


st.set_page_config(
    page_title="台股當沖選股 v9",
    page_icon="📈",
    layout="wide"
)


st.title("📈 台股當沖選股 v9")
st.caption("台股當沖 TOP 10｜資料由 Python 自動分析")


def run_selector():
    with st.spinner("正在執行台股當沖選股，請稍候..."):
        result = subprocess.run(
            [sys.executable, "stock_selector_v9.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

    if result.returncode != 0:
        st.error("選股程式執行失敗")
        st.code(result.stderr)
        return False

    return True


def load_excel():
    if not os.path.exists(OUTPUT_FILE):
        return None

    return pd.read_excel(OUTPUT_FILE)


# ============================================================
# 更新按鈕
# ============================================================

if st.button("🔄 立即更新選股", type="primary"):

    success = run_selector()

    if success:
        st.success("選股完成！")
        st.rerun()


# ============================================================
# 顯示 Excel
# ============================================================

df = load_excel()

if df is None:

    st.info("目前還沒有選股結果，請按「立即更新選股」。")

else:

    st.success("目前已有最新選股結果")

    st.subheader("🏆 台股當沖 TOP 10")

    # 只顯示 TOP 10
    top10 = df.head(10)

    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # Excel 下載
    # ========================================================

    with open(OUTPUT_FILE, "rb") as f:
        excel_data = f.read()

    st.download_button(
        label="📥 下載完整 Excel",
        data=excel_data,
        file_name=OUTPUT_FILE,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )