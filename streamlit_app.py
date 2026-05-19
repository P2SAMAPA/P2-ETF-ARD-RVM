import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="ARD-RVM Sparse Bayesian", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📉 Sparse Bayesian ARD‑RVM Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Relevance Vector Machine | Automatic Relevance Determination | Type‑II ML | Prunes irrelevant features | Probabilistic predictions | Multi‑window evaluation</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 📉 ARD‑RVM")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown("**Method:** Bayesian linear model with ARD")
st.sidebar.markdown("**Windows evaluated:** 63, 252, 504, 1008, 2016 days (best per ETF)")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'ard_rvm_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by ARD‑RVM Predicted Return")

with st.expander("📖 How it works", expanded=True):
    st.markdown("""
    - **Relevance Vector Machine (RVM)** is a Bayesian linear model with a separate precision hyperparameter (αᵢ) for each feature.
    - **Automatic Relevance Determination (ARD)** drives irrelevant αᵢ to infinity, pruning those features (exact zeros).
    - Type‑II maximum likelihood (evidence maximisation) optimises αᵢ and noise precision β.
    - The model is sparser than Lasso and provides probabilistic predictions (variance available, not used here).
    - Features: lagged returns of the ETF (1,2,3,5,10,21 days) and macro levels (VIX, DXY, etc.).
    - For each ETF and each rolling window, the RVM is trained, then predicts the next day's return.
    - **Ranking:** ETFs with highest predicted return are selected.
    - For each ETF, the window that gives the highest predicted return is chosen.
    """)

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">pred return = {etf['pred_return']:.6f}</div>
                <div class="etf-score">best window = {etf.get('best_window', 'N/A')}d</div>
            </div>
            """, unsafe_allow_html=True)
    # Optionally show relevance counts for the best window
    win_res = uni_data.get("window_results", {})
    if win_res:
        best_win = top_etfs[0]['best_window'] if top_etfs else None
        if best_win is not None and str(best_win) in win_res:
            rel_counts = win_res[str(best_win)].get("relevance_counts", {})
            if rel_counts:
                # Show a small table of relevance counts for top 3 ETFs
                st.write(f"**Relevance (number of kept features) for the best window ({best_win}d):**")
                for etf in top_etfs:
                    cnt = rel_counts.get(etf['ticker'], 0)
                    st.write(f"- {etf['ticker']}: {cnt} relevant features")
    with st.expander("📋 Full ranking (all ETFs, best window per ETF)"):
        full = uni_data.get("full_scores", {})
        if full:
            rows = []
            for ticker, info in full.items():
                if isinstance(info, dict):
                    score = info.get("score", 0.0)
                    win = info.get("best_window", "N/A")
                else:
                    score = info
                    win = "N/A"
                rows.append({"ETF": ticker, "Predicted Return": score, "Best Window": win})
            df = pd.DataFrame(rows)
            df["Predicted Return"] = pd.to_numeric(df["Predicted Return"], errors='coerce')
            df = df.dropna(subset=["Predicted Return"]).sort_values("Predicted Return", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("ARD‑RVM trains a Bayesian linear model with feature‑specific precisions. Irrelevant features are pruned automatically (α → ∞). The predicted return is the linear combination of kept features. Higher predicted return → stronger long signal. For each ETF, the window giving the highest predicted return is selected.")
