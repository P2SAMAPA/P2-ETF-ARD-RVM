import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from ard_rvm import ARDRVM

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def create_features(returns_df, macro_df, etf, window, lags):
    """Build feature matrix X and target y for a single ETF."""
    ret = returns_df[etf].iloc[-window:].copy()
    macro_win = macro_df.iloc[-window:] if not macro_df.empty else pd.DataFrame(0, index=ret.index, columns=config.MACRO_COLUMNS)
    # Align indices
    common = ret.index.intersection(macro_win.index)
    ret = ret.loc[common]
    macro_win = macro_win.loc[common]
    # Create feature vectors for each day: lagged returns + macro levels
    # We need one observation per day (target = next day's return)
    data = pd.DataFrame(index=ret.index)
    for lag in lags:
        data[f'lag_{lag}'] = ret.shift(lag)
    for col in macro_win.columns:
        data[col] = macro_win[col]
    data['target'] = ret.shift(-1)
    data = data.dropna()
    if len(data) < 20:
        return None, None
    X = data.drop('target', axis=1).values
    y = data['target'].values
    return X, y

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (ARD-RVM) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 50:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        macro = data_manager.get_macro_data(df)
        if macro.empty:
            print("  No macro data; using zeros")
            macro = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 20:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            etf_pred = {}
            relevance_counts = {}
            for etf in tickers:
                X, y = create_features(returns, macro, etf, win, config.LAG_DAYS)
                if X is None:
                    continue
                model = ARDRVM(max_iter=config.MAX_ITER, tol=config.TOL)
                model.fit(X, y)
                # Predict for the most recent feature vector (last row of X)
                last_X = X[-1:].reshape(1, -1)
                pred = model.predict(last_X)[0]
                etf_pred[etf] = pred
                relevance_counts[etf] = model.get_relevance_counts()
            window_results[win] = {"predictions": etf_pred, "relevance_counts": relevance_counts}
            for etf, pred in etf_pred.items():
                if etf not in best_per_etf or pred > best_per_etf[etf][0]:
                    best_per_etf[etf] = (pred, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "pred_return": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by predicted return: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/ard_rvm_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Sparse Bayesian ARD-RVM Engine complete ===")

if __name__ == "__main__":
    main()
