#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SINO Pipeline - Seleção de Estratégias
--------------------------------------
Fluxo:
1. Carrega CSV com pnl cumulativo (colunas = estratégias, índice = datas).
2. Converte em retornos diários.
3. Faz walk-forward IS/OOS folds.
4. Calcula métricas (Sharpe, Drawdown, Calmar, etc).
5. Estima PBO simplificado.
6. Faz bootstrap para IC do Sharpe.
7. Filtra estratégias e clusteriza por correlação.
8. Seleciona e aloca pesos via Equal Risk Contribution (ERC).
9. Estima margem requerida (VaR/ES básico).
10. Exporta relatórios (CSV/JSON).
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.cluster import AgglomerativeClustering

# -----------------------------
# Utils
# -----------------------------
def compute_returns(df_cum):
    return df_cum.diff().fillna(0)

def sharpe_ratio(returns, freq=252):
    mean = returns.mean() * freq
    vol = returns.std(ddof=1) * np.sqrt(freq)
    return mean / vol if vol > 0 else 0.0

def max_drawdown(cum):
    roll_max = cum.cummax()
    dd = cum - roll_max
    return dd.min()

def calmar_ratio(cum, returns, freq=252):
    dd = abs(max_drawdown(cum))
    if dd == 0:
        return 0.0
    cagr = (cum.iloc[-1] / cum.iloc[0]) ** (freq / len(cum)) - 1
    return cagr / dd

# -----------------------------
# Walk-forward folds
# -----------------------------
def walkforward_folds(df_returns, n_folds=5, train_size=0.7):
    T = len(df_returns)
    train_len = int(T * train_size)
    step = int((T - train_len) / n_folds)
    folds = []
    for i in range(n_folds):
        train_start = i * step
        train_end = train_start + train_len
        test_end = min(train_end + step, T)
        if test_end <= train_end:
            break
        folds.append(((train_start, train_end), (train_end, test_end)))
    return folds

# -----------------------------
# Metrics por estratégia
# -----------------------------
def compute_metrics(df_cum, n_folds=5):
    df_ret = compute_returns(df_cum)
    folds = walkforward_folds(df_ret, n_folds=n_folds)

    metrics = {}
    for col in df_cum.columns:
        strat = {}
        sharpe_is = []
        sharpe_oos = []
        for (tr0, tr1), (te0, te1) in folds:
            r_train = df_ret[col].iloc[tr0:tr1]
            r_test = df_ret[col].iloc[te0:te1]
            sharpe_is.append(sharpe_ratio(r_train))
            sharpe_oos.append(sharpe_ratio(r_test))
        strat["sharpe_is_mean"] = np.mean(sharpe_is)
        strat["sharpe_oos_mean"] = np.mean(sharpe_oos)
        strat["dd"] = max_drawdown(df_cum[col])
        strat["calmar"] = calmar_ratio(df_cum[col], df_ret[col])
        metrics[col] = strat

    df_metrics = pd.DataFrame(metrics).T
    return df_metrics

# -----------------------------
# PBO simplificado
# -----------------------------
def pbo_estimate(df_cum, n_folds=5):
    df_ret = compute_returns(df_cum)
    folds = walkforward_folds(df_ret, n_folds=n_folds)
    n_fail = 0
    for (tr0, tr1), (te0, te1) in folds:
        sharpe_train = df_ret.iloc[tr0:tr1].apply(sharpe_ratio)
        sharpe_test = df_ret.iloc[te0:te1].apply(sharpe_ratio)
        best = sharpe_train.idxmax()
        if sharpe_test[best] < sharpe_test.median():
            n_fail += 1
    return n_fail / len(folds)

# -----------------------------
# Bootstrap IC Sharpe
# -----------------------------
def bootstrap_sharpe(series, n_boot=200, block_size=20):
    T = len(series)
    n_blocks = T // block_size
    sr_list = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n_blocks, n_blocks)
        sampled = np.concatenate([series[i*block_size:(i+1)*block_size] for i in idx])
        sr_list.append(sharpe_ratio(pd.Series(sampled)))
    return np.percentile(sr_list, [5, 50, 95])

# -----------------------------
# Clusterização por correlação
# -----------------------------
def cluster_strategies(df_returns, threshold=0.7):
    corr = df_returns.corr()
    dist = 1 - corr.abs()
    model = AgglomerativeClustering(
        affinity="precomputed", linkage="average", distance_threshold=1-threshold, n_clusters=None
    )
    labels = model.fit_predict(dist)
    return pd.Series(labels, index=df_returns.columns)

# -----------------------------
# Alocação Equal Risk Contribution
# -----------------------------
def equal_risk_contribution(df_returns, selected):
    X = df_returns[selected]
    cov = LedoitWolf().fit(X).covariance_
    vol = np.sqrt(np.diag(cov))
    inv_vol = 1 / vol
    w = inv_vol / inv_vol.sum()
    return pd.Series(w, index=selected)

# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(outdir="./sino_out", capital=1_000_000):
    os.makedirs(outdir, exist_ok=True)
    df_cum = pd.read_parquet('mts2.parquet')

    # Metrics
    df_metrics = compute_metrics(df_cum)
    df_metrics.to_csv(os.path.join(outdir, "metrics_all.csv"))

    # PBO
    pbo = pbo_estimate(df_cum)
    with open(os.path.join(outdir, "pbo.json"), "w") as f:
        json.dump({"pbo": pbo}, f)

    # Bootstrap ICs
    df_ret = compute_returns(df_cum)
    ic_data = {}
    for col in df_ret.columns:
        ic_data[col] = bootstrap_sharpe(df_ret[col])
    pd.DataFrame(ic_data, index=["p5", "p50", "p95"]).T.to_csv(
        os.path.join(outdir, "bootstrap_sharpe.csv")
    )

    # Seleção: filtros simples
    filt = df_metrics[
        (df_metrics["sharpe_oos_mean"] > 0.5) &
        (df_metrics["calmar"] > 0.2)
    ]
    selected = filt.index.tolist()

    # Clusterização
    if len(selected) > 1:
        labels = cluster_strategies(df_ret[selected])
        reps = labels.groupby(labels).apply(lambda g: g.index[0])
        selected = reps.tolist()

    # Pesos ERC
    weights = equal_risk_contribution(df_ret, selected)
    weights.to_csv(os.path.join(outdir, "weights.csv"))

    # Margem (VaR/ES simplificado)
    margin_report = {}
    for strat, w in weights.items():
        r = df_ret[strat]
        var = -np.percentile(r, 1) * capital * w
        es = -r[r <= np.percentile(r, 1)].mean() * capital * w
        margin_report[strat] = {"weight": w, "VaR_99": var, "ES_99": es}
    with open(os.path.join(outdir, "margin_report.json"), "w") as f:
        json.dump(margin_report, f, indent=2)

    # Save selected
    pd.Series(selected).to_csv(os.path.join(outdir, "selected_strats.csv"), index=False)

    print(f"Pipeline concluído. Resultados em {outdir}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="./sino_out", help="Diretório de saída")
    parser.add_argument("--capital", type=float, default=100_000, help="Capital base para margem")
    args = parser.parse_args()
    run_pipeline(outdir=args.outdir, capital=args.capital)