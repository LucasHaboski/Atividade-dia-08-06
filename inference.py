import os
import warnings
import joblib
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from pipeline import MODEL_PATH, FIGURES_DIR, PAY_COLS, BILL_COLS, PAY_AMT_COLS, create_features, clean_data

CLIENT_COLUMNS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS

_RISK_LEVELS = [
    (0.20, "BAIXO",       "Cliente de baixo risco"),
    (0.40, "MÉDIO-BAIXO", "Monitorar com atenção"),
    (0.60, "MÉDIO",       "Risco considerável"),
    (0.80, "MÉDIO-ALTO",  "Alto risco de inadimplência"),
    (1.01, "ALTO",        "Risco muito alto de inadimplência"),
]


# Carrega o pipeline e o limiar ótimo salvos pelo pipeline.py
def load_model(path: str = MODEL_PATH) -> tuple:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{path}'.\nExecute primeiro: python pipeline.py"
        )
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        model, threshold = artifact["model"], artifact["threshold"]
    else:
        model, threshold = artifact, 0.5
    print(f"[OK] Modelo carregado  (limiar={threshold:.3f})")
    return model, threshold


# Retorna predição, score e distribuição de probabilidade para um único cliente
def predict_default(client_data: dict[str, Any], model=None, threshold: float = None) -> dict:
    if model is None or threshold is None:
        loaded_model, loaded_threshold = load_model()
        model     = model     or loaded_model
        threshold = threshold or loaded_threshold

    df = _prepare_input(pd.DataFrame([client_data]))
    return _make_result(model, df, threshold)


# Retorna predições em lote com score e nível de risco para cada cliente
def predict_batch(clients_df: pd.DataFrame, model=None, threshold: float = None) -> pd.DataFrame:
    if model is None or threshold is None:
        loaded_model, loaded_threshold = load_model()
        model     = model     or loaded_model
        threshold = threshold or loaded_threshold

    df    = _prepare_input(clients_df.copy())
    probs = model.predict_proba(df)[:, 1]

    out = clients_df.copy()
    out["prediction"]          = (probs >= threshold).astype(int)
    out["probability_default"] = probs
    out["score"]               = (probs * 100).astype(int)
    out["risk_level"]          = [_risk_label(p)[0] for p in probs]
    return out


# Gera gráfico de barras com a distribuição de probabilidade de inadimplência do cliente
def plot_probability_distribution(result: dict, client_id: str = "Cliente") -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    dist = result["probability_distribution"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Não Inadimplente", "Inadimplente"],
        [dist["nao_inadimplente"], dist["inadimplente"]],
        color=["steelblue", "tomato"], edgecolor="white", alpha=0.85,
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Probabilidade")
    ax.set_title(f"Distribuição de Probabilidade — {client_id}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    for bar, val in zip(bars, [dist["nao_inadimplente"], dist["inadimplente"]]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
        )

    color = "tomato" if result["prediction"] == 1 else "steelblue"
    plt.suptitle(
        f"{result['prediction_label']} — Score {result['score']}/100",
        fontsize=11, color=color,
    )
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, f"prob_{client_id.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [fig] {path}")


# Imprime o resultado de risco formatado no terminal
def print_result(result: dict, client_id: str = "Cliente") -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print(f" RESULTADO — {client_id}")
    print(sep)
    print(f"  Predição      : {result['prediction_label']}")
    print(f"  Score de risco: {result['score']:>3}/100")
    print(f"  Probabilidade : {result['probability']:.1%}")
    print(f"  Limiar usado  : {result['threshold_used']:.3f}")
    print(f"  Nível de risco: {result['risk_level']}")
    print(f"  Avaliação     : {result['risk_message']}")
    d = result["probability_distribution"]
    print(f"\n  Distribuição de probabilidade:")
    print(f"    Não Inadimplente : {d['nao_inadimplente']:.1%}")
    print(f"    Inadimplente     : {d['inadimplente']:.1%}")
    print(sep)


# Aplica a mesma limpeza e feature engineering usados no treino
def _prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "ID", range(len(df)))
    df["default payment next month"] = 0
    df = clean_data(df)
    df = create_features(df)
    return df.drop(columns=["default"], errors="ignore")


def _make_result(model, df: pd.DataFrame, threshold: float) -> dict:
    probs = model.predict_proba(df)[0]
    p_def = float(probs[1])
    pred  = int(p_def >= threshold)
    level, message = _risk_label(p_def)

    return {
        "prediction":              pred,
        "prediction_label":        "INADIMPLENTE" if pred == 1 else "NÃO INADIMPLENTE",
        "probability":             p_def,
        "threshold_used":          threshold,
        "score":                   int(p_def * 100),
        "risk_level":              level,
        "risk_message":            message,
        "probability_distribution": {
            "nao_inadimplente": float(probs[0]),
            "inadimplente":     p_def,
        },
    }


def _risk_label(prob: float) -> tuple:
    for threshold, level, message in _RISK_LEVELS:
        if prob < threshold:
            return level, message
    return _RISK_LEVELS[-1][1], _RISK_LEVELS[-1][2]


_EXEMPLO_BAIXO_RISCO = {
    "LIMIT_BAL": 350000, "SEX": "F", "EDUCATION": "University",
    "MARRIAGE": "Single", "AGE": 32,
    "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 12000, "BILL_AMT2": 11000, "BILL_AMT3": 10500,
    "BILL_AMT4": 10000, "BILL_AMT5": 9500,  "BILL_AMT6": 9000,
    "PAY_AMT1": 12000,  "PAY_AMT2": 11000,  "PAY_AMT3": 10500,
    "PAY_AMT4": 10000,  "PAY_AMT5": 9500,   "PAY_AMT6": 9000,
}

_EXEMPLO_MEDIO_RISCO = {
    "LIMIT_BAL": 80000, "SEX": "M", "EDUCATION": "High School",
    "MARRIAGE": "Married", "AGE": 38,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 55000, "BILL_AMT2": 52000, "BILL_AMT3": 48000,
    "BILL_AMT4": 45000, "BILL_AMT5": 40000, "BILL_AMT6": 38000,
    "PAY_AMT1": 2000,  "PAY_AMT2": 1500,  "PAY_AMT3": 2500,
    "PAY_AMT4": 2000,  "PAY_AMT5": 1800,  "PAY_AMT6": 2000,
}

_EXEMPLO_ALTO_RISCO = {
    "LIMIT_BAL": 20000, "SEX": "M", "EDUCATION": "High School",
    "MARRIAGE": "Married", "AGE": 26,
    "PAY_0": 3, "PAY_2": 2, "PAY_3": 2, "PAY_4": 1, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 19500, "BILL_AMT2": 19000, "BILL_AMT3": 18500,
    "BILL_AMT4": 18000, "BILL_AMT5": 17500, "BILL_AMT6": 17000,
    "PAY_AMT1": 300,  "PAY_AMT2": 0,    "PAY_AMT3": 500,
    "PAY_AMT4": 0,    "PAY_AMT5": 200,  "PAY_AMT6": 300,
}


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    try:
        model, threshold = load_model()
    except FileNotFoundError as e:
        print(f"\n[ERRO] {e}")
        raise SystemExit(1)

    exemplos = [
        (_EXEMPLO_BAIXO_RISCO, "Cliente_Baixo_Risco"),
        (_EXEMPLO_MEDIO_RISCO, "Cliente_Medio_Risco"),
        (_EXEMPLO_ALTO_RISCO,  "Cliente_Alto_Risco"),
    ]

    for dados, nome in exemplos:
        resultado = predict_default(dados, model, threshold)
        print_result(resultado, nome)
        plot_probability_distribution(resultado, nome)

    print("\n=== PREDIÇÃO EM LOTE (3 clientes) ===")
    lote = pd.DataFrame([d for d, _ in exemplos])
    lote.insert(0, "cliente_id", [n for _, n in exemplos])

    resultado_lote = predict_batch(lote.drop(columns=["cliente_id"]), model, threshold)
    resultado_lote.insert(0, "cliente_id", [n for _, n in exemplos])

    cols = ["cliente_id", "prediction", "probability_default", "score", "risk_level"]
    print(resultado_lote[cols].to_string(index=False))
    print("\n[OK] Inferência concluída — figuras salvas em figures/")
