import os
import warnings
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, f1_score,
)

warnings.filterwarnings("ignore")

DATA_PATH    = "datasets/default_of_credit_card_clients.csv"
MODEL_PATH   = "models/best_model.pkl"
FIGURES_DIR  = "figures"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

TARGET_COL   = "default"
PAY_COLS     = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS    = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CAT_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
NUM_FEATURES = (
    ["LIMIT_BAL", "AGE"] + BILL_COLS + PAY_AMT_COLS + [
        "TOTAL_BILL", "TOTAL_PAY_AMT", "PAY_RATIO",
        "AVG_PAY_STATUS", "MONTHS_DELAYED",
        "MONTHS_NO_CONSUMPTION", "LIMIT_UTILIZATION",
    ]
)


# Carrega o dataset a partir do CSV com separador ponto-e-vírgula
def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=";")
    print(f"[OK] Dataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    return df


# Imprime estatísticas exploratórias essenciais do dataset
def run_eda(df: pd.DataFrame) -> None:
    raw_target = "default payment next month"

    missing = df.isnull().sum()
    print("\nValores ausentes:", "nenhum" if not missing.any() else missing[missing > 0])

    counts = df[raw_target].value_counts()
    print(f"\nDistribuição do alvo:")
    print(f"  Não inadimplente (0): {counts[0]:,} ({counts[0]/len(df):.1%})")
    print(f"  Inadimplente     (1): {counts[1]:,} ({counts[1]/len(df):.1%})")

    print("\nVariáveis categóricas:")
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        print(f"\n{col}:\n{df[col].str.strip().value_counts().to_string()}")

    print("\nColunas PAY — valores únicos por coluna:")
    for col in PAY_COLS:
        print(f"  {col}: {sorted(df[col].unique())}")


# Remove ID, renomeia o alvo e trata categorias não documentadas
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.rename(columns={"default payment next month": TARGET_COL})

    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        df[col] = df[col].str.strip()

    known_edu = {
        "Graduate School", "University", "Short-Cycle Tertiary Education",
        "High School", "Middle School", "Elementary School", "Others",
    }
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x in known_edu else "Others")

    known_mar = {"Married", "Single", "Divorced", "Widowed", "Others"}
    df["MARRIAGE"] = df["MARRIAGE"].apply(lambda x: x if x in known_mar else "Others")

    return df


# Cria features derivadas de pagamento, fatura e utilização de crédito
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TOTAL_BILL"]            = df[BILL_COLS].sum(axis=1)
    df["TOTAL_PAY_AMT"]         = df[PAY_AMT_COLS].sum(axis=1)
    df["PAY_RATIO"]             = np.where(df["TOTAL_BILL"] > 0, df["TOTAL_PAY_AMT"] / df["TOTAL_BILL"], 0.0)
    df["AVG_PAY_STATUS"]        = df[PAY_COLS].mean(axis=1)
    df["MONTHS_DELAYED"]        = (df[PAY_COLS] > 0).sum(axis=1)
    df["MONTHS_NO_CONSUMPTION"] = (df[PAY_COLS] == -2).sum(axis=1)
    df["LIMIT_UTILIZATION"]     = np.where(df["LIMIT_BAL"] > 0, df["BILL_AMT1"] / df["LIMIT_BAL"], 0.0)
    return df


# Monta o ColumnTransformer com OHE para categóricas e StandardScaler para numéricas
def build_preprocessor() -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CAT_FEATURES),
            ("pay", num_pipe, PAY_COLS),
            ("num", num_pipe, NUM_FEATURES),
        ],
        remainder="drop",
    )


# Retorna os modelos candidatos configurados para dataset desbalanceado
def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE,
        ),
    }

    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=4,
            eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
        print("[OK] XGBoost adicionado")
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            class_weight="balanced", subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        print("[OK] LightGBM adicionado")
    except ImportError:
        pass

    return models


# Treina todos os modelos com cross-validation estratificado e retorna as métricas
def train_and_compare(X_train, y_train, preprocessor, models) -> dict:
    print(f"\n{'='*60}")
    print(f"COMPARAÇÃO DE MODELOS ({CV_FOLDS}-fold Stratified CV)")
    print(f"{'='*60}")

    results = {}
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n  Treinando: {name} ...")
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])

        cv_res = cross_validate(
            pipe, X_train, y_train,
            cv=skf,
            scoring=["roc_auc", "f1", "precision", "recall"],
            n_jobs=1,
            return_train_score=False,
        )

        results[name] = {
            "pipeline":       pipe,
            "roc_auc_mean":   cv_res["test_roc_auc"].mean(),
            "roc_auc_std":    cv_res["test_roc_auc"].std(),
            "f1_mean":        cv_res["test_f1"].mean(),
            "f1_std":         cv_res["test_f1"].std(),
            "precision_mean": cv_res["test_precision"].mean(),
            "recall_mean":    cv_res["test_recall"].mean(),
        }

        r = results[name]
        print(f"    ROC-AUC:   {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}")
        print(f"    F1:        {r['f1_mean']:.4f} ± {r['f1_std']:.4f}")
        print(f"    Precision: {r['precision_mean']:.4f}  |  Recall: {r['recall_mean']:.4f}")

    return results


# Seleciona o modelo com maior ROC-AUC médio e imprime o ranking completo
def select_best_model(results: dict):
    best_name = max(results, key=lambda k: results[k]["roc_auc_mean"])

    print(f"\n{'='*60}")
    print("RANKING DE MODELOS (por ROC-AUC médio)")
    print(f"{'='*60}")
    for rank, (name, r) in enumerate(
        sorted(results.items(), key=lambda x: x[1]["roc_auc_mean"], reverse=True), 1
    ):
        marker = " ← MELHOR" if name == best_name else ""
        print(f"  {rank}. {name:30s} AUC={r['roc_auc_mean']:.4f}{marker}")

    return best_name, results[best_name]["pipeline"]


# Encontra o limiar que maximiza F1 — necessário por causa do desbalanceamento de classes
def find_optimal_threshold(y_true, y_prob) -> tuple:
    thresholds = np.linspace(0.01, 0.99, 300)
    f1_scores  = [f1_score(y_true, y_prob >= t, zero_division=0) for t in thresholds]

    best_idx = int(np.argmax(f1_scores))
    best_t   = float(thresholds[best_idx])
    best_f1  = float(f1_scores[best_idx])

    print(f"\n[LIMIAR] Ótimo: {best_t:.3f} (F1={best_f1:.4f})  |  Padrão 0.5: F1={f1_score(y_true, y_prob >= 0.5, zero_division=0):.4f}")
    return best_t, best_f1


# Treina o modelo final, calcula o limiar ótimo e imprime métricas no conjunto de teste
def evaluate_final_model(pipeline, X_train, y_train, X_test, y_test, model_name) -> tuple:
    print(f"\n{'='*60}")
    print(f"AVALIAÇÃO FINAL — {model_name}")
    print(f"{'='*60}")

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    threshold, _ = find_optimal_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\nROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\nRelatório de Classificação (limiar={threshold:.3f}):")
    print(classification_report(y_test, y_pred, target_names=["Não Inadimplente", "Inadimplente"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(pd.DataFrame(
        cm,
        index=["Real: Não Inad.", "Real: Inadimplente"],
        columns=["Pred: Não Inad.", "Pred: Inadimplente"],
    ))

    return y_pred, y_prob, threshold


# Salva o pipeline treinado e o limiar ótimo juntos em um único arquivo
def save_model(pipeline, threshold: float, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": pipeline, "threshold": threshold}, path)
    print(f"\n[OK] Modelo salvo em: {path}")


def main() -> None:
    print("=" * 60)
    print("BANCO AGORAVAI — PREDIÇÃO DE INADIMPLÊNCIA")
    print("=" * 60)

    df = load_data()
    run_eda(df)
    df = clean_data(df)
    df = create_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"\n[INFO] Treino: {len(X_train):,}  |  Teste: {len(X_test):,}  |  Positivos: {y.mean():.1%}")

    preprocessor = build_preprocessor()
    models       = get_models()
    results      = train_and_compare(X_train, y_train, preprocessor, models)

    best_name, best_pipeline = select_best_model(results)
    _, _, threshold = evaluate_final_model(best_pipeline, X_train, y_train, X_test, y_test, best_name)

    save_model(best_pipeline, threshold)

    print("\n" + "=" * 60)
    print("CONCLUÍDO")
    print("=" * 60)
    print(f"  Melhor modelo : {best_name}")
    print(f"  Limiar ótimo  : {threshold:.3f}")
    print(f"  Modelo salvo  : {MODEL_PATH}")
    print("\nPróximo passo: python inference.py")


if __name__ == "__main__":
    main()
