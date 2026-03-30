import numpy as np
import pandas as pd
import os
import re
import pathlib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from carbontracker.tracker import CarbonTracker

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

dataset = pd.read_csv('../carbontracker/german_credit_data.csv') 

dataset = dataset.drop(columns=['Saving accounts', 'Checking account'])
dataset.drop(columns='Unnamed: 0', inplace=True)

categoricas = ['Sex', 'Housing', 'Purpose', 'Risk']
dataset = pd.get_dummies(dataset, columns=categoricas, drop_first=True)

bool_cols = dataset.select_dtypes(include='bool').columns
dataset[bool_cols] = dataset[bool_cols].astype(int)

y = dataset['Risk_good']
X = dataset.drop(columns='Risk_good')
colunas_originais = X.columns

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=colunas_originais)

# Cálculo de MI Score
mi_scores = mutual_info_classif(X, y)
mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=True)

def extrair_carbontracker_log(log_path):
    tempo = energia = emiss = None
    
    with open(log_path, "r") as f:
        linhas = f.readlines()
        
    for line in reversed(linhas):
        
        if "Time:" in line and tempo is None:
            match = re.search(r"Time:\s+(\d+):(\d+):(\d+)", line)
            if match:
                h, m, s = map(int, match.groups())
                tempo = h * 3600 + m * 60 + s

        
        if "Energy:" in line and energia is None:
            match = re.search(r"Energy:\s+([\d\.]+)", line)
            if match:
                energia = float(match.group(1))

        
        if "CO2eq:" in line and emiss is None:
            match = re.search(r"CO2eq:\s+([\d\.]+)", line)
            if match:
                emiss = float(match.group(1))
                
        if tempo is not None and energia is not None and emiss is not None:
            break

    return tempo, energia, emiss

def signed_log(series):
    arr = series.to_numpy(dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr))

def safe_log1p_nonneg(series):
    arr = series.to_numpy(dtype=float)
    arr = np.where(arr > 0, arr, 0.0)
    return np.log1p(arr)

def aumentar_features(df, percent_increase):
    df_new = df.copy()
    epsilon = 1e-6

    # ---------------------- NÍVEL 10% (+1 feature) ----------------------
    if percent_increase >= 10:
        df_new['feat_Credit_per_month'] = df_new['Credit amount'] / (df_new['Duration'] + epsilon)

    # ---------------------- NÍVEL 20% (+2 features total) ----------------------
    if percent_increase >= 20:
        df_new['feat_Age_young'] = (df_new['Age'] < 30).astype(int)

    # ---------------------- NÍVEL 30% (+4 features total) ----------------------
    if percent_increase >= 30:
        df_new['feat_Job_Credit_Interaction'] = df_new['Job'] * df_new['Credit amount']
        df_new['feat_High_Amount'] = (df_new['Credit amount'] > df_new['Credit amount'].median()).astype(int)

    # ---------------------- NÍVEL 40% (+6 features total) ----------------------
    if percent_increase >= 40:
        df_new['feat_Housing_Score'] = df_new['Housing_own'] * 2 + df_new['Housing_rent']
        df_new['feat_Duration_Age_Ratio'] = df_new['Duration'] / (df_new['Age'] + epsilon)

    # ---------------------- NÍVEL 50% (+7 features total) ----------------------
    if percent_increase >= 50:
        risk_cols = ['Purpose_education', 'Purpose_repairs', 'Purpose_vacation/others']
        df_new['feat_High_Risk_Purpose'] = df_new[risk_cols].sum(axis=1)

    df_new = df_new.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df_new

def run_experimento(modelo_nome, model, X_train, X_val, y_train, y_val, reducao_pct):
    print(f"\nTreinando modelo: {modelo_nome}...")

    log_dir = f"carbontracker_logs_german/{modelo_nome}_{int(reducao_pct*100)}_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)

    tracker = CarbonTracker(
        epochs=1,
        log_dir=log_dir,
        verbose=0
    )

    start = time.time()

    tracker.epoch_start()

    if modelo_nome == "RNA":
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
    else:
        model.fit(X_train, y_train)

    tracker.epoch_end()
    tracker.stop()

    tempo_real = time.time() - start
    time.sleep(2)
    log_files = [f for f in os.listdir(log_dir) if f.endswith("_output.log")]

    if not log_files:
        raise FileNotFoundError(f"ERRO CRÍTICO: Arquivo _output.log não gerado em {log_dir}. Verifique o CarbonTracker.")

    log_path = os.path.join(log_dir, log_files[0])
    print(f"✅ Lendo dados reais de: {log_path}")
    
    _, energia, emiss = extrair_carbontracker_log(log_path)

    if energia is None or emiss is None:
        raise RuntimeError(
            f"ERRO DE EXTRAÇÃO: O arquivo {log_path} existe, mas não contém a string 'Actual consumption'. "
            "Isso acontece se o treino for rápido demais para o CarbonTracker registrar."
        )

    if modelo_nome == "RNA":
        y_pred_proba = model.predict(X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

    # ===== MÉTRICAS =====
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    brier = brier_score_loss(y_val, y_pred_proba)
    auc = roc_auc_score(y_val, y_pred_proba)

    rga = emiss / acc

    print(
        f"{modelo_nome:12s} | "
        f"Red {int(reducao_pct*100):>2}% | "
        f"AUC {auc:.4f} | "
        f"Acc {acc:.4f} | "
        f"Tempo {tempo_real:.2f}s | "
        f"Energia {energia:.8f} kWh | "
        f"Emiss {emiss:.6f} g | "
        f"RGA {rga:.4f}"
    )

    return acc, prec, rec, f1, brier, auc, tempo_real, emiss, energia, rga, np.nan, np.nan


percentuais_reducao = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
resultados_reducao = []

print("="*60)
print("INICIANDO FASE DE REDUÇÃO DE FEATURES (CARBONTACKER - German Credit)")
print("="*60)

for p in percentuais_reducao:
    print(f"\n===== Redução de {int(p * 100)}% =====")

    n_remove = int(len(mi_ranking) * p)
    cols_reduzido = mi_ranking.index[n_remove:]
    X_reduz = X[cols_reduzido]

    print(f"Antes: {X.shape[1]} | Removidas: {n_remove} | Depois: {X_reduz.shape[1]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_reduz, y, test_size=0.3, random_state=10
    )

    modelos = {
        "RandomForest": RandomForestClassifier(n_estimators=30, random_state=10),
        "LightGBM": LGBMClassifier(n_estimators=30, random_state=10),
        "RNA": Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
            ])
        }

    for nome, modelo in modelos.items():
        if nome == "RNA":
            modelo.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        acc, prec, rec, f1, brier, auc, tempo, emiss, energia, rga, cpu, gpu = run_experimento(
            nome, modelo, X_train, X_val, y_train, y_val, p
        )

        resultados_reducao.append([
            nome,
                f"-{int(p * 100)}%",
                X_reduz.shape[1],
            acc, prec, rec, f1, brier, auc, tempo, emiss, energia, rga, cpu, gpu
        ])

# --- FASE 2: AUMENTO DE FEATURES ---

percentuais_aumento = [10, 20, 30, 40, 50]
resultados_aumento = []

X_original = X.copy()
y_original = y.copy()

print("\n" + "="*60)
print("INICIANDO FASE DE AUMENTO DE FEATURES (CARBONTACKER - German Credit)")
print("="*60)

for p_int in percentuais_aumento:
    print(f"\n--- Iniciando experimento para AUMENTO de {p_int}% ---")

    # 1. AUMENTA AS FEATURES
    X_aumentado = aumentar_features(X_original, p_int)

    print(f"Features Originais: {X_original.shape[1]}")
    n_adicionadas = X_aumentado.shape[1] - X_original.shape[1]
    print(f"Features Adicionadas: {n_adicionadas}")
    print(f"Features Totais: {X_aumentado.shape[1]}\n")

    # 2. Divide os dados
    X_train, X_val, y_train, y_val = train_test_split(
        X_aumentado, y_original, test_size=0.3, random_state=10 + p_int
    )

    print(f"Tamanho de entrada para RNA: {X_train.shape[1]} features")

    # 3. Define os modelos
    input_dim_rna = X_train.shape[1]
    modelos = {
        "RandomForest": RandomForestClassifier(n_estimators=30, random_state=10),
        "LightGBM": LGBMClassifier(n_estimators=30, random_state=10),
        "RNA": Sequential([
            Input(shape=(input_dim_rna,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    }

    # 4. Treina e avalia
    for nome, modelo in modelos.items():
        if nome == "RNA":
            modelo.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        acc, prec, rec, f1, brier, auc, tempo, emiss, energia, rga, cpu, gpu = run_experimento(
            nome, modelo, X_train, X_val, y_train, y_val, p_int
        )

        resultados_aumento.append([
            nome,
            f"+{p_int}%",
            X_aumentado.shape[1],
            acc, prec, rec, f1, brier, auc, tempo, emiss, energia, rga, cpu, gpu
        ])

print("\n--- Todos os experimentos de AUMENTO finalizados! ---")

colunas_finais = [
    "Modelo", "Alteracao", "N_Features",
    "Acuracia", "Precisao", "Recall", "F1-Score",
    "Brier", "AUC",
    "Tempo (s)", "Emissoes (g)", "Energia (kWh)", "RGA (kg/ac)",
    "CPU", "GPU"
]

# Salvar resultados de Redução
output_dir = "carbontracker[GERMAN]"
os.makedirs(output_dir, exist_ok=True)
output_path_reducao = os.path.join(output_dir, "[REDUCAO - CARBONTRACKER]german_credit_data.csv")

df_resultados_reducao = pd.DataFrame(resultados_reducao, columns=colunas_finais)
df_resultados_reducao.to_csv(output_path_reducao, sep=';', index=False, encoding='utf-8-sig')

# Salvar resultados de Aumento
output_path_aumento = os.path.join(output_dir, "[AUMENTO - CARBONTRACKER]german_credit_data.csv")

df_resultados_aumento = pd.DataFrame(resultados_aumento, columns=colunas_finais)
df_resultados_aumento.to_csv(output_path_aumento, sep=';', index=False, encoding='utf-8-sig')

print("\n✅ Arquivo de Redução salvo em:", output_path_reducao)
print("✅ Arquivo de Aumento salvo em:", output_path_aumento)