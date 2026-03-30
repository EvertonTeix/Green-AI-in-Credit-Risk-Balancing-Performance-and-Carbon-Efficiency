import numpy as np
import pandas as pd
import time
import re
import seaborn as sns
import matplotlib.pyplot as plt
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
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

dataset = pd.read_csv('[BONDORA]preprocessed_database.csv')

y = dataset['Status']
X = dataset.drop('Status', axis=1)
colunas_originais = X.columns

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=colunas_originais)

# MI Score

mi_scores = mutual_info_classif(X, y)
mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=True)

percentuais = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
resultados = []

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

def run_experimento(modelo_nome, model, X_train, X_val, y_train, y_val, reducao_pct):
    print(f"\nTreinando modelo: {modelo_nome}...")

    log_dir = f"carbontracker_logs/{modelo_nome}_{int(reducao_pct*100)}_{int(time.time())}"
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


for p in percentuais:
    print(f"\n===== Redução de {int(p * 100)}% =====")

    n_remove = int(len(mi_ranking) * p)
    cols_reduzido = mi_ranking.index[n_remove:]
    X_reduz = X[cols_reduzido]

    print(f"Antes: {X.shape[1]} | Removidas: {n_remove} | Depois: {X_reduz.shape[1]}")


    X_train, X_val, y_train, y_val = train_test_split(X_reduz, y, test_size=0.3, random_state=10)

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

        acc, prec, rec, f1, brier, auc, tempo, emissions, energia, rga, cpu, gpu = run_experimento(
            nome, modelo, X_train, X_val, y_train, y_val, p
        )

        resultados.append([
            nome,
            f"-{int(p * 100)}%",
            X_reduz.shape[1],
            acc, prec, rec, f1, brier, auc, tempo, emissions, energia, rga, cpu, gpu
        ])


output_dir = "carbontracker[BONDORA]"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "[REDUCAO - DATA]bondora.csv")

df_resultados = pd.DataFrame(
    resultados,
    columns=[
        "Modelo", "Reducao", "N_Features",
        "Acuracia", "Precisao", "Recall", "F1-Score",
        "Brier", "AUC",
        "Tempo (s)", "Emissoes (g)", "Energia (kWh)", "RGA",
        "CPU", "GPU"
    ]
)

df_resultados.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')
print("\n✅ Arquivo salvo em:", output_path)

print("Iniciando a fase de aumento das features:")

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

    print(f"\n--- Processando Nível: {percent_increase}% ---")

    # Nível 10%
    if percent_increase >= 10:
        print("Adicionando 11 features do Nível 10%...")
        df_new['feat_Amount_vs_Income'] = df_new['AppliedAmount'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_Amount_vs_FreeCash'] = df_new['AppliedAmount'] / (df_new['FreeCash'] + epsilon)
        df_new['feat_Liabilities_vs_FreeCash'] = df_new['LiabilitiesTotal'] / (df_new['FreeCash'] + epsilon)
        df_new['feat_Liabilities_vs_Income'] = df_new['LiabilitiesTotal'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_EstimatedMonthlyPayment'] = df_new['AppliedAmount'] / (df_new['LoanDuration'] + epsilon)
        df_new['feat_Payment_vs_FreeCash_Ratio'] = df_new['feat_EstimatedMonthlyPayment'] / (df_new['FreeCash'] + epsilon)
        df_new['feat_Payment_vs_Income_Ratio'] = df_new['feat_EstimatedMonthlyPayment'] / (df_new['IncomeTotal'] + epsilon)
        for col in ['BidsPortfolioManager', 'BidsApi', 'BidsManual']:
            if col not in df_new.columns:
                df_new[col] = 0
        df_new['feat_TotalBids'] = df_new['BidsPortfolioManager'].fillna(0) + df_new['BidsApi'].fillna(0) + df_new['BidsManual'].fillna(0)
        df_new['feat_Api_Bids_Ratio'] = df_new['BidsApi'] / (df_new['feat_TotalBids'] + epsilon)
        df_new['feat_Duration_vs_Age'] = df_new['LoanDuration'] / (df_new['Age'] + epsilon)
        df_new['feat_EarlyRepayment_Rate'] = df_new['PreviousEarlyRepaymentsCountBeforeLoan'] / (df_new['NoOfPreviousLoansBeforeLoan'] + epsilon)

    # Nível 20%
    if percent_increase >= 20:
        print("Adicionando 10 features do Nível 20%...")
        income_cols = ['IncomeFromPension', 'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
                       'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther']
        for c in income_cols:
            if c not in df_new.columns:
                df_new[c] = 0

        df_new['feat_OtherIncome_Total'] = (df_new['IncomeFromPension'].fillna(0) + df_new['IncomeFromFamilyAllowance'].fillna(0) +
                                           df_new['IncomeFromSocialWelfare'].fillna(0) + df_new['IncomeFromLeavePay'].fillna(0) +
                                           df_new['IncomeFromChildSupport'].fillna(0) + df_new['IncomeOther'].fillna(0))
        df_new['feat_OtherIncome_Ratio'] = df_new['feat_OtherIncome_Total'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_PrincipalIncome_Ratio'] = df_new['IncomeFromPrincipalEmployer'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_Pension_Ratio'] = df_new['IncomeFromPension'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_FamilyAllowance_Ratio'] = df_new['IncomeFromFamilyAllowance'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_SocialWelfare_Ratio'] = df_new['IncomeFromSocialWelfare'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_Manual_Bids_Ratio'] = df_new['BidsManual'] / (df_new['feat_TotalBids'] + epsilon)
        df_new['feat_Portfolio_Bids_Ratio'] = df_new['BidsPortfolioManager'] / (df_new['feat_TotalBids'] + epsilon)
        for c in ['RefinanceLiabilities', 'LiabilitiesTotal', 'AppliedAmount']:
            if c not in df_new.columns:
                df_new[c] = 0
        df_new['feat_Refinance_vs_Liabilities'] = df_new['RefinanceLiabilities'] / (df_new['LiabilitiesTotal'] + epsilon)
        df_new['feat_Refinance_vs_Amount'] = df_new['RefinanceLiabilities'] / (df_new['AppliedAmount'] + epsilon)

    if percent_increase >= 30:
        print("Adicionando 10 features do Nível 30% (log seguros)...")
        nonneg_cols = ['AppliedAmount', 'IncomeTotal', 'LiabilitiesTotal', 'FreeCash', 'Age',
                       'LoanDuration', 'Interest', 'ExistingLiabilities', 'NoOfPreviousLoansBeforeLoan', 'feat_TotalBids']
        for c in nonneg_cols:
            if c not in df_new.columns:
                df_new[c] = 0
        df_new['feat_Log_AppliedAmount'] = safe_log1p_nonneg(df_new['AppliedAmount'])
        df_new['feat_Log_IncomeTotal'] = safe_log1p_nonneg(df_new['IncomeTotal'])
        df_new['feat_Log_LiabilitiesTotal'] = safe_log1p_nonneg(df_new['LiabilitiesTotal'])
        df_new['feat_Log_FreeCash'] = signed_log(df_new['FreeCash'])
        df_new['feat_Log_Age'] = safe_log1p_nonneg(df_new['Age'])
        df_new['feat_Log_LoanDuration'] = safe_log1p_nonneg(df_new['LoanDuration'])
        df_new['feat_Log_Interest'] = signed_log(df_new['Interest']) 
        df_new['feat_Log_ExistingLiabilities'] = safe_log1p_nonneg(df_new['ExistingLiabilities'])
        df_new['feat_Log_NoOfPreviousLoans'] = safe_log1p_nonneg(df_new['NoOfPreviousLoansBeforeLoan'])
        df_new['feat_Log_TotalBids'] = safe_log1p_nonneg(df_new['feat_TotalBids'])

    # Nível 40%:
    if percent_increase >= 40:
        print("Adicionando 11 features do Nível 40%...")
        df_new['feat_Age_sq'] = df_new['Age']**2
        df_new['feat_Interest_sq'] = df_new['Interest']**2
        df_new['feat_LoanDuration_sq'] = df_new['LoanDuration']**2
        df_new['feat_Log_IncomeTotal_sq'] = df_new.get('feat_Log_IncomeTotal', 0)**2
        df_new['feat_Log_AppliedAmount_sq'] = df_new.get('feat_Log_AppliedAmount', 0)**2
        df_new['feat_Age_x_Interest'] = df_new['Age'] * df_new['Interest']
        df_new['feat_Age_x_Amount'] = df_new['Age'] * df_new['AppliedAmount']
        df_new['feat_Age_x_Income'] = df_new['Age'] * df_new['IncomeTotal']
        df_new['feat_Interest_x_Amount'] = df_new['Interest'] * df_new['AppliedAmount']
        df_new['feat_Interest_x_Duration'] = df_new['Interest'] * df_new['LoanDuration']
        df_new['feat_Interest_x_Income'] = df_new['Interest'] * df_new['IncomeTotal']

    # Nível 50%:
    if percent_increase >= 50:
        print("Adicionando 10 features do Nível 50%...")
        df_new['feat_Amount_x_Duration'] = df_new['AppliedAmount'] * df_new['LoanDuration']
        df_new['feat_DebtToIncome_x_Age'] = df_new['DebtToIncome'] * df_new['Age']
        df_new['feat_DebtToIncome_x_Interest'] = df_new['DebtToIncome'] * df_new['Interest']
        for c in ['ExistingLiabilities', 'AppliedAmount', 'IncomeTotal', 'FreeCash']:
            if c not in df_new.columns:
                df_new[c] = 0
        df_new['feat_Amount_vs_ExistingLiabilities'] = df_new['AppliedAmount'] / (df_new['ExistingLiabilities'] + epsilon)
        df_new['feat_Income_vs_ExistingLiabilities'] = df_new['IncomeTotal'] / (df_new['ExistingLiabilities'] + epsilon)
        df_new['feat_FreeCash_vs_ExistingLiabilities'] = df_new['FreeCash'] / (df_new['ExistingLiabilities'] + epsilon)
        df_new['feat_Income_minus_Liabilities'] = df_new['IncomeTotal'] - df_new['LiabilitiesTotal']
        df_new['feat_Liabilities_minus_Refinance'] = df_new['LiabilitiesTotal'] - df_new['RefinanceLiabilities'].fillna(0)
        df_new['feat_Age_cubed'] = df_new['Age']**3
        df_new['feat_Interest_cubed'] = df_new['Interest']**3

    df_new = df_new.replace([np.inf, -np.inf], np.nan)
    nan_count = df_new.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaNs encontrados — preenchendo com 0.")
    df_new = df_new.fillna(0)

    print(f"Número final de features: {df_new.shape[1]}")
    return df_new

percentuais_aumento = [0.1, 0.2, 0.3, 0.4, 0.5]
resultados_aumento = []

X_original = X.copy()
y_original = y.copy()

for p in percentuais_aumento:
    nivel_percentual = int(p * 100)
    print(f"\n--- Iniciando experimento para AUMENTO de {nivel_percentual}% ---")

    X_aumentado = aumentar_features(X_original, nivel_percentual)

    print(f"Features Originais: {X_original.shape[1]}")
    n_adicionadas = X_aumentado.shape[1] - X_original.shape[1]
    print(f"Features Adicionadas: {n_adicionadas}")
    print(f"Features Totais: {X_aumentado.shape[1]}\n")

    X_train, X_val, y_train, y_val = train_test_split(
        X_aumentado, y_original, test_size=0.3, random_state=10 + nivel_percentual
    )

    print(f"Tamanho de entrada para RNA: {X_train.shape[1]} features")

    modelos = {
        "RandomForest": RandomForestClassifier(n_estimators=30, random_state=10),
        "LightGBM": LGBMClassifier(n_estimators=30, random_state=10),
        "RNA": Sequential([
            Dense(64, input_dim=X_train.shape[1], activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    }

    for nome, modelo in modelos.items():
        if nome == "RNA":
            modelo.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        acc, prec, rec, f1, brier, auc, tempo, emissions, energia, rga, cpu, gpu = run_experimento(
            nome, modelo, X_train, X_val, y_train, y_val, p
        )

        resultados_aumento.append([
            nome,
            f"+{nivel_percentual}%",
            X_aumentado.shape[1],
            acc, prec, rec, f1, brier, auc, tempo, emissions, energia, rga, cpu, gpu
        ])

print("\n--- Todos os experimentos de AUMENTO finalizados! ---")

colunas = ["Modelo", "Alteração", "N_Features", "Accuracy", "Precision", "Recall",
            "F1", "Brier", "AUC", "Tempo", "Emissões", "Energia", "RGA", "CPU", "GPU"]

resultados_aumento_df = pd.DataFrame(resultados_aumento, columns=colunas)

output_dir = "carbontracker[BONDORA]"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "[AUMENTO - CARBONTRACKER]bondora.csv")

df_resultados_aumento = pd.DataFrame(
    resultados_aumento,
    columns=[
        "Modelo",
        "Aumento",
        "N_Features",
        "Acurácia",
        "Precisão",
        "Recall",
        "F1-Score",
        "Brier",
        "AUC",
        "Tempo (s)",
        "Emissoes (g)",
        "Energia (kWh)",
        "RGA",
        "CPU",
        "GPU"
    ]
)

df_resultados_aumento.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')
print("\n✅ Arquivo salvo em:", output_path)
