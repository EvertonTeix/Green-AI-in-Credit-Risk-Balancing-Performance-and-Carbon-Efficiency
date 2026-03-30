import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import time
from carbontracker.tracker import CarbonTracker
import os
import re

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from codecarbon import EmissionsTracker
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_column",100)
pd.set_option("display.max_rows",100)

data = pd.read_csv('/home/enginelab/Everton /XYZ_CorpLending/XYZCorp_LendingData.txt',
                      index_col=0, delimiter='\t',parse_dates=['issue_d'])

cols_to_drop = [
    'collection_recovery_fee',
    'recoveries',
    'total_pymnt',
    'total_pymnt_inv',
    'total_rec_prncp',
    'total_rec_int',
    'total_rec_late_fee',
    'last_pymnt_amnt',
    'last_pymnt_d',
    'next_pymnt_d',
    'out_prncp',
    'out_prncp_inv',
    'member_id',
    'desc',
    'emp_title',
    'title',
    'annual_inc_joint',
    'dti_joint'
]

data = data.drop(cols_to_drop, axis=1)

def missing_data(data):
    total=data.isnull().sum().sort_values(ascending=False)
    percent=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data=pd.concat([total,percent],axis=1,keys=['total','percent'])
    return(missing_data.head(20))

missing_data(data)

del data['pymnt_plan']

for x in data.columns[:]:
    if data[x].dtype=='str':
        data[x].fillna(data[x].mode()[0],inplace=True) #preenche com a moda
    elif data[x].dtype=='int64' or data[x].dtype=='float64':
        data[x].fillna(data[x].mean(),inplace=True) #preenche com a média

colname=[]
for x in data.columns:
    if data[x].dtype=='float64':
        colname.append(x)


#APLICAÇÃO DO LABEL ENCODER

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

for x in colname:
    data[x]=le.fit_transform(data[x])


del data['issue_d']
del data['zip_code']
data['initial_list_status'] = data['initial_list_status'].map({'f': 0, 'w': 1})
data['initial_list_status'] = data['initial_list_status'].astype(int)

colname=[]
for x in data.columns:
    if data[x].dtype=='str':
        colname.append(x)


from sklearn import preprocessing

le=preprocessing.LabelEncoder()

for x in colname:
    data[x]=le.fit_transform(data[x])

data=data.drop(['addr_state', 'earliest_cr_line','last_credit_pull_d', 'policy_code'],axis=1)
data=data.drop(['application_type'],axis=1)

data=data.drop(['funded_amnt_inv','funded_amnt','delinq_2yrs','collections_12_mths_ex_med','acc_now_delinq',
                        'tot_coll_amt'],axis=1)

X = data.drop('default_ind', axis=1)
y = data['default_ind']

approved = data[data['default_ind']==1]

reject = data[data['default_ind']==0]

from imblearn.over_sampling import SMOTE

# Usando SMOTE puro em vez de SMOTETomek para otimizar o tempo de execução
smk = SMOTE(random_state=42)

X_o_res, Y_o_res = smk.fit_resample(X, y)

colunas_originais = X.columns

scaler = StandardScaler()
scaler.fit(X_o_res)
X_o_res=scaler.transform(X_o_res)

X_o_res = pd.DataFrame(X_o_res, columns=colunas_originais)
print(X_o_res)

# Cálculo do MI Score
mi_scores = mutual_info_classif(X_o_res, Y_o_res)

# Ranking crescente (menor MI = menos informativa)
mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=True)
print("MI Score finalizado")

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

print("Todos os experimentos de redução finalizados finalizados!\n")


df_resultados = pd.DataFrame(
    resultados,
    columns=[
        "Modelo",
        "Redução",
        "N_Features",
        "Acurácia",
        "Precisão",
        "Recall",
        "F1-Score",
        "Brier",
        "AUC",
        "Tempo (s)",
        "Energia (kWh)",
        "Emissoes (g)",
        "RGA",
        "CPU",
        "GPU"
    ]
)

df_resultados.to_csv(
    "/home/enginelab/Everton /XYZ_CorpLending/Carbontracker/[REDUCAO - CARBONTRACKER]resultados_experimentos[XYZ_Corp].csv",
    index=False,
    sep=';',
    encoding='utf-8-sig'
)

print("Iniciando Aumento das Features")

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

    print(f"\n--- Aumento de {percent_increase}% ---")

    # =========================
    # 🔹 NÍVEL 10% (Razões básicas de risco)
    # =========================
    if percent_increase >= 10:

        df_new['feat_loan_vs_income'] = df_new['loan_amnt'] / (df_new['annual_inc'] + epsilon)
        df_new['feat_installment_vs_income'] = df_new['installment'] / (df_new['annual_inc'] + epsilon)
        df_new['feat_revol_bal_vs_income'] = df_new['revol_bal'] / (df_new['annual_inc'] + epsilon)
        df_new['feat_totalacc_ratio'] = df_new['open_acc'] / (df_new['total_acc'] + epsilon)


    # =========================
    # 🔹 NÍVEL 20% (Endividamento + crédito)
    # =========================
    if percent_increase >= 20:

        df_new['feat_credit_pressure'] = df_new['dti'] * df_new['revol_util']
        df_new['feat_inq_intensity'] = df_new['inq_last_6mths'] + df_new['inq_last_12m']
        df_new['feat_delinq_risk'] = df_new['mths_since_last_delinq'] / (df_new['total_acc'] + epsilon)
        df_new['feat_pubrec_ratio'] = df_new['pub_rec'] / (df_new['total_acc'] + epsilon)


    # =========================
    # 🔹 NÍVEL 30% (Logs financeiros)
    # =========================
    if percent_increase >= 30:

        df_new['feat_log_loan'] = safe_log1p_nonneg(df_new['loan_amnt'])
        df_new['feat_log_income'] = safe_log1p_nonneg(df_new['annual_inc'])
        df_new['feat_log_revol_bal'] = safe_log1p_nonneg(df_new['revol_bal'])
        df_new['feat_log_tot_cur_bal'] = safe_log1p_nonneg(df_new['tot_cur_bal'])


    # =========================
    # 🔹 NÍVEL 40% (Interações fortes de risco)
    # =========================
    if percent_increase >= 40:

        df_new['feat_dti_x_loan'] = df_new['dti'] * df_new['loan_amnt']
        df_new['feat_int_x_term'] = df_new['int_rate'] * df_new['term']
        df_new['feat_income_x_grade'] = df_new['annual_inc'] * df_new['grade']
        df_new['feat_revol_x_util'] = df_new['revol_bal'] * df_new['revol_util']


    # =========================
    # 🔹 NÍVEL 50% (Polinômios e pressão sistêmica)
    # =========================
    if percent_increase >= 50:

        df_new['feat_dti_sq'] = df_new['dti'] ** 2
        df_new['feat_int_sq'] = df_new['int_rate'] ** 2
        df_new['feat_loan_sq'] = df_new['loan_amnt'] ** 2
        df_new['feat_income_minus_debtproxy'] = df_new['annual_inc'] - df_new['loan_amnt']


    # =========================
    # 🔹 Limpeza final
    # =========================
    df_new = df_new.replace([np.inf, -np.inf], np.nan)
    df_new = df_new.fillna(0)

    print("Total de features:", df_new.shape[1])

    return df_new

percentuais_aumento = [0.1, 0.2, 0.3, 0.4, 0.5]
resultados_aumento = []

X_original = X_o_res.copy()
y_original = Y_o_res.copy()

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

        acc, prec, rec, f1, brier, auc, tempo, em, energia, rga, cpu, gpu = run_experimento(
            nome, modelo, X_train, X_val, y_train, y_val, p)

        resultados_aumento.append([
            nome,
            f"+{nivel_percentual}%",
            X_aumentado.shape[1],
            acc, prec, rec, f1, brier, auc, tempo, energia, em, rga
        ])

print("\n--- Todos os experimentos de AUMENTO finalizados! ---")

colunas = ["Modelo", "Alteração", "N_Features", "Accuracy", "Precision", "Recall",
            "F1", "Brier", "AUC", "Tempo", "Energia", "EM", "RGA"]

resultados_aumento_df = pd.DataFrame(resultados_aumento, columns=colunas)


df_resultados = pd.DataFrame(
    resultados_aumento,
    columns=[
        "Modelo",
        "Redução",
        "N_Features",
        "Acurácia",
        "Precisão",
        "Recall",
        "F1-Score",
        "Brier",
        "AUC",
        "Tempo (s)",
        "Energia (kWh)",
        "Emissoes (g)",
        "RGA"
    ]
)


caminho_saida = "/home/enginelab/Everton /XYZ_CorpLending/Carbontracker/[AUMENTO - CARBONTRACKER]resultados_experimentos[XYZ_Corp].csv"
df_resultados.to_csv(
    caminho_saida,
    index=False,
    sep=';',
    encoding='utf-8-sig'
)

print(f"Resultados salvos em: {caminho_saida}")


