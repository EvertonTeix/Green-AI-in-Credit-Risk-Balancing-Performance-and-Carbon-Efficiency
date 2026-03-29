# 🌱 Green AI in Credit Risk: Carbon-Aware Machine Learning

This repository contains the implementation and experimental framework for the study:

**"A Comparative Analysis of Carbon Monitoring Frameworks Under Variable Feature Dimensionality in Machine Learning Models"**

## 📌 Overview

The rapid growth of Artificial Intelligence has increased computational demand and, consequently, energy consumption. This project investigates how **feature engineering and feature selection** impact both:

- 📊 Predictive performance  
- 🌍 Environmental impact (carbon emissions)

The experiments focus on **credit risk prediction**, a domain where models are widely used and computational efficiency is critical.

---

## 🎯 Objectives

- Analyze the trade-off between **model performance** and **carbon footprint**
- Evaluate how **feature dimensionality (±10% to 50%)** affects:
  - Accuracy
  - AUC
  - Energy consumption
  - CO₂ emissions
- Compare two carbon tracking frameworks:
  - **CodeCarbon**
  - **CarbonTracker**

---

## 🧪 Experimental Design

Each dataset was evaluated under **11 scenarios**:

- ✅ Baseline (original dataset)
- 🔽 Feature reduction: -10%, -20%, -30%, -40%, -50%
- 🔼 Feature augmentation: +10%, +20%, +30%, +40%, +50%

### 🤖 Models Used

- Random Forest  
- LightGBM  
- Artificial Neural Networks (ANN)

---

## 📊 Datasets

The experiments were conducted using three real-world credit risk datasets:

- **German Credit Dataset**  
  - 1,000 instances  
  - Binary classification (good/bad credit)

- **Bondora Peer-to-Peer Lending**  
  - 179,235 instances  
  - Financial + demographic features

- **XYZ Corp Lending Dataset**  
  - 855,969 instances  
  - Large-scale lending data

---

## ⚙️ Methodology

### 🔹 Preprocessing
- Missing value imputation (mean/mode)
- One-Hot Encoding for categorical variables

### 🔹 Feature Engineering
- Creation of synthetic features (ratios, aggregations)

### 🔹 Feature Selection
- Mutual Information (MI Score)
- Ranking and removal of least relevant features

---

## 🌍 Carbon Monitoring

Two frameworks were used:

- **CodeCarbon**
  - Uses regional carbon intensity estimates

- **CarbonTracker**
  - Uses real-time energy tracking

Both estimate emissions based on:

`Emissions = Energy Consumption × Carbon Intensity`


---

## 📏 Evaluation Metrics

### Predictive Performance
- Accuracy  
- AUC (Area Under ROC Curve)  
- Precision, Recall, F1-score  

### Environmental Metrics
- CO₂ emissions (g)  
- Energy consumption (kWh)  
- Execution time (s)  

### ⚡ Main Metric: RGA

`RGA = Emissions (g) / Accuracy`


- Lower RGA = better efficiency (high performance + low emissions)

---

## 🧠 Key Findings

- ✅ **LightGBM** is the most energy-efficient model across all datasets  
- ❌ **Neural Networks** showed the highest carbon cost with limited gains  
- 🔽 Reducing features by up to **50% significantly lowers emissions**
- ⚖️ Feature dimensionality directly impacts sustainability-performance trade-offs  
- 📈 Larger datasets amplify environmental costs, making optimization essential  

---

## 🖥️ Experimental Environment

- Google Colab (CPU) → German Credit & Bondora  
- Local Machine:
  - Intel Xeon W-1350  
  - 16GB RAM  
  - NVIDIA RTX A4500  
  - Ubuntu Linux  
  → XYZ Dataset  

---
## 👨‍💻 Author

**Everton Teixeira** 🎓 Computer Science  
📊 Interests: Machine Learning, Data Science, Green AI  

---

## 📬 Contact

- Email: tteverton2024@gmail.com

---

## ⭐ Highlight

> 💡 This project explores a critical and often overlooked contemporary issue:  
> **The invisible environmental cost of Artificial Intelligence.**
