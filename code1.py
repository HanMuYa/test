import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# 定义使用的特征变量
var = [
    "Weight", "Charlson_Comorbidity_Index", "SOFA", "Heart_Rate",
    "Resp_Rate", "Lactate", "Hematocrit", "Calcium", "Potassium", "WBC", "Albumin"
]

d = [float(i) for i in "78.4	10	6	137	33	0.9	29	0.92	3.4	5.9	2.9".split("\t")]

df = pd.DataFrame([d], columns=var)

m1 = joblib.load("preprocessor.pkl")
m2 = joblib.load("tabnet_model.pkl")

d = m1.transform(df)
st.write(d)

st.write(m2.predict(d))
