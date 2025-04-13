import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import streamlit as st

title = "This is title"

st.set_page_config(    
    page_title=f"{title}",
    page_icon="⭕",
    layout="wide"
)

# 设置标题
st.markdown(f'''
    <h1 style="font-size: 20px; text-align: center; color: black; border-bottom: 2px solid black; margin-bottom: 1rem;">
    {title}
    </h1>''', unsafe_allow_html=True)

# 定义使用的特征变量
var = [
    "Weight", "Charlson_Comorbidity_Index", "SOFA", "Heart_Rate",
    "Resp_Rate", "Lactate", "Hematocrit", "Calcium", "Potassium", 
    "WBC", "Albumin"
]

# 初始值
dinput = [float(i) for i in "78.4	10	6	137	33	0.9	29	0.92	3.4	5.9	2.9".split("\t")]

d = {}
col = st.columns(4)
# 输入
k = 0
for i, j in zip(var, dinput):
    d[i] = col[k%4].number_input(i, value=j)
    k = k+1
    
# 输入值
df = pd.DataFrame([d])

# 导入模型
m1 = joblib.load("preprocessor.pkl")
m2 = joblib.load("tabnet_model.pkl")

# 预处理输入数据
d = m1.transform(df)
st.dataframe(df, hide_index=True, use_container_width=True)

st.write(m2.predict(d))
