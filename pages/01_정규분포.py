import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Estimation of Population Mean (t-Test)", layout="centered")

st.title("Estimation of Population Mean & t Distribution Visualization")
st.write("""
Upload a sample data file (Excel or CSV).  
If the sample size is not very large, we perform population mean estimation using the t-test,  
show the estimated population mean, standard deviation, confidence interval, and visualize the estimated t distribution according to degrees of freedom (df).
""")

uploaded_file = st.file_uploader("Upload your sample data file (Excel or CSV)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='cp949')
    else:
        df = pd.read_excel(uploaded_file)
    
    st.dataframe(df)
    columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not columns:
        st.error("No numeric columns found in the uploaded file.")
    else:
        col = st.selectbox("Select a column for analysis", columns)
        data = df[col].dropna().values
        
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        dfree = n - 1  # 자유도
        
        alpha = st.slider("Confidence Level (1-α)", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
        conf_level = alpha
        alpha = 1 - conf_level
        
        # t분포의 임계값
        t_crit = stats.t.ppf(1 - alpha/2, df=dfree)
        margin_error = t_crit * (sample_std/np.sqrt(n))
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        st.subheader("Statistics Summary")
        st.write(f"Sample size n = {n}")
        st.write(f"Degrees of freedom (df) = {dfree}")
        st.write(f"Sample mean (estimated population mean): {sample_mean:.4f}")
        st.write(f"Sample standard deviation (estimated population std): {sample_std:.4f}")
        st.write(f"{int(conf_level*100)}% Confidence Interval (t): ({ci_lower:.4f} ~ {ci_upper:.4f})")
        
        st.subheader("Estimated Population t Distribution (df에 따른)")
        x = np.linspace(sample_mean - 4*sample_std, sample_mean + 4*sample_std, 200)
        # t분포 PDF (표본 평균을 중심, 표본표준편차, 자유도 사용)
        y = stats.t.pdf((x - sample_mean) / (sample_std/np.sqrt(n)), df=dfree) / (sample_std/np.sqrt(n))
        
        fig, ax = plt.subplots()
        ax.plot(x, y, label=f"t Distribution (df={dfree})")
        ax.axvline(sample_mean, color='r', linestyle='--', label='Estimated Population Mean')
        ax.fill_between(x, 0, y, where=(x >= ci_lower) & (x <= ci_upper), color='skyblue', alpha=0.5, label=f"{int(conf_level*100)}% Confidence Interval")
        ax.legend()
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        st.pyplot(fig)
