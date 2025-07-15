import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="t-test and Normal Distribution Visualization", layout="centered")

st.title("t-test & Normal Distribution Visualization Site")
st.write("""
Upload a sample data file (Excel or CSV). The app will estimate the population mean and standard deviation by t-test, calculate the confidence interval, and draw the estimated normal distribution graph.
""")

uploaded_file = st.file_uploader("Upload a sample data file (Excel or CSV)", type=['csv', 'xlsx', 'xls'])

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
        col = st.selectbox("Select the column to analyze", columns)
        data = df[col].dropna().values
        
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        alpha = st.slider("Confidence Level (1-α)", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
        conf_level = alpha
        alpha = 1 - conf_level
        
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        margin_error = t_crit * (sample_std/np.sqrt(n))
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        st.subheader("Statistics Summary")
        st.write(f"Sample size n = {n}")
        st.write(f"Sample mean (Estimated population mean): {sample_mean:.4f}")
        st.write(f"Sample std (Estimated population std): {sample_std:.4f}")
        st.write(f"{int(conf_level*100)}% Confidence Interval: ({ci_lower:.4f} ~ {ci_upper:.4f})")
        
        st.subheader("Estimated Population Normal Distribution")
        # *** 표준화 없이 원본 정규분포 ***
        x = np.linspace(sample_mean - 4*sample_std, sample_mean + 4*sample_std, 200)
        y = stats.norm.pdf(x, loc=sample_mean, scale=sample_std)
        
        fig, ax = plt.subplots()
        ax.plot(x, y, label="Estimated Normal Distribution")
        ax.axvline(sample_mean, color='r', linestyle='--', label='Estimated Population Mean')
        ax.fill_between(x, 0, y, where=(x >= ci_lower) & (x <= ci_upper), color='skyblue', alpha=0.5, label=f"{int(conf_level*100)}% Confidence Interval")
        ax.legend()
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        st.pyplot(fig)
