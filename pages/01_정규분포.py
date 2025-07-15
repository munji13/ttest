import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Streamlit page configuration
st.set_page_config(page_title="Estimation of Population Mean (Large Sample)", layout="centered")

st.title("Estimation of Population Mean & Normal Distribution Visualization (Large Sample)")
st.write("""
Upload a sample data file (Excel or CSV).  
If the sample size is sufficiently large, we perform population mean estimation (Z-interval),  
show the estimated population mean, standard deviation, confidence interval, and visualize the estimated normal distribution.
""")

# File uploader widget
uploaded_file = st.file_uploader("Upload your sample data file (Excel or CSV)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    # Handle encoding for CSVs (prevent Korean character errors)
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='cp949')
    else:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
    
    # Display the dataframe
    st.dataframe(df)
    
    # Select only numeric columns
    columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not columns:
        st.error("No numeric columns found in the uploaded file.")
    else:
        # Column selection for analysis
        col = st.selectbox("Select a column for analysis", columns)
        data = df[col].dropna().values
        
        n = len(data)  # Sample size
        sample_mean = np.mean(data)  # Sample mean
        sample_std = np.std(data, ddof=1)  # Sample standard deviation
        alpha = st.slider("Confidence Level (1-α)", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
        conf_level = alpha
        alpha = 1 - conf_level
        
        # Use Z-distribution for large samples
        z_crit = stats.norm.ppf(1 - alpha/2)
        # Margin of error for confidence interval
        margin_error = z_crit * (sample_std/np.sqrt(n))
        # Lower and upper bounds of confidence interval
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        # Statistics summary output
        st.subheader("Statistics Summary")
        st.write(f"Sample size n = {n}")
        st.write(f"Sample mean (estimated population mean): {sample_mean:.4f}")
        st.write(f"Sample standard deviation (estimated population std): {sample_std:.4f}")
        st.write(f"{int(conf_level*100)}% Confidence Interval: ({ci_lower:.4f} ~ {ci_upper:.4f})")
        
        st.subheader("Estimated Population Normal Distribution")
        # x-axis range: sample_mean ± 4*sample_std (original normal distribution)
        x = np.linspace(sample_mean - 4*sample_std, sample_mean + 4*sample_std, 200)
        # Calculate normal PDF with estimated mean and std
        y = stats.norm.pdf(x, loc=sample_mean, scale=sample_std)
        
        # Plotting
        fig, ax = plt.subplots()
        ax.plot(x, y, label="Estimated Normal Distribution")
        ax.axvline(sample_mean, color='r', linestyle='--', label='Estimated Population Mean')
        ax.fill_between(x, 0, y, where=(x >= ci_lower) & (x <= ci_upper), color='skyblue', alpha=0.5, label=f"{int(conf_level*100)}% Confidence Interval")
        ax.legend()
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        st.pyplot(fig)
