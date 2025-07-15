import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Streamlit 페이지 설정
st.set_page_config(page_title="t-추정 및 정규분포 시각화", layout="centered")

st.title("t-추정 및 정규분포 시각화 사이트")
st.write("""
엑셀 또는 CSV 파일로 표본 데이터를 업로드하면 t-추정을 통해 모집단의 모평균, 모표준편차 추정치와 신뢰구간을 계산하고, 추정된 정규분포 그래프를 그려줍니다.
""")

# 파일 업로드 위젯
uploaded_file = st.file_uploader("표본 데이터 파일 업로드 (Excel 또는 CSV)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    # csv 파일일 경우 인코딩 자동 감지 (한글 깨짐 방지)
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='cp949')
    else:
        # 엑셀 파일 읽기
        df = pd.read_excel(uploaded_file)
    
    # 데이터프레임 표시
    st.dataframe(df)
    
    # 수치형 컬럼만 선택
    columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not columns:
        st.error("수치형 데이터가 포함된 컬럼이 없습니다.")
    else:
        # 분석할 컬럼 선택
        col = st.selectbox("분석할 컬럼을 선택하세요", columns)
        data = df[col].dropna().values
        
        n = len(data)  # 표본 크기
        sample_mean = np.mean(data)  # 표본평균
        sample_std = np.std(data, ddof=1)  # 표본표준편차 (ddof=1로 표본 표준편차 계산)
        alpha = st.slider("신뢰수준(1-α)", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
        conf_level = alpha
        alpha = 1 - conf_level
        
        # t-분포 임계값 계산
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        # 신뢰구간의 오차 한계 계산
        margin_error = t_crit * (sample_std/np.sqrt(n))
        # 신뢰구간 하한, 상한 계산
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        # 통계 요약 출력
        st.subheader("통계 요약")
        st.write(f"표본 크기 n = {n}")
        st.write(f"표본평균 (모평균 추정치): {sample_mean:.4f}")
        st.write(f"표본표준편차 (모표준편차 추정치): {sample_std:.4f}")
        st.write(f"{int(conf_level*100)}% 신뢰구간: ({ci_lower:.4f} ~ {ci_upper:.4f})")
        
        st.subheader("추정된 모집단 정규분포 그래프")
        # x축 범위는 표본평균±4*표본표준편차로 설정 (원본 정규분포를 그대로 그림)
        x = np.linspace(sample_mean - 4*sample_std, sample_mean + 4*sample_std, 200)
        # 표본평균과 표본표준편차로 정규분포 PDF 계산
        y = stats.norm.pdf(x, loc=sample_mean, scale=sample_std)
        
        # 그래프 그리기
        fig, ax = plt.subplots()
        ax.plot(x, y, label="추정된 정규분포")
        ax.axvline(sample_mean, color='r', linestyle='--', label='추정 모평균')
        ax.fill_between(x, 0, y, where=(x >= ci_lower) & (x <= ci_upper), color='skyblue', alpha=0.5, label=f"{int(conf_level*100)}% 신뢰구간")
        ax.legend()
        ax.set_xlabel("값")
        ax.set_ylabel("확률밀도")
        st.pyplot(fig)
