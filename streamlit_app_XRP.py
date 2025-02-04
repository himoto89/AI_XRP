import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# 실행코드 !streamlit run .\streamlit_app_XRP.py

# 데이터베이스 연결 함수
def get_connection():
    return sqlite3.connect('XRP_trades.db')

# 데이터 로드 함수
def load_data():
    conn = get_connection()
    query = "SELECT * FROM trades"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# 메인 함수
def main():
    st.title('XRP Trades Viewer')

    # 데이터 로드
    df = load_data()

    # 기본 통계
    st.header('Basic Statistics')
    st.write(f"Total number of trades: {len(df)}")
    st.write(f"First trade date: {df['timestamp'].min()}")
    st.write(f"Last trade date: {df['timestamp'].max()}")

    # 거래 내역 표시
    st.header('Trade History')
    st.dataframe(df)

    # 거래 결정 분포
    st.header('Trade Decision Distribution')
    decision_counts = df['decision'].value_counts()
    fig = px.pie(values=decision_counts.values, names=decision_counts.index, title='Trade Decisions')
    st.plotly_chart(fig)

    # XRP 잔액 변화
    st.header('XRP Balance Over Time')
    fig = px.line(df, x='timestamp', y='XRP_balance', title='XRP Balance')
    st.plotly_chart(fig)

    # KRW 잔액 변화
    st.header('KRW Balance Over Time')
    fig = px.line(df, x='timestamp', y='krw_balance', title='KRW Balance')
    st.plotly_chart(fig)

    # XRP 가격 변화
    st.header('XRP Price Over Time')
    fig = px.line(df, x='timestamp', y='XRP_krw_price', title='XRP Price (KRW)')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()