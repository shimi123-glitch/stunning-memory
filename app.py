import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# פונקציית ה-SAR (הלוגיקה שלך)
def calculate_sar(high, low, iaf=0.02, maxaf=0.2):
    length = len(high)
    sar = np.zeros(length)
    trend = np.zeros(length)
    ep = np.zeros(length)
    af = np.zeros(length)
    
    trend[0] = 1
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = iaf
    
    for i in range(1, length):
        if trend[i-1] == 1:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], low[i-1], low[max(0, i-2)])
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = iaf
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + iaf, maxaf)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:
            sar[i] = sar[i-1] - af[i-1] * (sar[i-1] - ep[i-1])
            sar[i] = max(sar[i], high[i-1], high[max(0, i-2)])
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = iaf
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + iaf, maxaf)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    return sar, trend

# --- ממשק האתר (Streamlit) ---
st.set_page_config(page_title="Parabolic SAR Tool", layout="wide")

st.title("📈 מחשבון Parabolic SAR אינטראקטיבי")
st.sidebar.header("הגדרות אינדיקטור")

# אפשרות למשתמש לשנות פרמטרים בצד
iaf = st.sidebar.slider("Initial AF", 0.01, 0.1, 0.02, step=0.01)
maxaf = st.sidebar.slider("Max AF", 0.1, 0.5, 0.2, step=0.05)

# נתונים לדוגמה (במציאות אפשר למשוך מ-Yahoo Finance)
data = {
    'High': [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 107, 105, 108, 110, 112, 115, 114, 110, 108],
    'Low': [98, 100, 102, 101, 103, 105, 104, 106, 108, 107, 105, 103, 106, 108, 110, 113, 111, 107, 105],
    'Close': [99, 101, 103, 102, 104, 106, 105, 107, 109, 108, 106, 104, 107, 109, 111, 114, 112, 108, 106]
}
df = pd.DataFrame(data)

# חישוב
df['SAR'], df['Trend'] = calculate_sar(df['High'].values, df['Low'].values, iaf, maxaf)

# יצירת הגרף עם Plotly
fig = go.Figure()

# הוספת קו מחיר
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='מחיר סגירה', line=dict(color='black', width=2)))

# הוספת נקודות SAR
df_up = df[df['Trend'] == 1]
df_down = df[df['Trend'] == -1]

fig.add_trace(go.Scatter(x=df_up.index, y=df_up['SAR'], name='לונג (עלייה)', mode='markers', marker=dict(color='green', size=8)))
fig.add_trace(go.Scatter(x=df_down.index, y=df_down['SAR'], name='שורט (ירידה)', mode='markers', marker=dict(color='red', size=8)))

fig.update_layout(title="ניתוח טכני - Parabolic SAR", xaxis_title="זמן", yaxis_title="מחיר", height=600)

# הצגת הגרף באתר
st.plotly_chart(fig, use_container_width=True)

# הצגת טבלה
st.subheader("נתוני מסחר")
st.write(df.tail(10))