import streamlit as st
import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

N = norm.cdf

def call_black_schole(S, K, r, T, sigma):
    """
    S: strike price
    K: current price 
    r: risk free rate of intrest
    T: time until expiration 
    sigma: annual volatility  of assets return
    """
    d1 = (np.log(S/K)+(r + sigma ** 2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S * N(d1) - K * np.exp(-r*T) * N(d2)
    return call 

def put_black_schole(S, K, r, T, sigma):
    """
    S: strike price
    K: current price 
    r: risk free rate of intrest
    T: time until expiration 
    sigma: annual volatility  of assets return
    """
    d1 = (np.log(S/K)+(r + sigma ** 2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put = N(-d2) * K * np.exp(-r*T) - N(-d1) * S
    return put

st.title("Black-Scholes Pricing Model")

st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Current Asset Price (S)", min_value=0.01, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=70.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, max_value=10.0, step=0.1)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

st.sidebar.header("Heatmap Parameters")
min_spot = st.sidebar.number_input("Min Spot Price", min_value=0.01, value=40.00, step=1.0)
max_spot = st.sidebar.number_input("Max Spot Price", min_value=0.01, value=150.00, step=1.0)

st.table({
    "Current Asset Price": [S],
    "Strike Price": [K],
    "Time to Maturity": [T],
    "Volatility": [sigma],
    "Risk-Free Interest Rate": [r]
})

calls = call_black_schole(S, K, r, T, sigma)
puts = put_black_schole(S, K, r, T, sigma)

st.markdown(
    """
    <style>
    .box {
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        color: white;
    }
    .call-box {
        background-color: #28a745;
    }
    .put-box {
        background-color: #dc3545;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div class="box call-box">
            <div>CALL Value</div>
            <div>${calls:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="box put-box">
            <div>PUT Value</div>
            <div>${puts:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Generate heatmap data
K_range = np.linspace(min_spot, max_spot, 10)
sigma_range = np.linspace(0.1, 1.0, 10)

K_mesh, sigma_mesh = np.meshgrid(K_range, sigma_range)

calls_heatmap = call_black_schole(S, K_mesh, r, T, sigma_mesh)
puts_heatmap = put_black_schole(S, K_mesh, r, T, sigma_mesh)

st.title("Option Price Heatmaps")

# Create two columns for square heatmaps
heatmap_col1, heatmap_col2 = st.columns(2)

with heatmap_col1:
    st.subheader("Put Option Heatmap")
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    
    sns.heatmap(
        puts_heatmap, 
        xticklabels=[f'{k:.1f}' for k in K_range],
        yticklabels=[f'{s:.2f}' for s in sigma_range], 
        cmap="RdYlGn", 
        annot=True, 
        fmt=".2f", 
        cbar_kws={'label': 'Put Option Price'},
        ax=ax1,
        square=True
    )
    
    ax1.set_title("Black-Scholes Put Option", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Strike Price (K)", fontsize=12)
    ax1.set_ylabel("Volatility (σ)", fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig1)

with heatmap_col2:
    st.subheader("Call Option Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    
    sns.heatmap(
        calls_heatmap, 
        xticklabels=[f'{k:.1f}' for k in K_range],
        yticklabels=[f'{s:.2f}' for s in sigma_range], 
        cmap="RdYlGn", 
        annot=True, 
        fmt=".2f", 
        cbar_kws={'label': 'Call Option Price'},
        ax=ax2,
        square=True
    )
    
    ax2.set_title("Black-Scholes Call Option", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Strike Price (K)", fontsize=12)
    ax2.set_ylabel("Volatility (σ)", fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig2)