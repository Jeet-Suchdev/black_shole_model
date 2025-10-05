import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

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

K1 = np.arange(90,100,1)
r1 = 0.05
# T = np.arange(0,1,0.01)
T1 = 1
sigma1 = np.arange(0.1,1,0.1)
S1 = 99

K_i,sigma_i = np.meshgrid(K1,sigma1)

calls = call_black_schole(S,K_i,r,T,sigma_i)
puts = put_black_schole(S,K_i,r,T,sigma_i)

# plt.plot(calls, label = "call option")
# plt.plot(puts, label = "put option")
# plt.legend()
# plt.show()



print(f" calls: {calls} \n puts: {puts}")

# --- Seaborn Heatmap ---

fig , (ax1, ax2) = plt.subplots(1,2,figsize = (15,10))



sns.heatmap(
    puts, 
    # xticklabels=K, 
    # yticklabels=np.round(sigma, 2), 
    cmap="viridis", 
    annot=True, 
    fmt=".2f", 
    cbar_kws={'label': 'Put Option Price'},
    ax=ax1
)

ax1.set_title("Black–Scholes Put Option Heatmap")
ax1.set_xlabel("Strike Price (K)")
ax1.set_ylabel("Volatility (σ)")

sns.heatmap(
    puts, 
    # xticklabels=K, 
    # yticklabels=np.round(sigma, 2), 
    cmap="viridis", 
    annot=True, 
    fmt=".2f", 
    cbar_kws={'label': 'Put Option Price'},
    ax=ax2
)
ax2.set_title("Black–Scholes Call Option Heatmap")
ax2.set_xlabel("Strike Price (K)")
ax2.set_ylabel("Volatility (σ)")

plt.show()


