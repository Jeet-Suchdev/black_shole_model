# Black-Scholes Options Pricing Dashboard

A interactive Streamlit web application for calculating and visualizing European call and put option prices using the Black-Scholes pricing model.

## Features

- **Real-time Option Pricing**: Calculate call and put option values instantly
- **Interactive Parameter Controls**: Adjust all Black-Scholes parameters via sidebar inputs
- **Visual Heatmaps**: Explore how option prices vary with strike price and volatility
- **Clean UI**: Color-coded cards displaying call (green) and put (red) values

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Libraries

Install the required dependencies using pip:

```bash
pip install streamlit numpy scipy matplotlib seaborn
```

Or create a `requirements.txt` file with:

```
streamlit
numpy
scipy
matplotlib
seaborn
```

Then install using:

```bash
pip install -r requirements.txt
```

## Usage

1. Navigate to the project directory
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## Application Interface

### Sidebar Controls

Adjust the following parameters:

- **Current Asset Price (S)**: The current market price of the underlying asset
- **Strike Price (K)**: The predetermined price at which the option can be exercised
- **Time to Maturity (Years)**: Time remaining until option expiration
- **Volatility (σ)**: Annual volatility of the asset's returns (0.01 to 1.0)
- **Risk-Free Interest Rate**: The theoretical return on a risk-free investment

### Heatmap Parameters

- **Min Spot Price**: Minimum strike price for heatmap visualization
- **Max Spot Price**: Maximum strike price for heatmap visualization

### Output

The dashboard displays:

1. **Parameter Summary Table**: Current values of all input parameters
2. **Option Values**: Large, color-coded cards showing:
   - Call Option Value (Green)
   - Put Option Value (Red)
3. **Interactive Heatmaps**: 
   - Put Option Heatmap: Shows how put prices vary with strike price and volatility
   - Call Option Heatmap: Shows how call prices vary with strike price and volatility

## Black-Scholes Formula

The application implements the classic Black-Scholes formulas for European options:

### Call Option
```
C = S * N(d1) - K * e^(-rT) * N(d2)
```

### Put Option
```
P = K * e^(-rT) * N(-d2) - S * N(-d1)
```

Where:
- `d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)`
- `d2 = d1 - σ√T`
- `N(x)` = Cumulative distribution function of the standard normal distribution

## Example Use Cases

- **Options Traders**: Quickly calculate theoretical option prices
- **Financial Analysts**: Analyze option sensitivity to different parameters
- **Students**: Learn about options pricing through interactive visualization
- **Risk Managers**: Assess option portfolio values under various scenarios

## Customization

You can modify the application by:

- Adjusting default parameter values in the `st.sidebar.number_input()` functions
- Changing heatmap resolution by modifying `np.linspace()` parameters
- Customizing color schemes in the heatmap `cmap` parameter
- Adding Greeks calculations (Delta, Gamma, Theta, Vega, Rho)

## Limitations

- Implements European-style options only (exercise only at maturity)
- Assumes constant volatility and risk-free rate
- Does not account for dividends
- No transaction costs or taxes included

## Contributing

Feel free to fork this project and submit pull requests for:

- Additional option pricing models (Binomial, Monte Carlo)
- Greeks calculations and visualization
- Implied volatility calculator
- Multiple option strategy analyzer
- Historical data integration

## License

This project is open source and available for educational and commercial use.

## Support

For issues, questions, or suggestions, please open an issue in the project repository.

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web application framework
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Scientific computing
- [Matplotlib](https://matplotlib.org/) - Visualization
- [Seaborn](https://seaborn.pydata.org/) - Statistical visualization
