# Silver Price Predictor

A Streamlit web application that analyzes historical silver prices and predicts future trends using machine learning.

## Features

- 📊 **Historical Data Analysis**: View 3 months of silver price history
- 🎯 **Price Predictions**: Forecast silver prices for up to 6 months
- 📈 **Interactive Charts**: Visualize historical data and predictions
- 🔗 **Sharing**: Generate shareable links and codes for your analysis
- 📋 **Data Export**: Download analysis data as CSV

## Technologies

- **Streamlit**: Interactive web framework
- **Yahoo Finance (yfinance)**: Real-time silver price data via SLV ETF
- **scikit-learn**: Machine learning for predictions
- **Plotly**: Interactive charts and visualizations
- **Pandas & NumPy**: Data manipulation and analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LinlaJ/silverprice.git
cd silverprice
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # On Windows
source .venv/bin/activate    # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. Use the sidebar to adjust:
   - Historical data date range (up to 3 months)
   - Prediction period (30-180 days)
   - Machine learning model (Linear, Polynomial Degree 2 or 3)

2. View the analysis:
   - Historical price statistics
   - Combined chart with predictions
   - Detailed prediction table

3. Share or export:
   - Copy the shareable URL or code
   - Download analysis data as CSV

## Data Source

- **Silver ETF**: SLV (iShares Silver Trust)
- **Provider**: Yahoo Finance

## Disclaimer

This prediction tool is for educational purposes only and should not be considered financial advice. Always conduct your own research before making investment decisions.

## License

MIT License

## Author

LinlaJ
