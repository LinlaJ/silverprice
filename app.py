import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import plotly.express as px
from urllib.parse import urlencode
import base64

# Page configuration
st.set_page_config(page_title="Silver Price Predictor", layout="wide")

# Title and description
st.title("🔮 Silver Price Predictor")
st.markdown("Analyze historical silver prices and predict future trends using machine learning")

# Sidebar configuration
st.sidebar.header("Configuration")

# Date picker for historical data
date_range = st.sidebar.date_input(
    "Select date range for historical data",
    value=(datetime.now() - timedelta(days=90), datetime.now()),
    max_value=datetime.now()
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()

# Prediction days
prediction_days = st.sidebar.slider("Prediction period (days)", 30, 180, 180)

# Model selection
model_type = st.sidebar.selectbox(
    "Select prediction model",
    ["Linear Regression", "Polynomial (Degree 2)", "Polynomial (Degree 3)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Silver Price Data Source:** Yahoo Finance (SLV - iShares Silver Trust ETF)")

# Fetch silver price data
@st.cache_data
def fetch_silver_data(start, end):
    try:
        # Using SLV (iShares Silver Trust) as proxy for silver prices
        data = yf.download("SLV", start=start, end=end, progress=False)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Load data
with st.spinner("Fetching silver price data..."):
    silver_data = fetch_silver_data(start_date, end_date)

if silver_data is not None and len(silver_data) > 0:
    # Prepare data
    silver_data_reset = silver_data.reset_index()
    
    # Use Adjusted Close price, with fallback to Close
    price_col = 'Adj Close' if 'Adj Close' in silver_data_reset.columns else 'Close'
    prices = silver_data_reset[['Date', price_col]].copy()
    prices.columns = ['Date', 'Price']
    
    # Display historical data statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${prices['Price'].iloc[-1]:.2f}")
    with col2:
        st.metric("Min Price", f"${prices['Price'].min():.2f}")
    with col3:
        st.metric("Max Price", f"${prices['Price'].max():.2f}")
    with col4:
        change = prices['Price'].iloc[-1] - prices['Price'].iloc[0]
        pct_change = (change / prices['Price'].iloc[0]) * 100
        st.metric("Change (3M)", f"${change:.2f}", f"{pct_change:.2f}%")
    
    st.markdown("---")
    
    # Historical chart
    st.subheader("📊 Historical Silver Prices (Last 3 Months)")
    
    fig_historical = go.Figure()
    fig_historical.add_trace(go.Scatter(
        x=prices['Date'],
        y=prices['Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig_historical.update_layout(
        title="Historical Silver Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_historical, use_container_width=True)
    
    st.markdown("---")
    
    # Prepare data for prediction
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices['Price'].values
    
    # Select model
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X, y)
        degree = 1
    elif model_type == "Polynomial (Degree 2)":
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        degree = 2
    else:  # Polynomial (Degree 3)
        poly_features = PolynomialFeatures(degree=3)
        X_poly = poly_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        degree = 3
    
    # Make predictions for future
    future_days = prediction_days
    last_date = prices['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
    
    X_future = np.arange(len(prices), len(prices) + future_days).reshape(-1, 1)
    
    if degree == 1:
        y_future = model.predict(X_future)
    else:
        X_future_poly = poly_features.transform(X_future)
        y_future = model.predict(X_future_poly)
    
    # Create combined dataframe
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': y_future
    })
    
    # Calculate predicted change
    current_price = prices['Price'].iloc[-1]
    final_predicted = y_future[-1]
    predicted_change = final_predicted - current_price
    predicted_pct = (predicted_change / current_price) * 100
    
    # Display prediction metrics
    st.subheader("🎯 6-Month Price Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Price (6M)", f"${final_predicted:.2f}")
    with col2:
        st.metric("Predicted Change", f"${predicted_change:.2f}", f"{predicted_pct:.2f}%")
    with col3:
        st.metric("Model", model_type)
    
    st.markdown("---")
    
    # Combined chart with historical and predictions
    st.subheader("📈 Historical Data + Future Predictions")
    
    fig_combined = go.Figure()
    
    # Historical data
    fig_combined.add_trace(go.Scatter(
        x=prices['Date'],
        y=prices['Price'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    # Predictions
    fig_combined.add_trace(go.Scatter(
        x=predictions_df['Date'],
        y=predictions_df['Predicted Price'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.2)'
    ))
    
    fig_combined.update_layout(
        title="Silver Price: Historical & Predicted (6 Months)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_combined, use_container_width=True)
    
    st.markdown("---")
    
    # Display prediction table
    st.subheader("📋 Detailed Predictions")
    
    display_df = predictions_df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df['Predicted Price'] = display_df['Predicted Price'].round(2)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Sharing section
    st.subheader("🔗 Share Your Analysis")
    
    # Create shareable parameters
    share_params = {
        'model': model_type,
        'period': f"{prediction_days}d",
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
    }
    
    # Generate share link (encoded)
    share_url = f"https://silverprice-predictor.streamlit.app/?{urlencode(share_params)}"
    
    # Generate a shorter shareable code
    share_data = base64.b64encode(
        f"{start_date.strftime('%Y-%m-%d')}|{end_date.strftime('%Y-%m-%d')}|{model_type}|{prediction_days}".encode()
    ).decode()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input(
            "Share this analysis:",
            value=share_url,
            disabled=True,
            label_visibility="collapsed"
        )
        st.caption("Copy the URL to share this analysis")
    
    with col2:
        st.text_input(
            "Share code:",
            value=share_data,
            disabled=True,
            label_visibility="collapsed"
        )
        st.caption("Short code format")
    
    # Download prediction data
    st.markdown("---")
    st.subheader("💾 Download Data")
    
    # Prepare CSV for download
    combined_data = pd.concat([
        prices.rename(columns={'Price': 'Value', 'Date': 'Date'}).assign(Type='Historical'),
        predictions_df.rename(columns={'Predicted Price': 'Value'}).assign(Type='Predicted')
    ], ignore_index=True)
    
    csv = combined_data.to_csv(index=False)
    
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"silver_price_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        <p>Data source: Yahoo Finance | Model: Scikit-learn</p>
        <p>Disclaimer: This prediction is for educational purposes only and should not be considered financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Unable to fetch silver price data. Please try again later.")
