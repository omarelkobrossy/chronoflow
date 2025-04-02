import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

def calculate_technical_indicators(df, market_data=None):
    # Calculate Momentum Indicators
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    
    # Calculate RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # Calculate Volatility Indicators
    df["Rolling_Std_10"] = df["Close"].rolling(window=10).std()
    df["ATR"] = df["High"] - df["Low"]
    
    # Market Correlation Analysis
    if market_data is not None:
        # Index Correlations
        for market_symbol, market_df in market_data.items():
            if market_symbol != 'VIX':
                # Align market data with stock data
                aligned_market = market_df.reindex(df.index, method='ffill')
                
                # Rolling correlations
                df[f"{market_symbol}_Corr_60"] = df["Close"].rolling(60).corr(aligned_market["Close"])
                df[f"{market_symbol}_Corr_20"] = df["Close"].rolling(20).corr(aligned_market["Close"])
                
                # Beta calculation (60-period)
                market_returns = aligned_market["Close"].pct_change()
                stock_returns = df["Close"].pct_change()
                df[f"{market_symbol}_Beta_60"] = (stock_returns.rolling(60).cov(market_returns) / 
                                                market_returns.rolling(60).var())
                
                # Relative strength
                df[f"{market_symbol}_Rel_Strength"] = (df["Close"] / aligned_market["Close"])
                df[f"{market_symbol}_Rel_Strength_MA"] = df[f"{market_symbol}_Rel_Strength"].rolling(20).mean()
        
        # VIX Impact Analysis
        if 'VIX' in market_data:
            vix_data = market_data['VIX'].reindex(df.index, method='ffill')
            
            # VIX correlation
            df["VIX_Corr_60"] = df["Close"].rolling(60).corr(vix_data["Close"])
            df["VIX_Corr_20"] = df["Close"].rolling(20).corr(vix_data["Close"])
            
            # VIX impact on volatility
            df["VIX_Impact"] = df["ATR"] / vix_data["Close"]
            df["VIX_Impact_MA"] = df["VIX_Impact"].rolling(20).mean()
            
            # VIX regime detection
            df["VIX_Regime"] = np.where(
                vix_data["Close"] > vix_data["Close"].rolling(20).mean() + vix_data["Close"].rolling(20).std(),
                "High Volatility",
                np.where(
                    vix_data["Close"] < vix_data["Close"].rolling(20).mean() - vix_data["Close"].rolling(20).std(),
                    "Low Volatility",
                    "Normal"
                )
            )
    
    # Time-Sensitive Analysis
    # Opening Range Analysis
    df.index = pd.to_datetime(df.index)
    df["Date"] = df.index.date
    df["Time"] = df.index.time
    df["Hour"] = df.index.hour  # Extract hour before using it
    df["Minute"] = df.index.minute
    
    # Calculate Opening Range (first 30 minutes)
    opening_range = df.groupby("Date").apply(
        lambda x: pd.Series({
            "Opening_Range_High": x["High"].iloc[:30].max(),
            "Opening_Range_Low": x["Low"].iloc[:30].min(),
            "Opening_Range_Mid": (x["High"].iloc[:30].max() + x["Low"].iloc[:30].min()) / 2
        })
    )
    
    # Map opening range values to each row
    df["Opening_Range_High"] = df["Date"].map(opening_range["Opening_Range_High"])
    df["Opening_Range_Low"] = df["Date"].map(opening_range["Opening_Range_Low"])
    df["Opening_Range_Mid"] = df["Date"].map(opening_range["Opening_Range_Mid"])
    
    # Opening Range Breakout Analysis
    df["ORB_Status"] = np.where(
        df["Close"] > df["Opening_Range_High"], "Breakout",
        np.where(
            df["Close"] < df["Opening_Range_Low"], "Breakdown",
            "Inside"
        )
    )
    
    df["ORB_Distance"] = (df["Close"] - df["Opening_Range_Mid"]) / df["Opening_Range_Mid"]
    df["ORB_Strength"] = abs(df["Close"] - df["Opening_Range_Mid"]) / df["ATR"]
    
    # VWAP Analysis
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()
    df["VWAP_Distance"] = (df["Close"] - df["VWAP"]) / df["VWAP"]
    df["VWAP_Distance_Std"] = df["VWAP_Distance"] / df["VWAP_Distance"].rolling(20).std()
    df["VWAP_Mean_Reversion"] = df["VWAP_Distance_Std"].abs() > 2  # Flag for potential mean reversion
    
    # Pivot Point Analysis (Moved up before institutional footprint analysis)
    df["Pivot_Point"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["Support_1"] = (df["Pivot_Point"] * 2) - df["High"]
    df["Resistance_1"] = (df["Pivot_Point"] * 2) - df["Low"]
    df["Support_2"] = df["Pivot_Point"] - (df["High"] - df["Low"])
    df["Resistance_2"] = df["Pivot_Point"] + (df["High"] - df["Low"])
    
    # Pivot Point Strength
    df["Pivot_Strength"] = abs(df["Close"] - df["Pivot_Point"]) / df["ATR"]
    df["Support_Strength"] = (df["Close"] - df["Support_1"]) / df["ATR"]
    df["Resistance_Strength"] = (df["Resistance_1"] - df["Close"]) / df["ATR"]
    
    # Institutional Footprint Analysis
    # Volume Profile at Key Levels
    df["Price_Level"] = df["Close"].round(2)
    level_volume = df.groupby("Price_Level")["Volume"].transform("sum")
    df["Level_Volume_Profile"] = level_volume / level_volume.max()
    
    # Sudden Liquidity Shifts
    df["Volume_Shift"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Price_Shift"] = df["Close"].pct_change()
    df["Liquidity_Shift"] = df["Volume_Shift"] * df["Price_Shift"].abs()
    
    # Key Level Touches
    df["Touches_Support"] = (df["Low"] <= df["Support_1"]) & (df["Close"] > df["Support_1"])
    df["Touches_Resistance"] = (df["High"] >= df["Resistance_1"]) & (df["Close"] < df["Resistance_1"])
    
    # Institutional Activity Indicators
    df["Institutional_Activity"] = (
        (df["Volume_Shift"] > 2) &  # High volume
        (df["VWAP_Distance_Std"].abs() > 1.5) &  # Price away from VWAP
        (df["Touches_Support"] | df["Touches_Resistance"])  # At key levels
    )
    
    # Market Session Analysis
    df["Session"] = np.where(
        df["Hour"] < 9, "Pre-Market",
        np.where(
            df["Hour"] < 16, "Regular",
            "After-Hours"
        )
    )
    
    # Session-specific metrics
    session_volume = df.groupby("Session")["Volume"].transform("mean")
    df["Session_Volume_Ratio"] = df["Volume"] / session_volume
    
    # Fractal Pattern Detection
    # Local Highs and Lows
    df["Local_High"] = (df["High"] > df["High"].shift(1)) & (df["High"] > df["High"].shift(-1))
    df["Local_Low"] = (df["Low"] < df["Low"].shift(1)) & (df["Low"] < df["Low"].shift(-1))
    
    # Fractal Strength
    df["Fractal_High_Strength"] = df["High"].rolling(5).max() - df["High"]
    df["Fractal_Low_Strength"] = df["Low"] - df["Low"].rolling(5).min()
    
    # Fractal Volume Profile
    df["Fractal_High_Volume"] = df["Local_High"].astype(int) * df["Volume"]
    df["Fractal_Low_Volume"] = df["Local_Low"].astype(int) * df["Volume"]
    
    # Liquidity Zone Analysis
    # Price Distribution
    df["Price_Bin"] = pd.qcut(df["Close"], q=10, labels=False, duplicates='drop')
    
    # Volume Profile
    volume_profile = df.groupby("Price_Bin")["Volume"].transform("sum")
    df["Volume_Profile"] = volume_profile / volume_profile.max()
    
    # Time Spent in Zone
    zone_counts = df["Price_Bin"].value_counts(normalize=True)
    df["Time_In_Zone"] = df["Price_Bin"].map(zone_counts)
    
    # Liquidity Zone Strength
    df["Liquidity_Strength"] = df["Volume_Profile"] * df["Time_In_Zone"]
    
    # Support/Resistance Clusters
    df["Price_Cluster"] = df["Close"].round(2)  # Round to 2 decimal places
    cluster_volume = df.groupby("Price_Cluster")["Volume"].transform("sum")
    df["Cluster_Strength"] = cluster_volume / cluster_volume.max()
    
    # Dynamic Support/Resistance
    df["Dynamic_Support"] = df["Low"].rolling(20).min()
    df["Dynamic_Resistance"] = df["High"].rolling(20).max()
    df["Support_Resistance_Range"] = df["Dynamic_Resistance"] - df["Dynamic_Support"]
    
    # Price Structure Analysis
    df["Price_Structure"] = np.where(
        df["Close"] > df["Dynamic_Resistance"], "Breakout",
        np.where(
            df["Close"] < df["Dynamic_Support"], "Breakdown",
            np.where(
                df["Close"] > df["Pivot_Point"], "Bullish",
                "Bearish"
            )
        )
    )
    
    # Time-of-Day Volume Analysis
    df["Time_of_Day"] = df.index.time
    
    # Calculate time-of-day volume averages
    time_volume_avg = df.groupby(["Hour", "Minute"])["Volume"].transform("mean")
    df["Time_Adjusted_Volume"] = df["Volume"] / time_volume_avg
    df["Time_Adjusted_Volume_MA"] = df["Time_Adjusted_Volume"].rolling(window=20).mean()
    
    # Relative Volume Analysis
    df["Relative_Volume"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Relative_Volume_MA"] = df["Relative_Volume"].rolling(window=10).mean()
    df["Relative_Volume_Trend"] = df["Relative_Volume"].diff()
    
    # Large Trade Imbalance Detection
    df["Large_Trade_Imbalance"] = df["Volume"].rolling(10).max() / df["Volume"].rolling(50).mean()
    df["Trade_Imbalance_MA"] = df["Large_Trade_Imbalance"].rolling(window=20).mean()
    df["Trade_Imbalance_Trend"] = df["Large_Trade_Imbalance"].diff()
    
    # Enhanced Volume Spike Detection
    df["Volume_Spike"] = df["Volume"] > (df["Volume"].rolling(20).mean() + 2 * df["Volume"].rolling(20).std())
    df["Volume_Spike_Size"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Volume_Spike_Duration"] = df["Volume_Spike"].astype(int).rolling(5).sum()
    
    # Tick Imbalance Analysis
    df["Price_Tick"] = df["Close"].diff()
    df["Tick_Direction"] = np.where(df["Price_Tick"] > 0, 1, np.where(df["Price_Tick"] < 0, -1, 0))
    df["Tick_Sequence"] = df["Tick_Direction"].rolling(5).sum()
    df["Tick_Volume_Impact"] = df["Tick_Sequence"] * df["Time_Adjusted_Volume"]
    
    # Volume Price Trend
    df["VP_Trend"] = (df["Close"] - df["Close"].shift(1)) * df["Volume"]
    df["VP_Trend_SMA_5"] = df["VP_Trend"].rolling(window=5).mean()
    
    # Additional Technical Indicators
    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["Close"].rolling(window=20).std()
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["Close"].rolling(window=20).std()
    
    # Price momentum
    df["Price_Change"] = df["Close"].pct_change()
    df["Price_Change_5"] = df["Close"].pct_change(periods=5)
    
    # Candlestick Direction
    df["Candle_Direction"] = np.where(df["Close"] >= df["Open"], 1, -1)  # 1 for bullish, -1 for bearish
    
    # Candlestick Components (Absolute)
    df["Real_Body_Size"] = abs(df["Close"] - df["Open"])  # Always positive
    df["Upper_Shadow"] = df["High"] - np.maximum(df["Open"], df["Close"])  # Upper wick
    df["Lower_Shadow"] = np.minimum(df["Open"], df["Close"]) - df["Low"]  # Lower wick
    df["Total_Range"] = df["High"] - df["Low"]  # Total candle range
    
    # Candlestick Components (Percentage)
    df["Real_Body_Size_Pct"] = df["Real_Body_Size"] / df["Open"] * 100  # Always positive
    df["Upper_Shadow_Pct"] = df["Upper_Shadow"] / df["Open"] * 100
    df["Lower_Shadow_Pct"] = df["Lower_Shadow"] / df["Open"] * 100
    df["Total_Range_Pct"] = df["Total_Range"] / df["Open"] * 100
    
    # Shadow Analysis
    df["Shadow_Ratio"] = df["Upper_Shadow"] / (df["Lower_Shadow"] + 1e-6)  # Relative shadow comparison
    df["Body_To_Range_Ratio"] = df["Real_Body_Size"] / (df["Total_Range"] + 1e-6)  # Body to total range ratio
    
    # Directional Body and Shadow Impact
    df["Directional_Body"] = df["Real_Body_Size"] * df["Candle_Direction"]  # Positive for bullish, negative for bearish
    df["Directional_Body_Pct"] = df["Real_Body_Size_Pct"] * df["Candle_Direction"]
    
    # Moving averages of components
    df["Body_Size_MA5"] = df["Real_Body_Size_Pct"].rolling(window=5).mean()
    df["Upper_Shadow_MA5"] = df["Upper_Shadow_Pct"].rolling(window=5).mean()
    df["Lower_Shadow_MA5"] = df["Lower_Shadow_Pct"].rolling(window=5).mean()
    
    # Volume-weighted analysis
    rel_volume = df["Volume"] / df["Volume"].rolling(window=5).mean()
    df["Body_Vol_Impact"] = df["Directional_Body_Pct"] * rel_volume
    df["Upper_Shadow_Vol_Impact"] = df["Upper_Shadow_Pct"] * rel_volume
    df["Lower_Shadow_Vol_Impact"] = df["Lower_Shadow_Pct"] * rel_volume
    
    # Trend Analysis
    df["Body_Size_Trend"] = df["Body_Size_MA5"].diff()  # Increasing or decreasing body size
    df["Shadow_Balance"] = df["Upper_Shadow_MA5"] - df["Lower_Shadow_MA5"]  # Shadow balance trend
    
    # Pattern Detection (basic)
    df["Doji"] = (df["Real_Body_Size_Pct"] <= 0.1) & (df["Total_Range_Pct"] > 0.2)
    df["Marubozu"] = (df["Upper_Shadow_Pct"] <= 0.1) & (df["Lower_Shadow_Pct"] <= 0.1) & (df["Real_Body_Size_Pct"] > 0.5)
    df["Long_Upper_Shadow"] = df["Upper_Shadow_Pct"] > (2 * df["Real_Body_Size_Pct"])
    df["Long_Lower_Shadow"] = df["Lower_Shadow_Pct"] > (2 * df["Real_Body_Size_Pct"])
    
    # Composite Features  
    # 1. Volatility and Body Size Interaction
    df['Vol_Body_Interaction'] = df['Body_Size_MA5'] * df['Rolling_Std_10']
    
    # 2. Enhanced Price Action Signal
    df['Price_Action_Signal'] = (df['Body_Size_MA5'] * df['Shadow_Balance']) / (df['Rolling_Std_10'] + 1e-6)
    
    # 3. Market Impact Score
    if market_data is not None:
        df['Market_Impact'] = df['DIA_Corr_20'] * df['Level_Volume_Profile']
    
    # 4. Support/Resistance Strength
    df['SR_Strength'] = np.abs(df['Dynamic_Support'] - df['Price_Shift'])
    
    # 5. Volatility Regime
    df['Volatility_Regime'] = df['Rolling_Std_10'] / df['Rolling_Std_10'].rolling(window=20).mean()
    
    # 6. Enhanced Volume Impact
    df['Enhanced_Volume_Impact'] = df['Volume_Shift'] * df['Body_To_Range_Ratio']
    
    # 7. Trend Strength
    df['Trend_Strength'] = df['Directional_Body'] * df['Volume_Profile'] * (1 + df['Body_Vol_Impact'])
    
    # 8. Support/Resistance Break Potential
    df['SR_Break_Potential'] = (df['Close'] - df['Dynamic_Support']) / (df['Dynamic_Resistance'] - df['Dynamic_Support'])
    
    # Drop NaN values from rolling calculations
    df.dropna(inplace=True)
    
    return df

def fetch_data(symbol, interval="1min", outputsize="full"):
    # Alpha Vantage API configuration
    API_KEY = "REDACTED_ALPHA_VANTAGE_KEY"
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Initialize an empty list to store all data
    all_data = []
    
    # Alpha Vantage returns data in chunks, so we need to make multiple requests
    # Each request will get 50000 data points
    while True:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": API_KEY,
            "outputsize": outputsize
        }
        
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Time Series (1min)" not in data:
            print("Error: No data received from Alpha Vantage")
            break
            
        # Extract the time series data
        time_series = data["Time Series (1min)"]
        
        # Convert the data to a DataFrame
        df_chunk = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns
        df_chunk.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert string values to float
        for col in df_chunk.columns:
            df_chunk[col] = df_chunk[col].astype(float)
            
        # Add the chunk to our list
        all_data.append(df_chunk)
        
        # Check if we have enough data
        if len(df_chunk) < 50000:  # If we got less than the maximum, we've reached the end
            break
            
        # Alpha Vantage has a rate limit of 5 calls per minute
        time.sleep(12)  # Wait 12 seconds before making the next request
    
    # Combine all chunks
    df = pd.concat(all_data)
    
    # Sort by date
    df = df.sort_index()
    
    # Fetch market data for correlation analysis
    market_symbols = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'Nasdaq ETF',
        'DIA': 'Dow ETF',
        'VIX': 'Volatility Index'
    }
    
    market_data = {}
    for market_symbol in market_symbols:
        try:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": market_symbol,
                "interval": interval,
                "apikey": API_KEY,
                "outputsize": outputsize
            }
            
            response = requests.get(BASE_URL, params=params)
            print(f"Fetching {market_symbol} data...")
            data = response.json()
            
            if f"Time Series ({interval})" in data:
                market_df = pd.DataFrame.from_dict(data[f"Time Series ({interval})"], orient='index')
                market_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                market_df = market_df.astype(float)
                market_data[market_symbol] = market_df
                
            time.sleep(12)  # Respect rate limit
        except Exception as e:
            print(f"Error fetching {market_symbol} data: {str(e)}")
    
    # Calculate technical indicators with market context
    return calculate_technical_indicators(df, market_data)

if __name__ == "__main__":
    try:
        df = fetch_data("AAPL", "1min", "full")
        print("\nFirst few rows of the data with indicators:")
        print(df.head())
    except Exception as e:
        print(f"An error occurred: {str(e)}")
