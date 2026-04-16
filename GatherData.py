import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from GetMarketConditions import get_market_data_for_period

def calculate_technical_indicators(df, market_data=None, timeframe=1):
    """
    Calculate technical indicators with timeframe-specific adjustments
    All features are calculated using only information available up to the current bar
    to avoid forward-looking bias.
    
    Args:
        df: DataFrame with OHLCV data
        market_data: Optional dictionary of market data DataFrames
        timeframe: Timeframe in minutes (default=1)
    """
    # Adjust periods based on timeframe
    period_adjustment = timeframe
    
    # Calculate Momentum Indicators (using only past data)
    df["SMA_5"] = df["Close"].shift(1).rolling(window=5).mean()
    df["EMA_10"] = df["Close"].shift(1).ewm(span=10, adjust=False).mean()
    
    # Calculate RSI (using only past data)
    # First calculate price changes using only past data
    price_changes = df["Close"].shift(1).diff()
    gain = (price_changes.where(price_changes > 0, 0)).rolling(window=14).mean()
    loss = (-price_changes.where(price_changes < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic RSI (using only past data)
    # Stochastic RSI combines RSI with Stochastic oscillator concepts
    rsi_min = df["RSI_14"].shift(1).rolling(window=14).min()
    rsi_max = df["RSI_14"].shift(1).rolling(window=14).max()
    df["Stoch_RSI_K"] = 3 #100 * (df["RSI_14"].shift(1) - rsi_min) / (rsi_max - rsi_min + 1e-8)
    df["Stoch_RSI_D"] = 3 #df["Stoch_RSI_K"].shift(1).rolling(window=3).mean()
    
    # Stochastic RSI overbought/oversold signals
    df["Stoch_RSI_Overbought"] = (df["Stoch_RSI_K"] > 80).astype(int)
    df["Stoch_RSI_Oversold"] = (df["Stoch_RSI_K"] < 20).astype(int)
    
    # Stochastic RSI divergence signals
    df["Stoch_RSI_Bullish_Divergence"] = (
        (df["Stoch_RSI_K"] > df["Stoch_RSI_K"].shift(1)) & 
        (df["Close"].shift(1) < df["Close"].shift(2))
    ).astype(int)
    
    df["Stoch_RSI_Bearish_Divergence"] = (
        (df["Stoch_RSI_K"] < df["Stoch_RSI_K"].shift(1)) & 
        (df["Close"].shift(1) > df["Close"].shift(2))
    ).astype(int)
    
    # Stochastic RSI momentum
    df["Stoch_RSI_Momentum"] = df["Stoch_RSI_K"] - df["Stoch_RSI_K"].shift(5)
    df["Stoch_RSI_Acceleration"] = df["Stoch_RSI_Momentum"].diff()
    
    # Stochastic RSI trend strength
    df["Stoch_RSI_Trend_Strength"] = abs(df["Stoch_RSI_K"] - 50) / 50
    
    # Stochastic RSI volatility
    df["Stoch_RSI_Volatility"] = df["Stoch_RSI_K"].shift(1).rolling(window=14).std()
    
    # Stochastic RSI mean reversion signals
    df["Stoch_RSI_Mean_Reversion_Signal"] = (
        (df["Stoch_RSI_K"] > 80) | (df["Stoch_RSI_K"] < 20)
    ).astype(int)
    
    # Stochastic RSI breakout signals
    df["Stoch_RSI_Breakout_Up"] = (
        (df["Stoch_RSI_K"] > df["Stoch_RSI_K"].shift(1)) & 
        (df["Stoch_RSI_K"].shift(1) < 20)
    ).astype(int)
    
    df["Stoch_RSI_Breakout_Down"] = (
        (df["Stoch_RSI_K"] < df["Stoch_RSI_K"].shift(1)) & 
        (df["Stoch_RSI_K"].shift(1) > 80)
    ).astype(int)
    
    # Mean Reversion Features
    print("Calculating mean reversion features...")
    
    # Z-Score based mean reversion
    df['Price_MA_20'] = df['Close'].shift(1).rolling(window=20).mean()
    df['Price_Std_20'] = df['Close'].shift(1).rolling(window=20).std()
    df['Z_Score_20'] = (df['Close'].shift(1) - df['Price_MA_20']) / df['Price_Std_20']
    
    # Bollinger Bands mean reversion
    df['BB_Upper'] = df['Price_MA_20'] + (df['Price_Std_20'] * 2)
    df['BB_Lower'] = df['Price_MA_20'] - (df['Price_Std_20'] * 2)
    df['BB_Position'] = (df['Close'].shift(1) - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Price_MA_20']
    
    # Z-score Bollinger
    df['Zscore_Bollinger'] = (df['Close'].shift(1) - df['Price_MA_20']) / (df['Price_Std_20'] + 1e-8)
    
    # Mean Reversion Trigger (fade signal)
    df['MeanReversion_Trigger'] = (
        (df['Zscore_Bollinger'] > 2) &  # Price above 2 standard deviations
        (df['RSI_14'] > 70)  # RSI above 70
    ).astype(int)
    
    # Multiple timeframe mean reversion
    for window in [5, 10, 20, 50]:
        # Percentage deviation from moving average
        df[f'Deviation_MA_{window}'] = (df['Close'].shift(1) - df['Close'].shift(1).rolling(window=window).mean()) / df['Close'].shift(1).rolling(window=window).mean() * 100
        
        # Rate of mean reversion
        df[f'Mean_Reversion_Speed_{window}'] = df[f'Deviation_MA_{window}'].diff()
        
        # Oscillator around moving average
        df[f'MA_Oscillator_{window}'] = df['Close'].shift(1) / df['Close'].shift(1).rolling(window=window).mean() - 1
    
    # RSI-based mean reversion
    df['RSI_Reversal'] = df['RSI_14'].shift(1) - 50  # Distance from neutral RSI level
    
    # Historical percentile mean reversion
    for window in [20, 50, 100]:
        # Price percentile within rolling window
        df[f'Price_Percentile_{window}'] = df['Close'].shift(1).rolling(window=window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # Distance from historical median
        rolling_median = df['Close'].shift(1).rolling(window=window).median()
        df[f'Median_Distance_{window}'] = (df['Close'].shift(1) - rolling_median) / rolling_median
    
    # Mean reversion strength indicators
    df['Mean_Reversion_Strength'] = abs(df['Z_Score_20']) * df['RSI_Reversal'].abs()
    
    # Volatility-adjusted mean reversion
    df['Vol_Adj_Mean_Reversion'] = df['Z_Score_20'] / (df['BB_Width'] + 1e-8)
    
    # Momentum-adjusted mean reversion
    df['Mom_Adj_Mean_Reversion'] = df['Z_Score_20'] * (1 - abs(df['RSI_14'] - 50) / 50)
    
    # Volume-weighted mean reversion
    relative_volume = df['Volume'].shift(1) / df['Volume'].shift(1).rolling(window=20).mean()
    df['Volume_Weighted_MR'] = df['Z_Score_20'] * relative_volume
    
    # Composite mean reversion score
    df['Mean_Reversion_Score'] = (
        df['Z_Score_20'].abs() +  # Price deviation
        abs(df['RSI_Reversal']) / 50 +  # RSI-based mean reversion
        df['BB_Position'].apply(lambda x: min(abs(x - 0.5) * 2, 1))  # Bollinger band position
    ) / 3  # Average of components
    
    # Calculate Volatility Indicators with adjusted periods (using only past data)
    df["Rolling_Std_10"] = df["Close"].shift(1).rolling(window=10).std()
    df["Rolling_Std_20"] = df["Close"].shift(1).rolling(window=20).std()
    df["Rolling_Std_60"] = df["Close"].shift(1).rolling(window=60).std()
    
    # Calculate True Range for ATR calculations (using only past data)
    # True Range = max(H-L, H-C_prev, C_prev-L)
    high_low = df["High"].shift(1) - df["Low"].shift(1)
    high_close_prev = abs(df["High"].shift(1) - df["Close"].shift(2))
    low_close_prev = abs(df["Low"].shift(1) - df["Close"].shift(2))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate multiple ATR periods using RMA (Wilder's smoothing)
    # RMA formula: RMA = (previous_RMA * (period-1) + current_value) / period
    def calculate_rma(series, period):
        """Calculate RMA (Wilder's smoothing) for ATR"""
        alpha = 1.0 / period
        rma = series.ewm(alpha=alpha, adjust=False).mean()
        return rma
    
    # ATR with different periods using RMA
    df["ATR_5"] = calculate_rma(true_range, 5)
    df["ATR_10"] = calculate_rma(true_range, 10)
    df["ATR_14"] = calculate_rma(true_range, 14)  # Traditional ATR period
    df["ATR_15"] = calculate_rma(true_range, 15)
    df["ATR_20"] = calculate_rma(true_range, 20)
    
    # Keep the original ATR for backward compatibility (using ATR_14)
    df["ATR"] = df["ATR_14"]
    
    
    # Market Correlation Analysis (using only past data)
    if market_data is not None:
        market_symbols = ['SPY', 'QQQ', 'DIA', 'VXX']
        for market_symbol in market_symbols:
            if market_symbol in market_data:
                market_df = market_data[market_symbol]
                
                # Check if we have enough data for this symbol
                if len(market_df) < 5:
                    print(f"Warning: Insufficient data for {market_symbol}. Skipping correlation calculations.")
                    continue
                
                # Check for missing values
                if market_df['Close'].isna().any():
                    print(f"Warning: Found NaN values in {market_symbol} data. Filling with backward fill only.")
                    market_df = market_df.bfill()
                
                # Align market data with stock data using backward fill
                aligned_market = market_df.reindex(df.index, method='ffill')
                
                # Basic correlations - with error handling
                try:
                    # Correlations over 5 bars - using only past data
                    df[f"{market_symbol}_Corr_5"] = df["Close"].shift(1).rolling(5).corr(aligned_market["Close"].shift(1))
                    df[f"{market_symbol}_Corr_5_MA"] = df[f"{market_symbol}_Corr_5"].rolling(5).mean()
                    # Correlations over 20 bars - using only past data
                    df[f"{market_symbol}_Corr_20"] = df["Close"].shift(1).rolling(20).corr(aligned_market["Close"].shift(1))
                    df[f"{market_symbol}_Corr_20_MA"] = df[f"{market_symbol}_Corr_20"].rolling(20).mean()
                    # Correlations over 60 bars - using only past data
                    df[f"{market_symbol}_Corr_60"] = df["Close"].shift(1).rolling(60).corr(aligned_market["Close"].shift(1))
                    df[f"{market_symbol}_Corr_60_MA"] = df[f"{market_symbol}_Corr_60"].rolling(60).mean()
                    # Correlations over 120 bars - using only past data
                    df[f"{market_symbol}_Corr_120"] = df["Close"].shift(1).rolling(120).corr(aligned_market["Close"].shift(1))
                    df[f"{market_symbol}_Corr_120_MA"] = df[f"{market_symbol}_Corr_120"].rolling(120).mean()
                except Exception as e:
                    print(f"Error calculating correlations for {market_symbol}: {str(e)}")
                
                try:
                    # Calculate returns for both stock and market - using only past data
                    market_returns = aligned_market["Close"].shift(1).pct_change()
                    stock_returns = df["Close"].shift(1).pct_change()
                    
                    # Beta calculations - using only past data
                    df[f"{market_symbol}_Beta_60"] = (
                        stock_returns.rolling(60).cov(market_returns) / 
                        market_returns.rolling(60).var()
                    )
                    df[f"{market_symbol}_Beta_20"] = (
                        stock_returns.rolling(20).cov(market_returns) / 
                        market_returns.rolling(20).var()
                    )
                    df[f"{market_symbol}_Beta_120"] = (
                        stock_returns.rolling(120).cov(market_returns) / 
                        market_returns.rolling(120).var()
                    )
                except Exception as e:
                    print(f"Error calculating beta for {market_symbol}: {str(e)}")
                
                try:
                    # Relative strength metrics
                    # Price of stock relative to market index
                    df[f"{market_symbol}_Rel_Strength"] = (df["Close"].shift(1) / aligned_market["Close"].shift(1))
                    # Moving average of relative strength
                    df[f"{market_symbol}_Rel_Strength_MA20"] = df[f"{market_symbol}_Rel_Strength"].rolling(20).mean()
                    # Rate of change in relative strength - using only past data
                    df[f"{market_symbol}_Rel_Strength_Change"] = df[f"{market_symbol}_Rel_Strength"].shift(1).pct_change(5)
                    
                    # Additional relative strength metrics using only past data
                    df[f"{market_symbol}_Rel_Strength_MA5"] = df[f"{market_symbol}_Rel_Strength"].shift(1).rolling(5).mean()
                    df[f"{market_symbol}_Rel_Strength_MA10"] = df[f"{market_symbol}_Rel_Strength"].shift(1).rolling(10).mean()
                    df[f"{market_symbol}_Rel_Strength_MA30"] = df[f"{market_symbol}_Rel_Strength"].shift(1).rolling(30).mean()
                    
                    # Relative strength momentum using only past data
                    df[f"{market_symbol}_Rel_Strength_Momentum"] = df[f"{market_symbol}_Rel_Strength"].shift(1) - df[f"{market_symbol}_Rel_Strength"].shift(6)
                    df[f"{market_symbol}_Rel_Strength_Acceleration"] = df[f"{market_symbol}_Rel_Strength_Momentum"].shift(1).diff(5)
                except Exception as e:
                    print(f"Error calculating relative strength for {market_symbol}: {str(e)}")
                
                try:
                    # Volatility comparison
                    # Ratio of stock volatility to market volatility
                    df[f"{market_symbol}_Vol_Ratio"] = (
                        df["Close"].shift(1).rolling(20).std() / 
                        aligned_market["Close"].shift(1).rolling(20).std()
                    )
                except Exception as e:
                    print(f"Error calculating volatility ratio for {market_symbol}: {str(e)}")
                
                try:
                    # Advanced correlation metrics
                    # Direction alignment (whether stock and market move in same direction)
                    df[f"{market_symbol}_Direction_Align"] = np.where(
                        (stock_returns > 0) == (market_returns > 0),
                        1, 0
                    )
                    # Moving average of direction alignment
                    df[f"{market_symbol}_Direction_Align_MA"] = df[f"{market_symbol}_Direction_Align"].rolling(20).mean()
                    
                    # Calculate correlation regime changes
                    df[f"{market_symbol}_Corr_Regime_Change"] = df[f"{market_symbol}_Corr_20"].diff().rolling(5).sum()
                except Exception as e:
                    print(f"Error calculating direction alignment for {market_symbol}: {str(e)}")
        
        # Additional indicators specific to VIX
        if 'VXX' in market_data and len(market_data['VXX']) >= 5:
            try:
                vix_data = market_data['VXX'].reindex(df.index, method='ffill')
                
                # Fill any missing values
                if vix_data['Close'].isna().any():
                    vix_data = vix_data.ffill().bfill()
                
                # Correlations with VIX
                df["VIX_Corr_20"] = df["Close"].shift(1).rolling(20).corr(vix_data["Close"].shift(1))
                df["VIX_Corr_60"] = df["Close"].shift(1).rolling(60).corr(vix_data["Close"].shift(1))
                
                # VIX impact on stock volatility
                df["VIX_Impact"] = df["ATR"] / vix_data["Close"].shift(1)
                df["VIX_Impact_MA"] = df["VIX_Impact"].rolling(20).mean()
                
                # VIX-based volatility regime
                vix_mean = vix_data["Close"].shift(1).rolling(20).mean()
                vix_std = vix_data["Close"].shift(1).rolling(20).std()
                
                df["VIX_Regime"] = np.where(
                    vix_data["Close"].shift(1) > vix_mean + vix_std,
                    "High Volatility",
                    np.where(
                        vix_data["Close"].shift(1) < vix_mean - vix_std,
                        "Low Volatility",
                        "Normal"
                    )
                )
                
                # VIX momentum
                df["VIX_Momentum"] = vix_data["Close"].shift(1).pct_change(5)
                
                # Stock performance during different VIX regimes
                df["VIX_Regime_Performance"] = np.where(
                    df["VIX_Regime"] == "High Volatility",
                    df["Close"].pct_change(),
                    np.where(
                        df["VIX_Regime"] == "Low Volatility",
                        df["Close"].pct_change(),
                        df["Close"].pct_change()
                    )
                )
            except Exception as e:
                print(f"Error calculating VIX indicators: {str(e)}")
    
    # Time-Sensitive Analysis (using only past data)
    df.index = pd.to_datetime(df.index)
    df["Date"] = df.index.date
    df["Time"] = df.index.time
    df["Hour"] = df.index.hour
    df["Minute"] = df.index.minute
    
    # Calculate Rolling Range instead of Opening Range
    df["Rolling_Range_High"] = df["High"].shift(1).rolling(window=30).max()
    df["Rolling_Range_Low"] = df["Low"].shift(1).rolling(window=30).min()
    df["Rolling_Range_Mid"] = (df["Rolling_Range_High"] + df["Rolling_Range_Low"]) / 2
    
    # Rolling Range Breakout Analysis
    df["RRB_Status"] = np.where(
        df["Close"].shift(1) > df["Rolling_Range_High"], "Breakout",
        np.where(
            df["Close"].shift(1) < df["Rolling_Range_Low"], "Breakdown",
            "Inside"
        )
    )
    
    df["RRB_Distance"] = (df["Close"].shift(1) - df["Rolling_Range_Mid"]) / df["Rolling_Range_Mid"]
    df["RRB_Strength"] = abs(df["Close"].shift(1) - df["Rolling_Range_Mid"]) / df["ATR"]
    
    # Rolling VWAP (using only past data)
    df["Rolling_VWAP"] = (df["Volume"].shift(1) * (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3).rolling(window=20).sum() / df["Volume"].shift(1).rolling(window=20).sum()
    df["VWAP_Distance"] = (df["Close"].shift(1) - df["Rolling_VWAP"]) / df["Rolling_VWAP"]
    df["VWAP_Distance_Std"] = df["VWAP_Distance"] / df["VWAP_Distance"].shift(1).rolling(20).std()
    df["VWAP_Mean_Reversion"] = df["VWAP_Distance_Std"].abs() > 2
    
    # Pivot Point Analysis (using only past data)
    df["Pivot_Point"] = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
    df["Support_1"] = (df["Pivot_Point"] * 2) - df["High"].shift(1)
    df["Resistance_1"] = (df["Pivot_Point"] * 2) - df["Low"].shift(1)
    df["Support_2"] = df["Pivot_Point"] - (df["High"].shift(1) - df["Low"].shift(1))
    df["Resistance_2"] = df["Pivot_Point"] + (df["High"].shift(1) - df["Low"].shift(1))
    
    # Pivot Point Strength
    df["Pivot_Strength"] = abs(df["Close"].shift(1) - df["Pivot_Point"]) / df["ATR"]
    df["Support_Strength"] = (df["Close"].shift(1) - df["Support_1"]) / df["ATR"]
    df["Resistance_Strength"] = (df["Resistance_1"] - df["Close"].shift(1)) / df["ATR"]
    
    # Rolling Volume Profile (using only past data)
    df["Price_Level"] = df["Close"].shift(1).round(2)
    df["Rolling_Volume_Profile"] = df["Volume"].shift(1).rolling(window=100).sum() / df["Volume"].shift(1).rolling(window=100).sum().max()
    
    # Sudden Liquidity Shifts with adjusted periods
    df["Volume_Shift"] = df["Volume"].shift(1) / df["Volume"].shift(1).rolling(20).mean()
    df["Price_Shift"] = df["Close"].shift(1).pct_change()
    df["Liquidity_Shift"] = df["Volume_Shift"] * df["Price_Shift"].abs()
    
    # Key Level Touches (using only past data)
    df["Touches_Support"] = (df["Low"].shift(1) <= df["Support_1"]) & (df["Close"].shift(1) > df["Support_1"])
    df["Touches_Resistance"] = (df["High"].shift(1) >= df["Resistance_1"]) & (df["Close"].shift(1) < df["Resistance_1"])
    
    # Institutional Activity Indicators
    df["Institutional_Activity"] = (
        (df["Volume_Shift"] > 2) &
        (df["VWAP_Distance_Std"].abs() > 1.5) &
        (df["Touches_Support"] | df["Touches_Resistance"])
    )
    
    # Market Session Analysis
    df["Session"] = np.where(
        df["Hour"] < 9, "Pre-Market",
        np.where(
            df["Hour"] < 16, "Regular",
            "After-Hours"
        )
    )
    
    # Historical Session Volume Ratios (using only past data)
    historical_session_volume = df.groupby("Session")["Volume"].shift(1).rolling(window=20).mean()
    df["Session_Volume_Ratio"] = df["Volume"].shift(1) / historical_session_volume
    
    # Fractal Pattern Detection (using only past data)
    df["Local_High"] = (df["High"].shift(1) > df["High"].shift(2)) & (df["High"].shift(1) > df["High"].shift(0))
    df["Local_Low"] = (df["Low"].shift(1) < df["Low"].shift(2)) & (df["Low"].shift(1) < df["Low"].shift(0))
    
    # Fractal Strength
    df["Fractal_High_Strength"] = df["High"].shift(1).rolling(5).max() - df["High"].shift(1)
    df["Fractal_Low_Strength"] = df["Low"].shift(1) - df["Low"].shift(1).rolling(5).min()
    
    # Fractal Volume Profile
    df["Fractal_High_Volume"] = df["Local_High"].astype(int) * df["Volume"].shift(1)
    df["Fractal_Low_Volume"] = df["Local_Low"].astype(int) * df["Volume"].shift(1)
    
    # Dynamic Support/Resistance (using only past data)
    df["Dynamic_Support"] = df["Low"].shift(1).rolling(20).min()
    df["Dynamic_Resistance"] = df["High"].shift(1).rolling(20).max()
    df["Support_Resistance_Range"] = df["Dynamic_Resistance"] - df["Dynamic_Support"]
    
    # Support/Resistance Break Potential
    df["SR_Break_Potential"] = (df["Close"].shift(1) - df["Dynamic_Support"]) / (df["Dynamic_Resistance"] - df["Dynamic_Support"])
    
    # Price Structure Analysis
    df["Price_Structure"] = np.where(
        df["Close"].shift(1) > df["Dynamic_Resistance"], "Breakout",
        np.where(
            df["Close"].shift(1) < df["Dynamic_Support"], "Breakdown",
            np.where(
                df["Close"].shift(1) > df["Pivot_Point"], "Bullish",
                "Bearish"
            )
        )
    )
    
    # Time-of-Day Volume Analysis (using historical averages)
    df["Time_of_Day"] = df.index.time
    historical_time_volume = df.groupby(["Hour", "Minute"])["Volume"].shift(1).rolling(window=20).mean()
    df["Time_Adjusted_Volume"] = df["Volume"].shift(1) / historical_time_volume
    df["Time_Adjusted_Volume_MA"] = df["Time_Adjusted_Volume"].shift(1).rolling(window=20).mean()
    
    # Relative Volume Analysis with adjusted periods
    df["Relative_Volume"] = df["Volume"].shift(1) / df["Volume"].shift(1).rolling(20).mean()
    df["Relative_Volume_MA"] = df["Relative_Volume"].shift(1).rolling(window=10).mean()
    df["Relative_Volume_Trend"] = df["Relative_Volume"].diff()
    
    # Large Trade Imbalance Detection
    df["Large_Trade_Imbalance"] = df["Volume"].shift(1).rolling(10).max() / df["Volume"].shift(1).rolling(50).mean()
    df["Trade_Imbalance_MA"] = df["Large_Trade_Imbalance"].shift(1).rolling(window=20).mean()
    df["Trade_Imbalance_Trend"] = df["Large_Trade_Imbalance"].diff()
    
    # Enhanced Volume Spike Detection
    df["Volume_Spike"] = df["Volume"].shift(1) > (df["Volume"].shift(1).rolling(20).mean() + 2 * df["Volume"].shift(1).rolling(20).std())
    df["Volume_Spike_Size"] = df["Volume"].shift(1) / df["Volume"].shift(1).rolling(20).mean()
    df["Volume_Spike_Duration"] = df["Volume_Spike"].astype(int).rolling(5).sum()
    
    # Tick Imbalance Analysis
    df["Price_Tick"] = df["Close"].shift(1).diff()
    df["Tick_Direction"] = np.where(df["Price_Tick"] > 0, 1, np.where(df["Price_Tick"] < 0, -1, 0))
    df["Tick_Sequence"] = df["Tick_Direction"].rolling(5).sum()
    df["Tick_Volume_Impact"] = df["Tick_Sequence"] * df["Time_Adjusted_Volume"]
    
    # Volume Price Trend
    df["VP_Trend"] = (df["Close"].shift(1) - df["Close"].shift(2)) * df["Volume"].shift(1)
    df["VP_Trend_SMA_5"] = df["VP_Trend"].shift(1).rolling(window=5).mean()
    
    # Additional Technical Indicators (using only past data)
    exp1 = df["Close"].shift(1).ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].shift(1).ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].shift(1).ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df["BB_Middle"] = df["Close"].shift(1).rolling(window=20).mean()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["Close"].shift(1).rolling(window=20).std()
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["Close"].shift(1).rolling(window=20).std()
    
    # Price momentum with adjusted periods
    df["Price_Change"] = df["Close"].shift().pct_change()
    df["Price_Change_5"] = df["Close"].pct_change(5).shift(-5)
    
    # Candlestick Direction
    df["Candle_Direction"] = np.where(df["Close"].shift(1) >= df["Open"].shift(1), 1, -1)
    
    # Candlestick Components (Absolute)
    df["Real_Body_Size"] = abs(df["Close"].shift(1) - df["Open"].shift(1))
    df["Upper_Shadow"] = df["High"].shift(1) - np.maximum(df["Open"].shift(1), df["Close"].shift(1))
    df["Lower_Shadow"] = np.minimum(df["Open"].shift(1), df["Close"].shift(1)) - df["Low"].shift(1)
    df["Total_Range"] = df["High"].shift(1) - df["Low"].shift(1)
    
    # Candlestick Components (Percentage)
    df["Real_Body_Size_Pct"] = df["Real_Body_Size"] / df["Open"].shift(1) * 100
    df["Upper_Shadow_Pct"] = df["Upper_Shadow"] / df["Open"].shift(1) * 100
    df["Lower_Shadow_Pct"] = df["Lower_Shadow"] / df["Open"].shift(1) * 100
    df["Total_Range_Pct"] = df["Total_Range"] / df["Open"].shift(1) * 100
    
    # Shadow Analysis
    df["Shadow_Ratio"] = df["Upper_Shadow"] / (df["Lower_Shadow"] + 1e-6)
    df["Body_To_Range_Ratio"] = df["Real_Body_Size"] / (df["Total_Range"] + 1e-6)
    
    # Directional Body and Shadow Impact
    df["Directional_Body"] = df["Real_Body_Size"] * df["Candle_Direction"]
    df["Directional_Body_Pct"] = df["Real_Body_Size_Pct"] * df["Candle_Direction"]
    
    # Moving averages of components
    df["Body_Size_MA5"] = df["Real_Body_Size_Pct"].shift(1).rolling(window=5).mean()
    df["Upper_Shadow_MA5"] = df["Upper_Shadow_Pct"].shift(1).rolling(window=5).mean()
    df["Lower_Shadow_MA5"] = df["Lower_Shadow_Pct"].shift(1).rolling(window=5).mean()
    
    # Volume-weighted analysis
    rel_volume = df["Volume"].shift(1) / df["Volume"].shift(1).rolling(window=5).mean()
    df["Body_Vol_Impact"] = df["Directional_Body_Pct"] * rel_volume
    df["Upper_Shadow_Vol_Impact"] = df["Upper_Shadow_Pct"] * rel_volume
    df["Lower_Shadow_Vol_Impact"] = df["Lower_Shadow_Pct"] * rel_volume
    
    # Trend Analysis
    df["Body_Size_Trend"] = df["Body_Size_MA5"].diff()
    df["Shadow_Balance"] = df["Upper_Shadow_MA5"] - df["Lower_Shadow_MA5"]
    
    # Pattern Detection (basic)
    df["Doji"] = (df["Real_Body_Size_Pct"] <= 0.1) & (df["Total_Range_Pct"] > 0.2)
    df["Marubozu"] = (df["Upper_Shadow_Pct"] <= 0.1) & (df["Lower_Shadow_Pct"] <= 0.1) & (df["Real_Body_Size_Pct"] > 0.5)
    df["Long_Upper_Shadow"] = df["Upper_Shadow_Pct"] > (2 * df["Real_Body_Size_Pct"])
    df["Long_Lower_Shadow"] = df["Lower_Shadow_Pct"] > (2 * df["Real_Body_Size_Pct"])
    
    # Support/Resistance Strength
    df["SR_Strength"] = np.abs(df["Dynamic_Support"] - df["Price_Shift"])
    
    # Trend Strength
    df["Trend_Strength"] = df["Directional_Body"] * df["Rolling_Volume_Profile"] * (1 + df["Body_Vol_Impact"])
    
    # Composite Features
    # 1. Volatility and Body Size Interaction
    df['Vol_Body_Interaction'] = df['Body_Size_MA5'] * df['Rolling_Std_10']
    
    # Enhanced Volume-Based Composite Features
    # 1. Volume Momentum Score
    df['Volume_Momentum'] = (
        df['Volume_Shift'] * 
        df['Relative_Volume'] * 
        (1 + df['Volume_Spike_Size'])
    )
    
    # 2. Volume Price Impact Score
    df['Volume_Price_Impact'] = (
        df['Volume_Shift'] * 
        abs(df['Price_Change']) * 
        df['Body_To_Range_Ratio']
    )
    
    # 3. Volume Support/Resistance Impact
    df['Volume_SR_Impact'] = (
        df['Volume_Shift'] * 
        df['Rolling_Volume_Profile'] * 
        (1 + abs(df['SR_Break_Potential']))
    )
    
    # 4. Smart Money Flow
    df['Smart_Money_Flow'] = (
        df['Volume_Shift'] * 
        df['Institutional_Activity'].astype(float) * 
        (1 + abs(df['VWAP_Distance']))
    )
    
    # 5. Volume Trend Quality
    df['Volume_Trend_Quality'] = (
        df['Volume_Shift'] * 
        df['Time_Adjusted_Volume'] * 
        (1 + abs(df['Trend_Strength']))
    )
    
    # 6. Volume Breakout Signal
    df['Volume_Breakout_Signal'] = (
        df['Volume_Shift'] * 
        (df['Close'].shift(1) > df['Dynamic_Resistance']).astype(float) * 
        df['RRB_Strength']
    )
    
    # 7. Volume Distribution Score
    df['Volume_Distribution'] = (
        df['Volume_Shift'] * 
        df['Rolling_Volume_Profile'] * 
        df['Relative_Volume']
    )
    
    # 8. Volume Volatility Impact
    df['Volume_Volatility_Impact'] = (
        df['Volume_Shift'] * 
        df['Rolling_Std_10'] * 
        (1 + abs(df['Price_Tick']))
    )
    
    # Original remaining composite features
    # 2. Enhanced Price Action Signal
    df['Price_Action_Signal'] = (df['Body_Size_MA5'] * df['Shadow_Balance']) / (df['Rolling_Std_10'] + 1e-6)
    
    # 3. Market Impact Score
    if market_data is not None:
        df['Market_Impact'] = df['DIA_Corr_20'] * df['Rolling_Volume_Profile']
    
    # 4. Support/Resistance Strength (already calculated above)
    
    # 5. Volatility Regime
    df['Volatility_Regime'] = df['Rolling_Std_10'] / df['Rolling_Std_10'].shift(1).rolling(window=20).mean()
    
    # 6. Enhanced Volume Impact
    df['Enhanced_Volume_Impact'] = df['Volume_Shift'] * df['Body_To_Range_Ratio']
    
    # 7. Trend Strength (already calculated above)
    
    # 8. Support/Resistance Break Potential (already calculated above)
    
    # Enhanced Price Action Features
    print("Calculating enhanced price action features...")
    
    # Price Structure Analysis
    df['Body_Size'] = abs(df['Close'].shift(1) - df['Open'].shift(1))
    df['Upper_Shadow'] = df['High'].shift(1) - df[['Open', 'Close']].shift(1).max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].shift(1).min(axis=1) - df['Low'].shift(1)
    df['Total_Range'] = df['High'].shift(1) - df['Low'].shift(1)
    
    # Candlestick Patterns
    df['Is_Doji'] = (df['Body_Size'] / df['Total_Range'] < 0.1).astype(int)
    df['Is_Marubozu'] = ((df['Upper_Shadow'] + df['Lower_Shadow']) / df['Total_Range'] < 0.1).astype(int)
    df['Is_Hammer'] = ((df['Lower_Shadow'] > 2 * df['Body_Size']) & 
                      (df['Upper_Shadow'] < df['Body_Size'])).astype(int)
    df['Is_Shooting_Star'] = ((df['Upper_Shadow'] > 2 * df['Body_Size']) & 
                             (df['Lower_Shadow'] < df['Body_Size'])).astype(int)
    
    # Volume Profile Analysis
    print("Calculating volume profile features...")
    
    # Volume at Price Levels
    price_levels = pd.qcut(df['Close'].shift(1), q=20, labels=False, duplicates='drop')
    df['Volume_Profile'] = df.groupby(price_levels)['Volume'].transform('sum').shift(1)
    
    # Volume Clusters
    df['Volume_Cluster_5'] = df['Volume'].shift(1).rolling(window=5).sum()
    df['Volume_Cluster_20'] = df['Volume'].shift(1).rolling(window=20).sum()
    
    # Volume Imbalance
    df['Buy_Volume'] = df['Volume'].shift(1) * (df['Close'].shift(1) > df['Open'].shift(1))
    df['Sell_Volume'] = df['Volume'].shift(1) * (df['Close'].shift(1) < df['Open'].shift(1))
    df['Volume_Imbalance'] = (df['Buy_Volume'] - df['Sell_Volume']) / (df['Buy_Volume'] + df['Sell_Volume'] + 1e-8)
    
    # Market Microstructure Features
    print("Calculating market microstructure features...")
    
    # Order Flow Analysis
    df['Price_Impact'] = (df['High'].shift(1) - df['Low'].shift(1)) / df['Volume'].shift(1)
    df['Efficiency_Ratio'] = abs(df['Close'].shift(1) - df['Open'].shift(1)) / (df['High'].shift(1) - df['Low'].shift(1) + 1e-8)
    
    # Liquidity Analysis
    df['Bid_Ask_Spread'] = (df['High'].shift(1) - df['Low'].shift(1)) / df['Close'].shift(1)
    df['Liquidity_Ratio'] = df['Volume'].shift(1) / df['Bid_Ask_Spread']
    
    # Momentum and Acceleration
    df['Price_Momentum'] = df['Close'].shift(1).pct_change(periods=5)
    df['Volume_Momentum'] = df['Volume'].shift(1).pct_change(periods=5)
    df['Momentum_Acceleration'] = df['Price_Momentum'].diff()
    
    # Support/Resistance Levels
    print("Calculating support and resistance features...")
    
    # Dynamic Support/Resistance
    df['Support_Level'] = df['Low'].shift(1).rolling(window=20).min()
    df['Resistance_Level'] = df['High'].shift(1).rolling(window=20).max()
    df['Price_to_SR_Distance'] = (df['Close'].shift(1) - df['Support_Level']) / (df['Resistance_Level'] - df['Support_Level'] + 1e-8)
    
    # Pivot Points
    df['Pivot_Point'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = 2 * df['Pivot_Point'] - df['Low'].shift(1)
    df['S1'] = 2 * df['Pivot_Point'] - df['High'].shift(1)
    
    # Trend Strength Indicators
    print("Calculating trend strength features...")
    
    # ADX and Directional Movement
    high_low = df['High'].shift(1) - df['Low'].shift(1)
    high_close = abs(df['High'].shift(1) - df['Close'].shift(2))
    low_close = abs(df['Low'].shift(1) - df['Close'].shift(2))
    
    df['+DM'] = ((df['High'].shift(1) - df['High'].shift(2)) > (df['Low'].shift(2) - df['Low'].shift(1))) & \
                ((df['High'].shift(1) - df['High'].shift(2)) > 0)
    df['-DM'] = ((df['Low'].shift(2) - df['Low'].shift(1)) > (df['High'].shift(1) - df['High'].shift(2))) & \
                ((df['Low'].shift(2) - df['Low'].shift(1)) > 0)
    
    df['+DM'] = df['+DM'] * (df['High'].shift(1) - df['High'].shift(2))
    df['-DM'] = df['-DM'] * (df['Low'].shift(2) - df['Low'].shift(1))
    
    # Composite Features
    print("Calculating composite features...")
    
    # Price-Volume Composite
    df['PV_Composite'] = df['Price_Momentum'] * df['Volume_Momentum']
    df['Trend_Quality'] = df['Price_Momentum'] * df['Efficiency_Ratio']
    
    # Volatility-Adjusted Features
    df['Volatility_Adjusted_Momentum'] = df['Price_Momentum'] / (df['Rolling_Std_20'] + 1e-8)
    df['Volume_Adjusted_Volatility'] = df['Rolling_Std_20'] * df['Volume_Momentum']
    
    # Market Regime Features
    df['Volatility_Regime'] = df['Rolling_Std_20'] / df['Rolling_Std_20'].rolling(window=100).mean()
    df['Trend_Regime'] = df['Close'].shift(1).rolling(window=50).mean() / df['Close'].shift(1).rolling(window=200).mean()
    
    # Drop NaN values from rolling calculations
    print("Before dropping NaN values:")
    print(f"Dataframe size: {len(df)}")
    print(f"NaN count per column:")
    print(df.isna().sum())
    
    # Only drop rows where all values are NaN
    df.dropna(how='all', inplace=True)
    
    # For remaining NaN values, fill them with appropriate values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            # For numerical columns, use forward fill then backward fill
            df[col] = df[col].ffill().bfill()
        elif df[col].dtype == 'bool':
            # For boolean columns, fill with False
            df[col] = df[col].fillna(False)
        elif df[col].dtype == 'object':
            # For categorical columns, fill with most frequent value
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    print("\nAfter handling NaN values:")
    print(f"Dataframe size: {len(df)}")
    print(f"NaN count per column:")
    print(df.isna().sum())
    
    print(f"Final dataframe size after indicators: {len(df)}")

    # Add HMM regime feature
    try:
        from hmmlearn.hmm import GaussianHMM
        # Use log returns for regime detection
        log_returns = np.log(df['Close']).diff().fillna(0).values.reshape(-1, 1)
        n_regimes = 3  # You can tune this
        hmm = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=1000, random_state=42)
        hmm.fit(log_returns)
        regimes = hmm.predict(log_returns)
        regime_probs = hmm.predict_proba(log_returns)
        df['Regime_Label'] = regimes
        # Add regime probabilities as features
        for i in range(n_regimes):
            df[f'Regime_Prob_{i}'] = regime_probs[:, i]
        print(f"HMM regime detection complete. Regimes: {np.unique(regimes)}")
    except Exception as e:
        print(f"Warning: HMM regime detection failed: {e}")

    return df

def fetch_data(symbol, interval="15min", outputsize="full", years=1):
    """
    Fetch historical intraday data for a symbol over a specified number of years
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        interval: Time interval (default="15min")
        outputsize: "full" or "compact" (default="full")
        years: Number of years of historical data to fetch (default=1)
    """
    # Alpha Vantage API configuration
    API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    print(f"Fetching data from {start_date} to {end_date}")
    
    # Generate list of months to fetch
    current_date = start_date
    months_to_fetch = []
    
    while current_date <= end_date:
        months_to_fetch.append(current_date.strftime("%Y-%m"))
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    print(f"Fetching {len(months_to_fetch)} months of data for {symbol} ({interval} intervals)...")
    
    # Initialize an empty list to store all data
    all_data = []
    
    # Track API calls to respect rate limit
    api_calls = 0
    last_reset_time = time.time()
    
    # Fetch data for each month
    for month in months_to_fetch:
        # Check if we need to pause to respect rate limit
        current_time = time.time()
        if api_calls >= 70:  # Leave some buffer below the 75 limit
            elapsed = current_time - last_reset_time
            if elapsed < 60:  # If less than a minute has passed
                sleep_time = 60 - elapsed
                print(f"Rate limit approaching. Pausing for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            api_calls = 0
            last_reset_time = time.time()
        
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "month": month,
            "outputsize": outputsize,
            "apikey": API_KEY
        }

        print(f"Fetching data for {month}...")
        
        try:
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            
            if f"Time Series ({interval})" not in data:
                print(f"Error: No data received for {month}. Response: {data}")
                continue
                
            # Extract the time series data
            time_series = data[f"Time Series ({interval})"]
            
            if not time_series:  # Check if time_series is empty
                print(f"Warning: Empty time series for {month}")
                continue
            
            print(f"Got {len(time_series)} data points for {month}")
            
            # Convert the data to a DataFrame
            df_chunk = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            df_chunk.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Convert string values to float
            for col in df_chunk.columns:
                df_chunk[col] = df_chunk[col].astype(float)
            
            print(f"Processed chunk for {month}: {len(df_chunk)} rows")
            
            # Add the chunk to our list
            all_data.append(df_chunk)
            
            # Increment API call counter
            api_calls += 1
            
            # Small delay to avoid hitting rate limits
            time.sleep(0.8)  # 75 calls per minute = ~0.8 seconds per call
            
        except Exception as e:
            print(f"Error fetching data for {month}: {str(e)}")
    
    if not all_data:
        raise Exception("No data was fetched successfully")
    
    print(f"Total chunks fetched: {len(all_data)}")
    
    # Combine all chunks
    df = pd.concat(all_data)
    print(f"Combined dataframe size: {len(df)}")
    
    # Sort by date
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    
    # Remove duplicates (in case of overlap between months)
    df = df[~df.index.duplicated(keep='first')]
    print(f"After removing duplicates: {len(df)}")
    
    # Get market data from the consolidated file
    print(f"Getting market data for period {start_date.date()} to {end_date.date()} from BroadMarket.csv...")
    try:
        market_data = get_market_data_for_period(start_date, end_date, interval=interval)
        
        # Handle missing or problematic data in the market dataframes
        for market_symbol, market_df in market_data.items():
            # Ensure index is datetime
            market_df.index = pd.to_datetime(market_df.index)
            
            # Fill missing values using backward fill first, then forward fill
            market_df = market_df.bfill().ffill()
            
            # If any NaN values remain, replace with zeros (should be rare)
            market_df.fillna(0, inplace=True)
            
            # Remove any duplicate indices
            market_df = market_df[~market_df.index.duplicated(keep='first')]
            
            # Update the cleaned dataframe in the dictionary
            market_data[market_symbol] = market_df
            
            print(f"Processed {market_symbol} data: {len(market_df)} rows from {market_df.index.min()} to {market_df.index.max()}")
    except Exception as e:
        print(f"Warning: Error getting market data: {str(e)}")
        print("Continuing without market data...")
        market_data = None
    
    print("Calculating technical indicators...")
    # Calculate technical indicators with market context
    df = calculate_technical_indicators(df, market_data, timeframe=interval)
    print(f"Final dataframe size after indicators: {len(df)}")
    
    return df

if __name__ == "__main__":
    try:
        # Fetch data for a specific stock
        symbol = "TSLA"
        df = fetch_data(symbol=symbol, interval="15min", outputsize="full", years=20)
        
        # Save to CSV
        df.to_csv(f"DB/{symbol}_15min_indicators.csv")
        print(f"\nSaved {len(df)} data points to DB/{symbol}_15min_indicators.csv")
        
        # Print sample of the data
        print("\nFirst few rows of the data:")
        print(df.head())
        
        # Print date range
        print(f"\nDate range: {df.index.min()} to {df.index.max()}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
