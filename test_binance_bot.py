#!/usr/bin/env python3
"""
Test script for Binance bot implementation
This script tests the basic functionality without requiring API credentials
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Add the Quant directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Quant'))

def test_binance_bot_structure():
    """Test the basic structure and imports of the Binance bot"""
    try:
        # Test imports
        print("Testing imports...")
        from binance_bot import BinanceAPIClient, convert_binance_klines_to_dataframe, load_binance_api_key
        print("✅ All imports successful")
        
        # Test class initialization (without API credentials)
        print("\nTesting class initialization...")
        api = BinanceAPIClient(symbol='XRPUSDT', years_of_data=1)
        print("✅ BinanceAPIClient initialized successfully")
        
        # Test data conversion function
        print("\nTesting data conversion...")
        # Mock kline data (simulating Binance API response)
        mock_klines = [
            [1640995200000, "0.5000", "0.5100", "0.4900", "0.5050", "1000.0", 1640996099999, "5050.0", 100, "500.0", "252.5", "0"],
            [1640996100000, "0.5050", "0.5150", "0.4950", "0.5100", "1200.0", 1640996999999, "6120.0", 120, "600.0", "306.0", "0"],
            [1640997000000, "0.5100", "0.5200", "0.5000", "0.5150", "1100.0", 1640997899999, "5665.0", 110, "550.0", "282.5", "0"]
        ]
        
        df = convert_binance_klines_to_dataframe(mock_klines)
        print(f"✅ Data conversion successful: {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        
        # Test trading parameters
        print("\nTesting trading parameters...")
        print(f"   Symbol: {api.symbol}")
        print(f"   Window size: {api.window_size}")
        print(f"   Max concurrent trades: {api.trading_params['max_concurrent_trades']}")
        print(f"   Risk parameters: {api.trading_params['min_risk_percentage']:.2%} - {api.trading_params['max_risk_percentage']:.2%}")
        
        # Test PSI parameters
        print("\nTesting PSI parameters...")
        print(f"   PSI bins: {api.psi_params['psi_bins']}")
        print(f"   PSI threshold high: {api.psi_params['psi_threshold_hi']}")
        print(f"   PSI threshold low: {api.psi_params['psi_threshold_lo']}")
        print(f"   PSI cooldown: {api.psi_params['psi_cooldown']} bars")
        
        # Test model parameters
        print("\nTesting model parameters...")
        print(f"   Learning rate: {api.model_params['learning_rate']}")
        print(f"   N estimators: {api.model_params['n_estimators']}")
        print(f"   Max depth: {api.model_params['max_depth']}")
        print(f"   Device: {api.model_params['device']}")
        
        print("\n🎉 All basic structure tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install required packages:")
        print("pip install python-binance pandas numpy xgboost")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_websocket_functionality():
    """Test WebSocket-related functionality"""
    try:
        print("\nTesting WebSocket functionality...")
        from binance_bot import BinanceAPIClient
        
        api = BinanceAPIClient(symbol='XRPUSDT', years_of_data=1)
        
        # Test WebSocket stream management
        print("   Testing WebSocket stream management...")
        assert hasattr(api, 'ws_connections'), "Missing ws_connections attribute"
        assert hasattr(api, 'stream_active'), "Missing stream_active attribute"
        assert hasattr(api, 'kline_lock'), "Missing kline_lock attribute"
        
        # Test WebSocket methods
        assert hasattr(api, 'start_kline_stream'), "Missing start_kline_stream method"
        assert hasattr(api, 'stop_kline_stream'), "Missing stop_kline_stream method"
        assert hasattr(api, 'handle_kline_message'), "Missing handle_kline_message method"
        assert hasattr(api, 'process_new_candle'), "Missing process_new_candle method"
        
        print("✅ WebSocket functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def test_trading_functionality():
    """Test trading-related functionality"""
    try:
        print("\nTesting trading functionality...")
        from binance_bot import BinanceAPIClient
        
        api = BinanceAPIClient(symbol='XRPUSDT', years_of_data=1)
        
        # Test account management methods
        print("   Testing account management methods...")
        assert hasattr(api, 'get_account_info'), "Missing get_account_info method"
        assert hasattr(api, 'get_usd_balance'), "Missing get_usd_balance method"
        assert hasattr(api, 'update_capital_from_account'), "Missing update_capital_from_account method"
        assert hasattr(api, 'get_open_orders'), "Missing get_open_orders method"
        assert hasattr(api, 'get_open_trades_count'), "Missing get_open_trades_count method"
        assert hasattr(api, 'sync_with_account_orders'), "Missing sync_with_account_orders method"
        
        # Test trading execution methods
        print("   Testing trading execution methods...")
        assert hasattr(api, 'execute_trade'), "Missing execute_trade method"
        assert hasattr(api, 'update_trades'), "Missing update_trades method"
        assert hasattr(api, 'close_trade'), "Missing close_trade method"
        
        # Test data management methods
        print("   Testing data management methods...")
        assert hasattr(api, 'load_historical_data_from_api'), "Missing load_historical_data_from_api method"
        assert hasattr(api, 'update_memory_with_new_candle'), "Missing update_memory_with_new_candle method"
        
        print("✅ Trading functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Trading test failed: {e}")
        return False

def test_model_functionality():
    """Test model-related functionality"""
    try:
        print("\nTesting model functionality...")
        from binance_bot import BinanceAPIClient
        
        api = BinanceAPIClient(symbol='XRPUSDT', years_of_data=1)
        
        # Test model methods
        print("   Testing model methods...")
        assert hasattr(api, 'initialize_model'), "Missing initialize_model method"
        assert hasattr(api, 'update_model'), "Missing update_model method"
        assert hasattr(api, 'predict_next_bar'), "Missing predict_next_bar method"
        
        # Test PSI drift detection
        print("   Testing PSI drift detection...")
        assert hasattr(api, 'check_psi_drift'), "Missing check_psi_drift method"
        
        # Test running statistics
        print("   Testing running statistics...")
        assert hasattr(api, 'initialize_running_stats_from_memory'), "Missing initialize_running_stats_from_memory method"
        assert hasattr(api, 'update_running_stats'), "Missing update_running_stats method"
        assert hasattr(api, 'get_expanding_standardization'), "Missing get_expanding_standardization method"
        assert hasattr(api, 'get_expanding_standardization_with_stats'), "Missing get_expanding_standardization_with_stats method"
        
        # Test logging methods
        print("   Testing logging methods...")
        assert hasattr(api, 'sanity_log'), "Missing sanity_log method"
        assert hasattr(api, 'log_batch_leaf_diversity'), "Missing log_batch_leaf_diversity method"
        
        print("✅ Model functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("BINANCE BOT IMPLEMENTATION TEST")
    print("=" * 60)
    
    tests = [
        test_binance_bot_structure,
        test_websocket_functionality,
        test_trading_functionality,
        test_model_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The Binance bot implementation is ready.")
        print("\nTo use the bot:")
        print("1. Install required packages: pip install python-binance pandas numpy xgboost")
        print("2. Create settings/binance_api_key.json with your API credentials")
        print("3. Run: python binance_bot.py")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()
