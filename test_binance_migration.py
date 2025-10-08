#!/usr/bin/env python3
"""
Test script to verify the migration from python-binance to binance-connector
"""

import sys
import os
from binance_bot import BinanceAPIClient, load_binance_api_key

def test_client_initialization():
    """Test that the client can be initialized without errors"""
    print("Testing client initialization...")
    
    # Load API credentials
    api_key, api_secret = load_binance_api_key()
    
    if not api_key or not api_secret:
        print("❌ No API credentials found. Please ensure settings/binance_api_key.json exists")
        return False
    
    try:
        # Initialize client
        client = BinanceAPIClient(api_key=api_key, api_secret=api_secret, symbol='XRPUSDC', years_of_data=0.1)
        print("✅ Client initialized successfully")
        
        # Test basic API calls
        print("Testing basic API calls...")
        
        # Test account info
        account_info = client.get_account_info()
        if account_info:
            print("✅ Account info retrieved successfully")
        else:
            print("❌ Failed to retrieve account info")
            return False
        
        # Test exchange info
        client.load_symbol_info()
        if client.symbol_info:
            print("✅ Symbol info loaded successfully")
        else:
            print("❌ Failed to load symbol info")
            return False
        
        # Test trading fees
        client.fetch_trading_fees()
        print(f"✅ Trading fees: Maker={client.maker_fee:.4f}, Taker={client.taker_fee:.4f}")
        
        # Test open orders
        open_orders = client.get_open_orders()
        print(f"✅ Open orders retrieved: {len(open_orders)} orders")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during client initialization: {e}")
        return False

def test_historical_data():
    """Test historical data loading"""
    print("\nTesting historical data loading...")
    
    api_key, api_secret = load_binance_api_key()
    if not api_key or not api_secret:
        print("❌ No API credentials found")
        return False
    
    try:
        client = BinanceAPIClient(api_key=api_key, api_secret=api_secret, symbol='XRPUSDC', years_of_data=0.1)
        
        # Load a small amount of historical data
        success = client.load_historical_data_from_api()
        if success:
            print(f"✅ Historical data loaded: {len(client.raw_data)} bars")
            print(f"   Date range: {client.raw_data['Date'].min()} to {client.raw_data['Date'].max()}")
            return True
        else:
            print("❌ Failed to load historical data")
            return False
            
    except Exception as e:
        print(f"❌ Error loading historical data: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Testing Binance Connector Migration ===")
    
    # Test 1: Client initialization
    test1_passed = test_client_initialization()
    
    # Test 2: Historical data loading
    test2_passed = test_historical_data()
    
    # Summary
    print(f"\n=== Test Results ===")
    print(f"Client initialization: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Historical data loading: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Migration to binance-connector successful!")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
