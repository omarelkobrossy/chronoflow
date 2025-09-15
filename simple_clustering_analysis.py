import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_exit_clustering(csv_file):
    """
    Simple analysis to check if exits are clustered in time or spread out normally
    """
    # Read the trade history CSV
    df = pd.read_csv(csv_file)
    
    # Convert exit_time to datetime
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # Sort by exit time
    df_sorted = df.sort_values('exit_time')
    
    print("=== EXIT CLUSTERING ANALYSIS ===\n")
    
    # Calculate time differences between consecutive exits
    time_diffs = df_sorted['exit_time'].diff().dt.total_seconds() / 60  # in minutes
    
    # Remove NaN (first trade)
    time_diffs = time_diffs.dropna()
    
    print(f"Total trades: {len(df)}")
    print(f"Time differences analyzed: {len(time_diffs)}")
    print()
    
    # Basic statistics
    print("TIME DIFFERENCES BETWEEN CONSECUTIVE EXITS:")
    print(f"  Mean: {time_diffs.mean():.1f} minutes")
    print(f"  Median: {time_diffs.median():.1f} minutes")
    print(f"  Std Dev: {time_diffs.std():.1f} minutes")
    print(f"  Min: {time_diffs.min():.1f} minutes")
    print(f"  Max: {time_diffs.max():.1f} minutes")
    print()
    
    # Distribution analysis
    print("DISTRIBUTION OF TIME DIFFERENCES:")
    print("  < 5 minutes:   ", (time_diffs < 5).sum(), f"({(time_diffs < 5).mean()*100:.1f}%)")
    print("  < 15 minutes:  ", (time_diffs < 15).sum(), f"({(time_diffs < 15).mean()*100:.1f}%)")
    print("  < 30 minutes:  ", (time_diffs < 30).sum(), f"({(time_diffs < 30).mean()*100:.1f}%)")
    print("  < 60 minutes:  ", (time_diffs < 60).sum(), f"({(time_diffs < 60).mean()*100:.1f}%)")
    print("  < 120 minutes: ", (time_diffs < 120).sum(), f"({(time_diffs < 120).mean()*100:.1f}%)")
    print("  >= 120 minutes:", (time_diffs >= 120).sum(), f"({(time_diffs >= 120).mean()*100:.1f}%)")
    print()
    
    # Clustering detection
    print("CLUSTERING ANALYSIS:")
    
    # Define different clustering thresholds
    thresholds = [5, 15, 30, 60]  # minutes
    
    for threshold in thresholds:
        clusters = []
        current_cluster = []
        
        for i, diff in enumerate(time_diffs):
            if diff <= threshold:
                current_cluster.append(i)
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [i]
        
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        cluster_trades = sum(len(c) for c in clusters)
        cluster_percentage = (cluster_trades / len(df)) * 100
        
        print(f"  Within {threshold:2d} minutes: {len(clusters):3d} clusters, {cluster_trades:4d} trades ({cluster_percentage:4.1f}%)")
        
        if clusters:
            cluster_sizes = [len(c) for c in clusters]
            print(f"    Largest cluster: {max(cluster_sizes)} trades")
            print(f"    Average cluster size: {np.mean(cluster_sizes):.1f} trades")
    
    print()
    
    # Result-specific clustering
    print("CLUSTERING BY RESULT TYPE:")
    
    for result in df['result'].unique():
        result_df = df[df['result'] == result].sort_values('exit_time')
        if len(result_df) < 2:
            continue
            
        result_diffs = result_df['exit_time'].diff().dt.total_seconds() / 60
        result_diffs = result_diffs.dropna()
        
        if len(result_diffs) == 0:
            continue
            
        # Check for clustering within 15 minutes
        clusters_15 = []
        current_cluster = []
        
        for i, diff in enumerate(result_diffs):
            if diff <= 15:
                current_cluster.append(i)
            else:
                if len(current_cluster) > 1:
                    clusters_15.append(current_cluster)
                current_cluster = [i]
        
        if len(current_cluster) > 1:
            clusters_15.append(current_cluster)
        
        cluster_trades = sum(len(c) for c in clusters_15)
        cluster_percentage = (cluster_trades / len(result_df)) * 100
        
        print(f"  {result:8s}: {len(clusters_15):2d} clusters, {cluster_trades:3d} trades ({cluster_percentage:4.1f}%)")
    
    print()
    
    # Conclusion
    print("CONCLUSION:")
    
    # Check if exits are heavily clustered
    very_close = (time_diffs < 5).mean() * 100
    close = (time_diffs < 15).mean() * 100
    moderate = (time_diffs < 60).mean() * 100
    
    if very_close > 30:
        print("  ⚠️  HEAVILY CLUSTERED: Over 30% of exits occur within 5 minutes of each other")
        print("     This suggests exits are NOT spread out normally - they're bunched together")
    elif close > 50:
        print("  ⚠️  MODERATELY CLUSTERED: Over 50% of exits occur within 15 minutes of each other")
        print("     This suggests some clustering - exits are not fully random")
    elif moderate > 80:
        print("  ⚠️  MILD CLUSTERING: Over 80% of exits occur within 1 hour of each other")
        print("     This suggests some time concentration but not severe clustering")
    else:
        print("  ✅ WELL DISTRIBUTED: Exits appear to be spread out normally over time")
        print("     No significant clustering detected")
    
    # Check for specific patterns
    if very_close > 20:
        print("\n  🔍 DETAILED ANALYSIS:")
        print("     - Many exits happen very close together (within 5 minutes)")
        print("     - This could indicate:")
        print("       * Market volatility causing multiple stops/targets to hit")
        print("       * Algorithmic trading patterns")
        print("       * News events affecting multiple positions")
        print("       * Take profit/stop loss levels that are too tight")
    
    print(f"\nAnalysis complete! Check the detailed visualization for more insights.")

if __name__ == "__main__":
    csv_file = "DB/trade_history.csv"
    analyze_exit_clustering(csv_file)
