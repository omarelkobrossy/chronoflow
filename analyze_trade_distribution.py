import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

def analyze_trade_distribution(csv_file):
    """
    Analyze the distribution of exit times and results from trade history
    """
    # Read the trade history CSV
    df = pd.read_csv(csv_file)
    
    # Convert exit_time to datetime
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # Extract time components
    df['exit_hour'] = df['exit_time'].dt.hour
    df['exit_minute'] = df['exit_time'].dt.minute
    df['exit_day_of_week'] = df['exit_time'].dt.day_name()
    df['exit_date'] = df['exit_time'].dt.date
    
    print("=== TRADE DISTRIBUTION ANALYSIS ===\n")
    
    # 1. Overall result distribution
    print("1. RESULT DISTRIBUTION:")
    result_counts = df['result'].value_counts()
    result_percentages = df['result'].value_counts(normalize=True) * 100
    
    for result in result_counts.index:
        count = result_counts[result]
        percentage = result_percentages[result]
        print(f"   {result}: {count} trades ({percentage:.1f}%)")
    
    print(f"\n   Total trades: {len(df)}")
    
    # 2. Exit time distribution by hour
    print("\n2. EXIT TIME DISTRIBUTION BY HOUR:")
    hour_dist = df['exit_hour'].value_counts().sort_index()
    for hour in range(24):
        count = hour_dist.get(hour, 0)
        percentage = (count / len(df)) * 100
        bar = "█" * int(percentage / 2)  # Visual bar
        print(f"   {hour:2d}:00 - {count:3d} trades ({percentage:4.1f}%) {bar}")
    
    # 3. Exit time distribution by day of week
    print("\n3. EXIT TIME DISTRIBUTION BY DAY OF WEEK:")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_dist = df['exit_day_of_week'].value_counts()
    for day in day_order:
        count = day_dist.get(day, 0)
        percentage = (count / len(df)) * 100
        bar = "█" * int(percentage / 3)  # Visual bar
        print(f"   {day:9s} - {count:3d} trades ({percentage:4.1f}%) {bar}")
    
    # 4. Result distribution by hour
    print("\n4. RESULT DISTRIBUTION BY HOUR:")
    result_by_hour = pd.crosstab(df['exit_hour'], df['result'], normalize='index') * 100
    print("   Hour | TP   | SL   | BE   | Partial TP | MAXHOLD | EOD")
    print("   -----|------|------|------|------------|---------|-----")
    for hour in range(24):
        if hour in result_by_hour.index:
            row = result_by_hour.loc[hour]
            tp_pct = row.get('TP', 0)
            sl_pct = row.get('SL', 0)
            be_pct = row.get('BE', 0)
            partial_pct = row.get('Partial TP', 0)
            maxhold_pct = row.get('MAXHOLD', 0)
            eod_pct = row.get('EOD', 0)
            print(f"   {hour:2d}:00 | {tp_pct:4.1f} | {sl_pct:4.1f} | {be_pct:4.1f} | {partial_pct:10.1f} | {maxhold_pct:7.1f} | {eod_pct:3.1f}")
    
    # 5. Clustering analysis - check if exits are clustered in time
    print("\n5. TIME CLUSTERING ANALYSIS:")
    
    # Calculate time differences between consecutive exits
    df_sorted = df.sort_values('exit_time')
    time_diffs = df_sorted['exit_time'].diff().dt.total_seconds() / 60  # in minutes
    
    # Find clusters (exits within 15 minutes of each other)
    cluster_threshold = 15  # minutes
    clusters = []
    current_cluster = []
    
    for i, diff in enumerate(time_diffs):
        if pd.isna(diff) or diff <= cluster_threshold:
            current_cluster.append(i)
        else:
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
            current_cluster = [i]
    
    if len(current_cluster) > 1:
        clusters.append(current_cluster)
    
    print(f"   Found {len(clusters)} clusters of exits within {cluster_threshold} minutes")
    print(f"   Largest cluster size: {max([len(c) for c in clusters]) if clusters else 0} trades")
    
    # 6. Analyze specific result types
    print("\n6. DETAILED ANALYSIS BY RESULT TYPE:")
    
    for result in df['result'].unique():
        result_df = df[df['result'] == result]
        print(f"\n   {result} Trades ({len(result_df)} total):")
        
        # Hour distribution for this result
        hour_dist_result = result_df['exit_hour'].value_counts().sort_index()
        peak_hours = hour_dist_result.nlargest(3)
        print(f"     Peak hours: {', '.join([f'{h}:00 ({c} trades)' for h, c in peak_hours.items()])}")
        
        # Day distribution for this result
        day_dist_result = result_df['exit_day_of_week'].value_counts()
        peak_days = day_dist_result.nlargest(3)
        print(f"     Peak days: {', '.join([f'{d} ({c} trades)' for d, c in peak_days.items()])}")
        
        # Profit analysis
        if 'profit' in result_df.columns:
            avg_profit = result_df['profit'].mean()
            print(f"     Average profit: ${avg_profit:.2f}")
    
    # 7. Create visualizations
    print("\n7. CREATING VISUALIZATIONS...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Trade Exit Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Exit time by hour
    ax1 = axes[0, 0]
    hour_counts = df['exit_hour'].value_counts().sort_index()
    ax1.bar(hour_counts.index, hour_counts.values, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Number of Exits')
    ax1.set_title('Exit Distribution by Hour')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Result distribution
    ax2 = axes[0, 1]
    result_counts = df['result'].value_counts()
    colors = ['green', 'red', 'orange', 'blue', 'purple', 'brown']
    ax2.pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%', 
            colors=colors[:len(result_counts)], startangle=90)
    ax2.set_title('Result Distribution')
    
    # Plot 3: Result distribution by hour (heatmap)
    ax3 = axes[1, 0]
    result_hour_pivot = pd.crosstab(df['exit_hour'], df['result'])
    sns.heatmap(result_hour_pivot.T, annot=True, fmt='d', cmap='YlOrRd', ax=ax3)
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Result Type')
    ax3.set_title('Result Distribution by Hour (Heatmap)')
    
    # Plot 4: Exit time distribution by day of week
    ax4 = axes[1, 1]
    day_counts = df['exit_day_of_week'].value_counts()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts_ordered = day_counts.reindex(day_order)
    ax4.bar(range(len(day_counts_ordered)), day_counts_ordered.values, 
            alpha=0.7, color='lightcoral')
    ax4.set_xlabel('Day of Week')
    ax4.set_ylabel('Number of Exits')
    ax4.set_title('Exit Distribution by Day of Week')
    ax4.set_xticks(range(len(day_order)))
    ax4.set_xticklabels(day_order, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('DB/trade_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("   Saved visualization to: DB/trade_distribution_analysis.png")
    
    # 8. Summary and recommendations
    print("\n8. SUMMARY AND RECOMMENDATIONS:")
    
    # Check for clustering
    if len(clusters) > 0:
        avg_cluster_size = np.mean([len(c) for c in clusters])
        print(f"   ⚠️  WARNING: Found {len(clusters)} clusters with average size {avg_cluster_size:.1f} trades")
        print("      This suggests exits may be clustered in time rather than distributed normally")
    else:
        print("   ✅ No significant clustering detected - exits appear well distributed")
    
    # Check for hour concentration
    peak_hour_count = hour_dist.max()
    peak_hour_percentage = (peak_hour_count / len(df)) * 100
    if peak_hour_percentage > 20:
        print(f"   ⚠️  WARNING: {peak_hour_percentage:.1f}% of exits occur in peak hour")
        print("      This suggests exits may be concentrated in specific time periods")
    else:
        print("   ✅ Exit times appear well distributed across hours")
    
    # Check result distribution
    tp_percentage = result_percentages.get('TP', 0)
    sl_percentage = result_percentages.get('SL', 0)
    if tp_percentage > 60:
        print(f"   ⚠️  WARNING: {tp_percentage:.1f}% of trades are Take Profits")
        print("      This may indicate take profit levels are too tight")
    elif sl_percentage > 40:
        print(f"   ⚠️  WARNING: {sl_percentage:.1f}% of trades are Stop Losses")
        print("      This may indicate stop loss levels are too tight")
    else:
        print("   ✅ Result distribution appears balanced")
    
    print(f"\nAnalysis complete! Check the visualization for detailed charts.")

if __name__ == "__main__":
    csv_file = "DB/trade_history.csv"
    analyze_trade_distribution(csv_file)
