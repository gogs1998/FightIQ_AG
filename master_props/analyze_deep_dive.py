import pandas as pd

def analyze_deep_dive():
    print("=== Master Props: Deep Dive Analysis ===")
    
    try:
        df = pd.read_csv('production_validation_results.csv')
    except:
        print("Error: production_validation_results.csv not found. Run validation first.")
        return
        
    print(f"Total Fights Analyzed: {len(df)}")
    print(f"Overall Trifecta Accuracy: {df['Correct_Trifecta'].mean():.2%}")
    
    # Group by Weight Class
    print("\n=== Accuracy by Weight Class ===")
    stats = df.groupby('Weight_Class')['Correct_Trifecta'].agg(['count', 'mean']).reset_index()
    stats.columns = ['Weight Class', 'Fights', 'Accuracy']
    stats = stats.sort_values('Accuracy', ascending=False)
    
    print(f"{'Weight Class':<25} | {'Fights':<10} | {'Accuracy':<10}")
    print("-" * 55)
    
    for i, row in stats.iterrows():
        print(f"{row['Weight Class']:<25} | {row['Fights']:<10} | {row['Accuracy']:.2%}")

if __name__ == "__main__":
    analyze_deep_dive()
