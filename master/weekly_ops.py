import os
import sys
import subprocess
import time
from datetime import datetime

def run_step(command, description):
    print(f"\n>>> STEP: {description}")
    print(f"    Command: {command}")
    start = time.time()
    result = subprocess.run(command, shell=True)
    duration = time.time() - start
    
    if result.returncode != 0:
        print(f"❌ FAILED in {duration:.1f}s")
        sys.exit(1)
    else:
        print(f"✅ COMPLETED in {duration:.1f}s")

def weekly_ops():
    print("=== FightIQ Weekly Operations Pipeline ===")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Update Odds Data (Optional/Manual for now as scraping is flaky)
    # run_step("python master/scrape_bestfightodds.py", "Update Odds Data")
    
    # 2. Run Integrity Tests
    run_step("pytest master/tests/", "Run Integrity & Regression Tests")
    
    # 3. Track Steam (Snapshot Odds)
    run_step("python master/experiment_2/track_steam.py", "Track Odds Movement (Steam)")
    
    # 4. Generate Predictions (Production Model)
    run_step("python master/predict_upcoming.py > weekly_predictions.txt", "Generate Predictions")
    
    # 5. Generate Premium Report
    run_step("python master/generate_fight_report.py", "Generate Premium PDF Report")
    
    print("\n=== 🚀 PIPELINE COMPLETE ===")
    print("Output files:")
    print("  - weekly_predictions.txt")
    print("  - full_premium_report.txt")
    print("  - master/data/odds_history/ (Snapshot)")

if __name__ == "__main__":
    weekly_ops()
