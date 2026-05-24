"""
COMPLETE MACHINE LEARNING PIPELINE
===================================

This script runs the entire ML pipeline in sequence:
1. Data Preprocessing
2. Model Training & Evaluation
3. Interactive Prediction System

Run this after the scraper completes!
"""

import subprocess
import os
import sys

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 80)
    print(f"{'▶'} {description}")
    print("=" * 80)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True)
        if result.returncode == 0:
            print(f"✅ {description} - COMPLETED SUCCESSFULLY!")
            return True
        else:
            print(f"❌ Error running {script_name}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n" + "=" * 80)
    print("🚀 MACHINE LEARNING PIPELINE - COMPLETE WORKFLOW")
    print("=" * 80)
    
    # Check if raw data exists
    if not os.path.exists('zameen_results.csv'):
        print("\n❌ ERROR: zameen_results.csv not found!")
        print("   Please run the scraper first: python 'zameen_scraper (1).py'")
        return
    
    print("\n✅ Raw data file found: zameen_results.csv")
    
    # Step 1: Data Preprocessing
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 80)
    success_1 = run_script('data_preprocessing.py', 
                          'Task 2: Data Preprocessing (Handle missing, duplicates, encoding)')
    
    if not success_1:
        print("\n❌ Preprocessing failed. Stopping pipeline.")
        return
    
    # Step 2: Model Training
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING & EVALUATION")
    print("=" * 80)
    success_2 = run_script('model_training.py', 
                          'Task 3 & 4: Train & Evaluate 6 ML Models')
    
    if not success_2:
        print("\n❌ Model training failed. Stopping pipeline.")
        return
    
    # Step 3: Prediction System
    print("\n" + "=" * 80)
    print("STEP 3: PROPERTY PRICE PREDICTION SYSTEM")
    print("=" * 80)
    print("\n▶ Task 5: Interactive Prediction System")
    run_script('prediction_system.py', 
              'Launch interactive prediction system')
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED!")
    print("=" * 80)
    print("\n📊 Generated Files:")
    print("   ✔ preprocessed_data.csv - Cleaned dataset")
    print("   ✔ X_train.csv, X_test.csv - Training/Testing features")
    print("   ✔ y_train.csv, y_test.csv - Training/Testing prices")
    print("   ✔ model_results.csv - Model performance comparison")
    print("   ✔ best_model.pkl - Best trained model")
    print("   ✔ all_models.pkl - All 6 trained models")
    print("\n🎯 Next Steps:")
    print("   1. Review model_results.csv to see model comparison")
    print("   2. Run 'python prediction_system.py' to predict prices interactively")

if __name__ == "__main__":
    main()
