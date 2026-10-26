import csv
import os
import pandas as pd
from LIBERO.Evaluation.vlm_evaluator import evaluate_sub_task

def run_evaluation(log_path="dataset/evaluation_log.csv", test_index=None):
    """
    Reads the evaluation log, calls the VLM for each entry without a prediction,
    and updates the log with the results.
    In test mode, it only runs on a single specified index.
    """
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return

    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(log_path)

    rows_to_process = []
    if test_index is not None:
        print(f"--- Running in TEST MODE for a single example (index: {test_index}) ---")
        # Ensure the vlm_prediction for the test row is cleared for re-evaluation
        df.loc[test_index, 'vlm_prediction'] = None 
        rows_to_process = [(test_index, df.iloc[test_index])]
    else:
        print("--- Running in FULL evaluation mode ---")
        rows_to_process = df[df['vlm_prediction'].isnull()].iterrows()


    # Iterate over rows where 'vlm_prediction' is empty or we are in test mode
    for index, row in rows_to_process:
        main_goal = row['main_goal']
        sub_task = row['sub_task']
        image_path = row['image_path']

        print(f"\n--- Evaluating Step {index} ---")
        
        # Call our VLM evaluator
        raw_response, standardized_answer = evaluate_sub_task(main_goal, sub_task, image_path)

        # Update the DataFrame with the results
        df.loc[index, 'vlm_prediction'] = standardized_answer
        
        # Overwrite the CSV with the updated DataFrame after each call
        df.to_csv(log_path, index=False)
        print(f"--- Updated log for step {index} ---")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    # For demonstration, we run on a single, clear example.
    # Index 2 should correspond to '..._step3_pick_orange_juice_1.png'
    run_evaluation(test_index=2)
