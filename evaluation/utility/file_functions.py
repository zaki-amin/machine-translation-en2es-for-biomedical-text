import pandas as pd


def save_to_csv(df: pd.DataFrame, folder_path: str, hpo_id: str):
    """Saves the given dataframe for a HPO term to a CSV file in the given folder path"""
    file_path = f"{folder_path}/{hpo_id}.csv"
    df.to_csv(file_path, index=False)
    print(f"\nCSV file {file_path} has been created")
