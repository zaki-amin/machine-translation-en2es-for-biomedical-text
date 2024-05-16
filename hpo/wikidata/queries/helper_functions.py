import json
import os
import re
import threading


def text_cleaning(term):
    if term:
        pattern = r"[^A-Za-z0-9áéíóúüÁÉÍÓÚÜñ\s]"
        cleaned_text = re.sub(pattern, " ", " ".join(term.split()).lower())
        return " ".join(cleaned_text.split())


def save_json_to_file(data, filepath):
    with open(filepath, "w") as jsonfile:
        json.dump(data, jsonfile, indent=4)


def read_json(filepath):
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
    return data


def create_batches(data, batch_size=100):
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def path_constructor(root_directory, data_file, script_file):
    data_path = os.path.join(root_directory, "data", "interim", data_file)
    script_path = os.path.join(root_directory, "src", "data", script_file)
    return data_path, script_path


def save_data_with_integrity(data, filename):
    """
    Save data to a JSON file with enhanced data integrity features.

    Args:
        data (list or dict): Data to be saved as JSON.
        filename (str): Name of the JSON file to save data to .
    """

    # Create a lock to ensure file integrity when writing
    lock = threading.Lock()

    # Acquire the lock to ensure exclusive access during file operations
    with lock:
        # Write data to a temporary file
        temp_filename = f"{filename}.temp"
        save_json_to_file(data, temp_filename)

        # Atomically replace the existing file with the temporary file
        os.rename(temp_filename, filename)


def save_to_text(data, file_name):
    with open(file_name, "w") as f:
        for row in data:
            f.write(row + "\n")
