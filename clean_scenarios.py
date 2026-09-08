import os
import shutil

# Path to the Scenarios folder
SCENARIOS_PATH = "Scenarios"

def remove_modified_folders(base_path):
    """
    Recursively remove all 'Modified' folders under the given base path.
    """
    removed_count = 0
    for root, dirs, files in os.walk(base_path, topdown=False):
        for d in dirs:
            if d == "Modified":
                folder_path = os.path.join(root, d)
                shutil.rmtree(folder_path)
                print(f"Removed: {folder_path}")
                removed_count += 1
    print(f"\nTotal 'Modified' folders removed: {removed_count}")

if __name__ == "__main__":
    remove_modified_folders(SCENARIOS_PATH)
