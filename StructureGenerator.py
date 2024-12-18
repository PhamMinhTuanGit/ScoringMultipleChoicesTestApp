import os

# Define the folder and file structure
structure = {
    "multichoice_cv": {
        "multichoice_cv": [
            "__init__.py",
            "bubble_detection.py",
            "quadrilateral.py",
            "utilities.py",
            "data_structures.py",
            "visualization.py",
        ],
        "tests": [
            "__init__.py",
            "test_bubble_detection.py",
            "test_quadrilateral.py",
            "test_utilities.py",
        ],
        "examples": ["example_usage.py"],
    },
    ".": ["setup.py", "README.md"],
}

# Create folders and files
for root, subdirs in structure.items():
    for folder, files in subdirs.items() if isinstance(subdirs, dict) else [(None, subdirs)]:
        # Define path
        path = os.path.join(root, folder) if folder else root
        
        # Create directory
        os.makedirs(path, exist_ok=True)
        
        # Create files in directory
        for file in files:
            file_path = os.path.join(path, file)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    # Add a simple comment to each file for readability
                    f.write(f"# {file}\n")

print("Folder and file structure created successfully!")
