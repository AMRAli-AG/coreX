import os
import zipfile

def create_kaggle_zip():
    zip_filename = "corex_v2_code.zip"
    source_dir = "Version_2_Hybrid"
    
    print(f"Creating clean codebase zip: '{zip_filename}' from '{source_dir}'...")
    
    # Exclude patterns to keep the zip file light
    exclude_dirs = {
        "__pycache__", 
        ".ipynb_checkpoints", 
        "results", 
        "model_coreX_v2_optimized", 
        "test_train_1_epoch", 
        "dashboard", # Dashboard not required on Kaggle
        "data" # We will upload the raw data separately
    }
    
    exclude_files = {
        "rtde_data.csv",
        "robot_data.xlsx",
        "RobotArm_final_report.png"
    }

    count = 0
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                if file.endswith(('.pyc', '.pyo', '.git')):
                    continue
                    
                file_path = os.path.join(root, file)
                # Keep directory structure inside the zip
                zipf.write(file_path, file_path)
                count += 1
                
    print(f"[OK] Success! Packed {count} files into '{os.path.abspath(zip_filename)}'")
    print("This zip file is now ready to be uploaded to Kaggle as a dataset.")

if __name__ == "__main__":
    create_kaggle_zip()
