import os
import urllib.request
import tarfile
import zipfile
import shutil

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")

def setup_glue():
    data_dir = "./data/glue_data"
    os.makedirs(data_dir, exist_ok=True)
    tar_name = "datasets.tar"
    
    if not os.path.exists(tar_name):
        download_file("https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar", tar_name)
    
    print("Extracting GLUE datasets...")
    with tarfile.open(tar_name) as tar:
        tar.extractall(path=data_dir)
    print(f"GLUE data extracted to {data_dir}")

def setup_e2e():
    data_dir = "./data/e2e_data"
    os.makedirs(data_dir, exist_ok=True)
    zip_name = "e2e-dataset.zip"
    
    if not os.path.exists(zip_name):
        download_file("https://github.com/tuetschek/e2e-dataset/releases/download/v1.0.0/e2e-dataset.zip", zip_name)
    
    print("Extracting E2E dataset...")
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Move files from subdirectory if necessary or just note path
    # The zip creates an 'e2e-dataset' folder inside.
    inner_path = os.path.join(data_dir, "e2e-dataset")
    if os.path.exists(inner_path):
        for f in os.listdir(inner_path):
            shutil.move(os.path.join(inner_path, f), data_dir)
        os.rmdir(inner_path)
        
    print(f"E2E data extracted to {data_dir}")

if __name__ == "__main__":
    print("Setting up datasets...")
    try:
        setup_glue()
        setup_e2e()
        # NLTK Downloads
        import nltk
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('punkt_tab')
        print("Setup complete.")
    except Exception as e:
        print(f"Error during setup: {e}")