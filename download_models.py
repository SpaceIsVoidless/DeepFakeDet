import os
import requests
import shutil
from pathlib import Path

def download_file(url, filename):
    """Download a file from a URL to the specified filename"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def setup_models():
    """Download and set up model weights"""
    # Create weights directory if it doesn't exist
    weights_dir = Path("models/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Create .gitkeep file if it doesn't exist
    (weights_dir / ".gitkeep").touch(exist_ok=True)
    
    # List of model files and their download URLs (replace with actual URLs)
    models = {
        "Meso4_DF.h5": "YOUR_DOWNLOAD_URL_FOR_Meso4_DF.h5",
        "Meso4_F2F.h5": "YOUR_DOWNLOAD_URL_FOR_Meso4_F2F.h5",
        "MesoInception_DF.h5": "YOUR_DOWNLOAD_URL_FOR_MesoInception_DF.h5",
        "MesoInception_F2F.h5": "YOUR_DOWNLOAD_URL_FOR_MesoInception_F2F.h5",
        "tf_model.h5": "YOUR_DOWNLOAD_URL_FOR_tf_model.h5",
        "xception_weights_tf_dim_ordering_tf_kernels_notop.h5": "YOUR_DOWNLOAD_URL_FOR_xception_weights.h5"
    }
    
    print("Setting up model weights...")
    for filename, url in models.items():
        filepath = weights_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            try:
                download_file(url, filepath)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists, skipping download")
    
    print("\nModel setup complete!")

if __name__ == "__main__":
    setup_models()
