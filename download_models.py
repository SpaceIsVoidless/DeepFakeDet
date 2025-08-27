import os
import requests
import shutil
from pathlib import Path
from tqdm import tqdm

def download_file(url, filename):
    """Download a file from a URL to the specified filename with progress bar"""
    try:
        # Set a timeout of 30 seconds for the request
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            # Initialize progress bar
            progress_bar = tqdm(
                total=total_size, 
                unit='iB', 
                unit_scale=True,
                desc=f"Downloading {filename.name}",
                ncols=100
            )
            
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        progress_bar.update(len(chunk))
                        f.write(chunk)
            
            progress_bar.close()
            return True
            
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading {filename}: {e}")
        # Clean up partially downloaded file if it exists
        if os.path.exists(filename):
            os.remove(filename)
        return False

def setup_models():
    """Download and set up model weights"""
    # Create weights directory if it doesn't exist
    weights_dir = Path("models/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Create .gitkeep file if it doesn't exist
    (weights_dir / ".gitkeep").touch(exist_ok=True)
    
    # List of model files and their download URLs
    models = {
        "Meso4_DF.h5": "https://github.com/ShikhaVishwakarma/Deepfake-Detection/raw/main/weights/Meso4_DF.h5",
        # Using direct download links for other models
        "Meso4_F2F.h5": "https://github.com/SpaceIsVoidless/DeepFakeDet/raw/main/Meso4_F2F.h5",
        "MesoInception_DF.h5": "https://github.com/SpaceIsVoidless/DeepFakeDet/raw/main/MesoInception_DF.h5",
        "MesoInception_F2F.h5": "https://github.com/SpaceIsVoidless/DeepFakeDet/raw/main/MesoInception_F2F.h5",
        # Using alternative sources for other models
        "xception_weights_tf_dim_ordering_tf_kernels_notop.h5": "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
    }
    
    print("\n=== Starting Model Weights Download ===")
    print(f"Downloading to: {weights_dir.absolute()}\n")
    
    success_count = 0
    total_models = len(models)
    
    for filename, url in models.items():
        filepath = weights_dir / filename
        if not filepath.exists():
            print(f"\n{'='*50}")
            print(f"Processing: {filename}")
            print(f"From: {url}")
            print(f"To: {filepath.absolute()}")
            print(f"{'='*50}\n")
            
            if download_file(url, filepath):
                success_count += 1
                print(f"✅ Successfully downloaded {filename}")
            else:
                print(f"❌ Failed to download {filename}")
        else:
            success_count += 1
            print(f"✅ {filename} already exists, skipping download")
    
    print("\n" + "="*50)
    print(f"Download Summary:")
    print(f"Total models: {total_models}")
    print(f"Successfully downloaded/verified: {success_count}")
    print(f"Failed: {total_models - success_count}")
    
    if success_count < total_models:
        print("\n⚠️  Some models failed to download. The application might not work correctly.")
        print("   Please check the error messages above and try again later.")
    else:
        print("\n🎉 All models are ready to use!")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    setup_models()
