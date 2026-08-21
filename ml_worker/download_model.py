import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def download_model(model_name: str = "medium"):
    print("\n" + "=" * 60)
    print(f"  Downloading Whisper Model: '{model_name}'")
    print("=" * 60)
    
    scratch_dir = Path(os.getenv("SCRATCH_DIR", os.path.join(os.path.expanduser("~"), ".whisper_scratch")))
    target_dir = scratch_dir / "models" / "whisper"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Target Directory: {target_dir}")
    print("Connecting to Hugging Face repository...")

    try:
        from faster_whisper import download_model as fw_download
        path = fw_download(model_name, output_dir=str(target_dir))
        print("\n" + "=" * 60)
        print(f" SUCCESS: Model '{model_name}' is fully downloaded and cached!")
        print(f" Location: {path}")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else os.getenv("WHISPER_MODEL", "medium")
    download_model(model)
