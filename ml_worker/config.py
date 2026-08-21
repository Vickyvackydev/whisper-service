import os
import uuid
import socket
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class WorkerConfig:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/whisper_service?sslmode=disable")
    
    # ML Models
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") or os.path.exists("/usr/local/cuda") else "auto")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # float16, int8_float16, int8
    WHISPER_CPU_THREADS: int = int(os.getenv("WHISPER_CPU_THREADS", "4"))
    
    # Speaker Diarization
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    DIARIZATION_MODEL: str = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    ENABLE_DIARIZATION_DEFAULT: bool = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
    
    # Worker Identifiers & Concurrency
    WORKER_ID: str = os.getenv("WORKER_ID", f"gpu-worker-{socket.gethostname()}-{uuid.uuid4().hex[:6]}")
    HEARTBEAT_INTERVAL: float = float(os.getenv("HEARTBEAT_INTERVAL", "5.0"))
    POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", "1.0"))
    
    # Audio & Safety limits
    MAX_AUDIO_DURATION_SECONDS: int = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "14400"))  # 4 hours
    MAX_AUDIO_DOWNLOAD_BYTES: int = int(os.getenv("MAX_AUDIO_DOWNLOAD_BYTES", str(1024 * 1024 * 1024 * 2)))  # 2 GB
    SCRATCH_DIR: Path = Path(os.getenv("SCRATCH_DIR", os.path.join(os.path.expanduser("~"), ".whisper_scratch")))

    @classmethod
    def ensure_scratch_dir(cls):
        cls.SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
