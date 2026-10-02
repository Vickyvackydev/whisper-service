import os
import uuid
import socket
from pathlib import Path
from typing import Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class WorkerConfig:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/whisper_service?sslmode=disable")
    
    # ML Models
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") or os.path.exists("/usr/local/cuda") else "auto")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # float16, int8_float16, int8
    WHISPER_CPU_THREADS: int = int(os.getenv("WHISPER_CPU_THREADS", "4"))
    WHISPER_INITIAL_PROMPT: str = os.getenv(
        "WHISPER_INITIAL_PROMPT",
        "Suit No. 5, 5A, Suit No. FHC/ABJ/CS/55/2024, Court 1, Order 5 Rule 6(d), Exhibit 1A, Section 4, Count 1, page 10, paragraph 3, No. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2024, 2025, 2026. Court, Plaintiff, Defendant, Counsel, Your Lordship, Milord, My Lord."
    )
    
    # Speaker Diarization
    @classmethod
    def get_hf_token(cls) -> str:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or ""
        if not token:
            hf_cache = Path.home() / ".cache" / "huggingface" / "token"
            if hf_cache.exists():
                token = hf_cache.read_text().strip()
        return token

    HF_TOKEN: str = get_hf_token.__func__(None)
    DIARIZATION_MODEL: str = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    DIARIZATION_THRESHOLD: Optional[float] = float(os.getenv("DIARIZATION_THRESHOLD", "0.70"))
    MIN_SPEAKERS: Optional[int] = int(os.getenv("MIN_SPEAKERS")) if os.getenv("MIN_SPEAKERS") else None
    MAX_SPEAKERS: Optional[int] = int(os.getenv("MAX_SPEAKERS")) if os.getenv("MAX_SPEAKERS") else None
    MIN_DURATION_ON: float = float(os.getenv("MIN_DURATION_ON", "0.20"))
    MIN_DURATION_OFF: float = float(os.getenv("MIN_DURATION_OFF", "0.35"))
    ENABLE_DIARIZATION_DEFAULT: bool = os.getenv("ENABLE_DIARIZATION_DEFAULT", "true").lower() in ("true", "1", "yes")
    
    # Worker Identifiers & Concurrency
    WORKER_ID: str = os.getenv("WORKER_ID", f"gpu-worker-{socket.gethostname()}-{uuid.uuid4().hex[:6]}")
    HEARTBEAT_INTERVAL: float = float(os.getenv("HEARTBEAT_INTERVAL", "5.0"))
    POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", "1.0"))
    
    # Audio & Safety limits
    MAX_AUDIO_DURATION_SECONDS: int = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "14400"))  # 4 hours
    MAX_AUDIO_DOWNLOAD_BYTES: int = int(os.getenv("MAX_AUDIO_DOWNLOAD_BYTES", str(1024 * 1024 * 1024 * 2)))  # 2 GB
    SCRATCH_DIR: Path = Path(os.getenv("SCRATCH_DIR", os.path.join(os.path.expanduser("~"), ".whisper_scratch")))

    # RunPod Auto-Stop & Cost Management
    IDLE_SHUTDOWN_MINUTES: int = int(os.getenv("IDLE_SHUTDOWN_MINUTES", "15"))
    RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")
    RUNPOD_POD_ID: str = os.getenv("RUNPOD_POD_ID", os.getenv("RUNPOD_POD_ID_ENV", ""))

    @classmethod
    def ensure_scratch_dir(cls):
        cls.SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
