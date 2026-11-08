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
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16")  # float16, int8_float16, int8
    WHISPER_CPU_THREADS: int = int(os.getenv("WHISPER_CPU_THREADS", "4"))
    # BatchedInferencePipeline batch_size=6 on 20GB VRAM speeds up 3h-10h audio 3.5x while keeping plenty of headroom for NeMo Sortformer
    WHISPER_BATCH_SIZE: int = min(8, max(1, int(os.getenv("WHISPER_BATCH_SIZE", "6"))))
    FORCE_WHISPERX_ALIGNMENT: bool = os.getenv("FORCE_WHISPERX_ALIGNMENT", "true").lower() in ("true", "1", "yes")
    WHISPER_INITIAL_PROMPT: str = os.getenv(
        "WHISPER_INITIAL_PROMPT",
        "Official Nigerian courtroom transcript. My Lord, As the Court pleases, I'm grateful My Lord, Learned Counsel, "
        "Writ of Summons, Statement of Claim, Statement of Defence, Counterclaim, Motion on Notice, front-loaded processes, "
        "substituted service, Affidavit of Service, interpleader summons, hearing notice, Order 8 Rule 6(d), "
        "ongoing proceedings, Justice Saidu, Mr. Komolafe, adjourned for mention."
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
    DIARIZATION_BACKEND: str = os.getenv("DIARIZATION_BACKEND", "nemo").lower()  # "nemo" or "pyannote"
    NEMO_DIARIZATION_MODEL: str = os.getenv("NEMO_DIARIZATION_MODEL", "nvidia/diar_sortformer_4spk-v1")
    DIARIZATION_MODEL: str = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    DIARIZATION_THRESHOLD: Optional[float] = float(os.getenv("DIARIZATION_THRESHOLD", "0.63"))
    MIN_SPEAKERS: Optional[int] = int(os.getenv("MIN_SPEAKERS")) if os.getenv("MIN_SPEAKERS") else None
    MAX_SPEAKERS: Optional[int] = int(os.getenv("MAX_SPEAKERS")) if os.getenv("MAX_SPEAKERS") else None
    MIN_DURATION_ON: float = float(os.getenv("MIN_DURATION_ON", "0.08"))
    MIN_DURATION_OFF: float = float(os.getenv("MIN_DURATION_OFF", "0.20"))
    ENABLE_DIARIZATION_DEFAULT: bool = os.getenv("ENABLE_DIARIZATION_DEFAULT", "true").lower() in ("true", "1", "yes")
    
    # Worker Identifiers & Concurrency
    WORKER_ID: str = os.getenv("WORKER_ID", f"gpu-worker-{socket.gethostname()}-{uuid.uuid4().hex[:6]}")
    WORKER_CONCURRENCY: int = max(1, int(os.getenv("WORKER_CONCURRENCY", "2")))
    WHISPER_NUM_WORKERS: int = 1
    HEARTBEAT_INTERVAL: float = float(os.getenv("HEARTBEAT_INTERVAL", "5.0"))
    POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", "0.25"))
    
    # Audio & Safety limits (10 hours = 36000s minimum enforced even if legacy .env has 14400)
    MAX_AUDIO_DURATION_SECONDS: int = max(36000, int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "36000")))  # 10 hours
    MAX_AUDIO_DOWNLOAD_BYTES: int = max(1024 * 1024 * 1024 * 5, int(os.getenv("MAX_AUDIO_DOWNLOAD_BYTES", str(1024 * 1024 * 1024 * 5))))  # 5 GB
    SCRATCH_DIR: Path = Path(os.getenv("SCRATCH_DIR", os.path.join(os.path.expanduser("~"), ".whisper_scratch")))

    # RunPod Multi-GPU & Burst Auto-Scaling Configuration
    IS_BURST_WORKER: bool = os.getenv("IS_BURST_WORKER", "false").lower() in ("true", "1", "yes")
    IDLE_SHUTDOWN_MINUTES: int = int(os.getenv("IDLE_SHUTDOWN_MINUTES", "15" if IS_BURST_WORKER else "0"))
    BURST_QUEUE_THRESHOLD: int = max(1, int(os.getenv("BURST_QUEUE_THRESHOLD", "3")))
    # Keep secondary burst GPU locked & awake throughout active court hours (08:00 - 17:00 WAT Mon-Fri)
    # so RunPod never rents its host GPU to someone else mid-day between court sessions
    BURST_KEEP_ALIVE_COURT_HOURS: bool = os.getenv("BURST_KEEP_ALIVE_COURT_HOURS", "true").lower() in ("true", "1", "yes")
    COURT_START_HOUR_WAT: int = int(os.getenv("COURT_START_HOUR_WAT", "8"))
    COURT_END_HOUR_WAT: int = int(os.getenv("COURT_END_HOUR_WAT", "17"))
    BURST_AUTO_PROVISION_FALLBACK: bool = os.getenv("BURST_AUTO_PROVISION_FALLBACK", "true").lower() in ("true", "1", "yes")
    RUNPOD_NETWORK_VOLUME_ID: str = os.getenv("RUNPOD_NETWORK_VOLUME_ID", "")
    RUNPOD_IMAGE_NAME: str = os.getenv("RUNPOD_IMAGE_NAME", "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04")
    RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")
    RUNPOD_POD_ID: str = os.getenv("RUNPOD_POD_ID", os.getenv("RUNPOD_POD_ID_ENV", ""))
    RUNPOD_BURST_POD_ID: str = os.getenv("RUNPOD_BURST_POD_ID", "")

    @classmethod
    def ensure_scratch_dir(cls):
        cls.SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
