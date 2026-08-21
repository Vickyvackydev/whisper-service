import os
import subprocess
import tempfile
import urllib.parse
import ipaddress
import socket
import logging
from pathlib import Path
from typing import Optional, Tuple
import requests
import soundfile as sf
from ml_worker.config import WorkerConfig

logger = logging.getLogger("ml_worker.audio")

class AudioProcessingError(Exception):
    pass

def is_private_ip(hostname: str) -> bool:
    try:
        ip_addresses = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in ip_addresses:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified:
                return True
        return False
    except Exception:
        return True

def safe_download_audio(audio_url: str, output_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(audio_url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise AudioProcessingError(f"Unsupported URL scheme: {parsed.scheme}")
    
    if is_private_ip(parsed.hostname):
        raise AudioProcessingError(f"Access to private IP address is forbidden for host: {parsed.hostname}")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract original extension from URL path if available
    url_suffix = Path(parsed.path).suffix.lower()
    if not url_suffix or len(url_suffix) > 5 or url_suffix not in (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus"):
        url_suffix = ".audio"

    temp_file = tempfile.NamedTemporaryFile(dir=output_dir, delete=False, suffix=url_suffix)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        headers = {"User-Agent": "WhisperService-AudioFetcher/1.0"}
        with requests.get(audio_url, headers=headers, stream=True, timeout=60) as response:
            if response.status_code != 200:
                raise AudioProcessingError(f"Failed to download audio. HTTP Status: {response.status_code}")

            total_bytes = 0
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunk
                    if chunk:
                        total_bytes += len(chunk)
                        if total_bytes > WorkerConfig.MAX_AUDIO_DOWNLOAD_BYTES:
                            raise AudioProcessingError(f"Audio file exceeded maximum download size limit of {WorkerConfig.MAX_AUDIO_DOWNLOAD_BYTES} bytes")
                        f.write(chunk)

        if total_bytes == 0:
            raise AudioProcessingError("Downloaded audio file is empty")

        logger.info(f"Downloaded audio file ({total_bytes / (1024*1024):.2f} MB) to {temp_path}")
        return temp_path
    except Exception as e:
        cleanup_file(temp_path)
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Audio download error: {str(e)}") from e

def convert_to_wav_16k_mono(input_path: Path, output_dir: Path) -> Tuple[Path, float]:
    """
    Transcodes any audio input to 16kHz 16-bit mono PCM WAV.
    Uses ffmpeg if available, otherwise falls back to pure Python soundfile + numpy.
    Returns (wav_path, duration_in_seconds).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_file = tempfile.NamedTemporaryFile(dir=output_dir, delete=False, suffix=".wav")
    wav_path = Path(wav_file.name)
    wav_file.close()

    import shutil
    import numpy as np

    # 1. Try FFmpeg first if installed on system PATH
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel", "error",
            "-i", str(input_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(wav_path)
        ]

        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if process.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0:
                info = sf.info(str(wav_path))
                duration = float(info.duration)
                if duration > 0:
                    logger.info(f"Converted audio via FFmpeg: duration={duration:.2f}s, path={wav_path}")
                    return wav_path, duration
        except Exception as ffmpeg_err:
            logger.warning(f"FFmpeg attempt failed ({ffmpeg_err}). Using pure-Python soundfile...")

    # 2. Pure Python fallback using soundfile & numpy
    try:
        logger.info(f"Converting audio using pure-Python soundfile engine: {input_path.name}")
        data, sample_rate = sf.read(str(input_path), dtype="float32")

        # Convert stereo/multichannel to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Resample to 16000 Hz using high-quality Sinc interpolation
        if sample_rate != 16000:
            try:
                import torch
                import torchaudio.functional as F
                tensor_data = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
                resampled = F.resample(tensor_data, orig_freq=sample_rate, new_freq=16000)
                data = resampled.squeeze(0).numpy()
            except Exception:
                target_length = int(len(data) * 16000.0 / sample_rate)
                data = np.interp(
                    np.linspace(0, len(data), target_length, endpoint=False),
                    np.arange(len(data)),
                    data
                )

        duration = float(len(data) / 16000.0)
        if duration <= 0:
            raise AudioProcessingError("Audio duration is 0 seconds or invalid")

        if duration > WorkerConfig.MAX_AUDIO_DURATION_SECONDS:
            raise AudioProcessingError(f"Audio duration ({duration:.1f}s) exceeds max limit ({WorkerConfig.MAX_AUDIO_DURATION_SECONDS}s)")

        # Save as 16kHz 16-bit PCM WAV
        sf.write(str(wav_path), data.astype(np.float32), 16000, subtype="PCM_16")

        logger.info(f"Converted audio via SoundFile: duration={duration:.2f}s, path={wav_path}")
        return wav_path, duration

    except Exception as py_err:
        cleanup_file(wav_path)
        if isinstance(py_err, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Audio conversion failed: {str(py_err)}") from py_err

def cleanup_file(path: Optional[Path]):
    if path and path.exists():
        try:
            path.unlink()
            logger.debug(f"Cleaned up temporary file: {path}")
        except Exception as e:
            logger.warning(f"Failed to remove temp file {path}: {e}")
