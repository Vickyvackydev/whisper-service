import os
import sys
import time
import argparse
import logging
from pathlib import Path
import psutil
import torch

from ml_worker.config import WorkerConfig
from ml_worker.audio import convert_to_wav_16k_mono, safe_download_audio, cleanup_file
from ml_worker.transcriber import Transcriber
from ml_worker.diarizer import SpeakerDiarizer
from ml_worker.pipeline import InferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

def run_benchmark(audio_input: str, enable_diarization: bool = True, mode: str = "fast", iterations: int = 1):
    print("\n" + "=" * 60)
    print("      WHISPER SERVICE ML PERFORMANCE BENCHMARK       ")
    print("=" * 60)

    # 1. System & GPU Info
    gpu_name = "N/A (CPU Mode)"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"GPU Device: {gpu_name} (Total VRAM: {vram_total:.0f} MB)")
    else:
        print("Running in CPU mode (No CUDA detected)")

    print(f"Whisper Model: {WorkerConfig.WHISPER_MODEL}")
    print(f"Compute Type: {WorkerConfig.WHISPER_COMPUTE_TYPE}")
    print(f"Transcription Mode: {mode}")
    print(f"Diarization Enabled: {enable_diarization}")
    print(f"Iterations: {iterations}")
    print("=" * 60 + "\n")

    # 2. Prepare audio
    WorkerConfig.ensure_scratch_dir()
    if audio_input.startswith("http://") or audio_input.startswith("https://"):
        print(f"Downloading benchmark audio from: {audio_input}")
        raw_audio = safe_download_audio(audio_input, WorkerConfig.SCRATCH_DIR)
        wav_path, audio_duration = convert_to_wav_16k_mono(raw_audio, WorkerConfig.SCRATCH_DIR)
        cleanup_file(raw_audio)
    else:
        local_path = Path(audio_input)
        if not local_path.exists():
            print(f"Error: Local audio file {audio_input} not found!")
            sys.exit(1)
        wav_path, audio_duration = convert_to_wav_16k_mono(local_path, WorkerConfig.SCRATCH_DIR)

    print(f"Audio Duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)\n")

    # 3. Model Initialization (Measured once)
    init_start = time.time()
    transcriber = Transcriber()
    transcriber.load_model()
    diarizer = SpeakerDiarizer()
    if enable_diarization:
        diarizer.load_model()
    init_time = time.time() - init_start
    print(f"Models Initialized in: {init_time:.2f}s\n")

    pipeline = InferencePipeline(transcriber, diarizer)

    # 4. Benchmark Runs
    results = []
    process = psutil.Process(os.getpid())

    for i in range(1, iterations + 1):
        print(f"--- Benchmark Run {i}/{iterations} ---")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)

        cpu_start = process.cpu_percent()
        ram_start = process.memory_info().rss / (1024 * 1024)
        run_start = time.time()

        res = pipeline.process(
            job_id=f"bench-{i}",
            audio_url=f"file://{wav_path}" if not audio_input.startswith("http") else audio_input,
            enable_diarization=enable_diarization,
            transcription_mode=mode
        )

        run_time = time.time() - run_start
        rtf = run_time / audio_duration

        vram_peak = 0
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

        ram_end = process.memory_info().rss / (1024 * 1024)

        results.append({
            "run": i,
            "processing_time": run_time,
            "rtf": rtf,
            "vram_mb": vram_peak,
            "ram_mb": ram_end,
            "word_count": res["word_count"],
            "speakers": res["num_speakers"],
            "language": res["language"]
        })

        print(f"Run {i}: Time = {run_time:.2f}s | RTF = {rtf:.3f} | Peak VRAM = {vram_peak:.0f} MB | Words = {res['word_count']} | Speakers = {res['num_speakers']}")

    cleanup_file(wav_path)

    # 5. Summary Table
    avg_time = sum(r["processing_time"] for r in results) / len(results)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    avg_vram = sum(r["vram_mb"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("                    BENCHMARK RESULTS SUMMARY                   ")
    print("=" * 60)
    print(f"Audio Duration:        {audio_duration:.2f}s ({audio_duration/60:.2f} mins)")
    print(f"Avg Processing Time:   {avg_time:.2f}s")
    print(f"Real-Time Factor (RTF):{avg_rtf:.4f}  (1 hour audio processed in {3600*avg_rtf:.1f}s)")
    if torch.cuda.is_available():
        print(f"Peak VRAM Usage:       {avg_vram:.0f} MB")
    print(f"Words Detected:        {results[0]['word_count']}")
    print(f"Speakers Identified:   {results[0]['speakers']}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Service ML Inference Benchmark")
    parser.add_argument("audio", help="Path to local audio file or HTTP(S) URL")
    parser.add_argument("--no-diarize", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--mode", default="fast", choices=["fast", "accurate", "balanced"], help="Transcription mode")
    parser.add_argument("--iterations", type=int, default=1, help="Number of benchmark iterations")
    args = parser.parse_args()

    run_benchmark(
        audio_input=args.audio,
        enable_diarization=not args.no_diarize,
        mode=args.mode,
        iterations=args.iterations
    )
