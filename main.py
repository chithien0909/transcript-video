import os
import time
import argparse
import subprocess
import json
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------
# Config
# ------------------------------
HARDCODED_FOLDER = r"C:\Users\HP\Documents\docs\SAA-K22\videos"

LANG_MODELS = {
    "en": {"backend": "openai-whisper", "models": ["tiny", "base", "small", "medium", "large"]},
    "vi": {"backend": "pho-whisper", "models": [
        "vinai/PhoWhisper-tiny",
        "vinai/PhoWhisper-base",
        "vinai/PhoWhisper-small",
        "vinai/PhoWhisper-medium",
        "vinai/PhoWhisper-large-v3",
    ]},
    "ja": {"backend": "hf-whisper", "models": ["openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v3"]},
    "ko": {"backend": "hf-whisper", "models": ["openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v3"]},
    "zh": {"backend": "hf-whisper", "models": ["openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v3"]},
}

# ------------------------------
# Utility
# ------------------------------
def format_srt_timestamp(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(path: str, segments: List[Tuple[float, float, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(segments, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            f.write(text.strip() + "\n\n")

def write_json(path: str, segments: List[Tuple[float, float, str]]):
    data = [{"start": s, "end": e, "text": t.strip()} for s, e, t in segments]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_vtt(path: str, segments: List[Tuple[float, float, str]]):
    def _fmt(seconds: float) -> str:
        ts = format_srt_timestamp(seconds)  # HH:MM:SS,mmm
        return ts.replace(",", ".")  # HH:MM:SS.mmm
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for start, end, text in segments:
            f.write(f"{_fmt(start)} --> {_fmt(end)}\n{text.strip()}\n\n")

def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def resolve_ffmpeg_path() -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    local_ffmpeg = os.path.join(project_root, "tools", "ffmpeg", "bin", "ffmpeg.exe")
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return "ffmpeg"

def extract_wav(input_file: str) -> str:
    base = os.path.splitext(input_file)[0]
    wav_path = base + ".wav"
    if os.path.exists(wav_path):
        return wav_path
    ffmpeg_bin = resolve_ffmpeg_path()
    cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error",
           "-i", input_file, "-ac", "1", "-ar", "16000", wav_path]
    subprocess.run(cmd, check=True)
    return wav_path

# ------------------------------
# Subtitle splitting
# ------------------------------
def split_for_subtitles(segments, max_len=8.0):
    new_segments = []
    for start, end, text in segments:
        duration = end - start
        if duration <= max_len:
            new_segments.append((start, end, text))
        else:
            words = text.split()
            chunk_start = start
            current_text = []
            approx_time = duration / len(words)
            for i, w in enumerate(words, 1):
                current_text.append(w)
                if (i * approx_time) >= max_len or i == len(words):
                    chunk_end = chunk_start + (len(current_text) * approx_time)
                    new_segments.append((chunk_start, chunk_end, " ".join(current_text)))
                    chunk_start = chunk_end
                    current_text = []
    return new_segments

# ------------------------------
# Backends
# ------------------------------
def load_backend(backend: str, model_name: str, allow_downloads: bool):
    if backend == "openai-whisper":
        import whisper, torch
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        return whisper.load_model(model_name, device="cpu")

    if backend == "pho-whisper":
        # detect if it's ct2 or not
        if model_name.endswith("-ct2"):
            from faster_whisper import WhisperModel
            return ("pho-ct2", WhisperModel(model_name, device="cpu", compute_type="int8",
                                            local_files_only=not allow_downloads))
        else:
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            import torch
            processor = AutoProcessor.from_pretrained(model_name, local_files_only=not allow_downloads)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, dtype=torch.float32,
                                                              local_files_only=not allow_downloads).to("cpu")
            return ("pho-hf", (processor, model))

    if backend == "hf-whisper":
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        import torch
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=not allow_downloads)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=torch.float32, local_files_only=not allow_downloads
        ).to("cpu")
        return ("hf", (processor, model))

    raise ValueError("Unknown backend")

# ------------------------------
# Transcription
# ------------------------------
def transcribe_openai(model, file_path: str, lang: str = None):
    result = model.transcribe(file_path, language=lang or None, fp16=False,
                              condition_on_previous_text=False, temperature=0.0, verbose=False)
    segments = [(seg["start"], seg["end"], seg["text"]) for seg in result.get("segments", [])]
    return result["text"].strip(), segments

def transcribe_pho_ct2(model, file_path: str, lang: str = None):
    segments, _ = model.transcribe(file_path, language=lang or None, beam_size=1,
                                   vad_filter=True, condition_on_previous_text=False)
    segs = [(s.start, s.end, s.text) for s in segments]
    return "".join(t for _, _, t in segs).strip(), segs

def transcribe_pho_hf(processor, model, file_path: str, lang: str = None, task: str = "transcribe"):
    import librosa, torch
    wav_path = extract_wav(file_path)
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to("cpu")
    attention_mask = inputs.get("attention_mask", None)
    gen_kwargs = {"task": task}
    if lang:
        gen_kwargs["language"] = lang
    with torch.no_grad():
        if attention_mask is not None:
            ids = model.generate(input_features, attention_mask=attention_mask, **gen_kwargs)
        else:
            ids = model.generate(input_features, **gen_kwargs)
    decoded = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return decoded.strip(), None

def transcribe_hf(processor, model, file_path: str, lang: str = None, task: str = "transcribe"):
    import librosa, torch
    wav_path = extract_wav(file_path)
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to("cpu")
    attention_mask = inputs.get("attention_mask", None)
    gen_kwargs = {"task": task}
    if lang:
        gen_kwargs["language"] = lang
    with torch.no_grad():
        if attention_mask is not None:
            ids = model.generate(input_features, attention_mask=attention_mask, **gen_kwargs)
        else:
            ids = model.generate(input_features, **gen_kwargs)
    decoded = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return decoded.strip(), None

def transcribe(backend, model_data, file_path, lang):
    if backend == "openai-whisper":
        return transcribe_openai(model_data, file_path, lang)
    elif backend == "pho-whisper":
        mode, data = model_data
        if mode == "pho-ct2":
            return transcribe_pho_ct2(data, file_path, lang)
        else:
            processor, hf_model = data
            return transcribe_pho_hf(processor, hf_model, file_path, lang)
    elif backend == "hf-whisper":
        mode, data = model_data
        processor, hf_model = data
        return transcribe_hf(processor, hf_model, file_path, lang)

# ------------------------------
# Worker
# ------------------------------
def process_file(file_path: str, backend: str, model_data, lang: str,
                 save_srt=False, save_json=False, save_vtt=False, max_sub_len: float = 8.0):
    video_dir = os.path.dirname(file_path)
    out_dir = os.path.join(video_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    base = os.path.join(out_dir, base_name)
    fname = os.path.basename(file_path)
    print(f"🎬 Processing {fname} ...")

    text, segments = transcribe(backend, model_data, file_path, lang)
    print(f"Segments size: {len(segments) if segments else 0}")

    # Always save TXT transcript
    write_text(base + ".txt", text)

    if segments:
        segments = split_for_subtitles(segments, max_len=max_sub_len)  # YouTube-friendly

    if save_srt and segments:
        write_srt(base + ".srt", segments)
    if save_vtt and segments:
        write_vtt(base + ".vtt", segments)
    if save_json and segments:
        write_json(base + ".json", segments)

    print(f"✅ {len(text.split())} words → {base}.txt")

# ------------------------------
# Interactive Menus
# ------------------------------
def interactive_language_menu():
    langs = ["auto", "en (English)", "vi (Vietnamese)", "ja (Japanese)", "ko (Korean)", "zh (Chinese)"]
    print("\nSelect language:")
    for i, name in enumerate(langs, 1):
        print(f" {i:2d}. {name}")
    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(langs):
            sel = langs[int(choice)-1]
            return None if sel == "auto" else sel.split(" ")[0]
        print("Invalid choice, try again.")

def interactive_model_menu(lang: str):
    models = LANG_MODELS.get(lang, LANG_MODELS["en"])
    backend = models["backend"]
    options = models["models"]
    print(f"\nSelect model for {lang or 'auto'}:")
    for i, name in enumerate(options, 1):
        print(f" {i:2d}. {name}")
    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return backend, options[int(choice)-1]
        print("Invalid choice, try again.")

# ------------------------------
# Orchestrator
# ------------------------------
def transcribe_videos(folder: str, backend: str, model_name: str, lang: str = None,
                      save_srt=False, allow_downloads=True, save_json=False, save_vtt=False, workers=None, max_sub_len=8.0):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp4")]
    if not files:
        print("❌ No MP4 files found")
        return
    model_data = load_backend(backend, model_name, allow_downloads)
    total_start = time.time()
    workers = workers or max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, f, backend, model_data, lang,
                                   save_srt, save_json, save_vtt, max_sub_len): f for f in files}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error processing {futures[future]}: {e}")
    print(f"\n🎉 All done in {time.time() - total_start:.2f}s")

# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch transcribe MP4 files (CPU + FFmpeg optimized, interactive)")
    parser.add_argument("--folder", default=HARDCODED_FOLDER, help="Folder with MP4 files")
    parser.add_argument("--backend", choices=["openai-whisper","pho-whisper","hf-whisper"], help="ASR backend")
    parser.add_argument("--model", help="Model name/id")
    parser.add_argument("--lang", help="Language code (en, vi, ja, ko, zh)")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers (default: CPU-1)")
    parser.add_argument("--srt", action="store_true",default=True, help="Also save .srt subtitle file")
    parser.add_argument("--vtt", action="store_true",default=True, help="Also save .vtt subtitle file")
    parser.add_argument("--json", action="store_true",default=True, help="Also save .json timeline file")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe",
                        help="Whisper task (for hf-whisper only)")
    parser.add_argument("--max-sub-len", type=float, default=8.0, help="Max subtitle segment length in seconds (default 8)")
    args = parser.parse_args()

    lang = args.lang or interactive_language_menu()
    if not args.backend or not args.model:
        backend, model = interactive_model_menu(lang or "en")
    else:
        backend, model = args.backend, args.model

    print(f"\n⚙️ Backend: {backend}")
    print(f"⚙️ Model:   {model}")
    print(f"⚙️ Lang:    {lang or 'auto'}")
    print(f"📁 Folder:  {args.folder}")
    print(f"📝 Outputs: srt={'on' if args.srt else 'off'}, vtt={'on' if args.vtt else 'off'}, json={'on' if args.json else 'off'}")
    print(f"⏱️ Max sub len: {args.max_sub_len}s")
    print(f"🧵 Workers: {args.workers or max(1, os.cpu_count()-1)}")
    print(f"📂 Output dir: output/ (under each video folder)")


    transcribe_videos(
        args.folder,
        backend,
        model,
        lang=lang,
        save_srt=args.srt,
        allow_downloads=True,
        save_json=args.json,
        save_vtt=args.vtt,
        workers=args.workers,
        max_sub_len=args.max_sub_len,
    )

if __name__ == "__main__":
    main()
