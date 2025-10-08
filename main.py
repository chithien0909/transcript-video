import os
import time
import argparse
import subprocess
import json
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ------------------------------
# Config
# ------------------------------
HARDCODED_FOLDER = r"./videos"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")

def configure_cache_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(CACHE_DIR, "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(CACHE_DIR, "hf", "hub"))
    os.environ.setdefault("WHISPER_CACHE_DIR", os.path.join(CACHE_DIR, "whisper"))
    os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)
    
    # macOS performance optimizations
    import platform
    if platform.machine() == "arm64":  # Apple Silicon
        # Enable Apple's Accelerate framework optimizations
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")  # Use all CPU threads
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
        os.environ.setdefault("MKL_NUM_THREADS", "8")
        # Memory optimization for Apple Silicon
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # Disable MPS caching
        # Faster I/O operations
        os.environ.setdefault("PYTHONUNBUFFERED", "1")

configure_cache_dirs()

LANG_MODELS = {
    "en": {"backend": "openai-whisper", "models": ["tiny", "base", "small", "medium", "large"]},
    "vi": {"backend": "openai-whisper", "models": [
        "small",   # Available locally - good balance for Vietnamese
        "base",    # Fast and lightweight  
        "medium",  # Better accuracy but requires download
        "tiny",    # Fastest option
    ]},
    "ja": {"backend": "openai-whisper", "models": ["small", "medium", "large"]},
    "ko": {"backend": "openai-whisper", "models": ["small", "medium", "large"]},
    "zh": {"backend": "openai-whisper", "models": ["small", "medium", "large"]},
}

# ------------------------------
# YouTube Download
# ------------------------------
def download_youtube_video(url: str, output_dir: str) -> str:
    """Download YouTube video and return the path to the downloaded file."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt_dlp is required for --youtube downloads. Install via: python3 -m pip install yt-dlp") from e
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'best[ext=mp4]/mp4/best',  # Prefer mp4 format
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'restrictfilenames': True,  # Avoid special characters in filenames
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"🎬 Downloading video from: {url}")
        try:
            # Extract info to get the filename that will be used
            info = ydl.extract_info(url, download=False)
            filename = ydl.prepare_filename(info)
            
            # Download the video
            ydl.download([url])
            
            print(f"✅ Downloaded: {os.path.basename(filename)}")
            return filename
            
        except Exception as e:
            print(f"❌ Error downloading video: {e}")
            raise

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

# Streaming file writers for real-time output
class StreamingFileWriter:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.srt_file = None
        self.vtt_file = None
        self.json_file = None
        self.txt_file = None
        self.segment_count = 0
        self.json_segments = []
        
    def open_files(self, save_srt=False, save_vtt=False, save_json=False):
        """Open all output files for streaming"""
        if save_srt:
            self.srt_file = open(self.base_path + ".srt", "w", encoding="utf-8")
        if save_vtt:
            self.vtt_file = open(self.base_path + ".vtt", "w", encoding="utf-8")
            self.vtt_file.write("WEBVTT\n\n")
        if save_json:
            self.json_segments = []  # Collect for final write
        # Always create TXT file for full transcript
        self.txt_file = open(self.base_path + ".txt", "w", encoding="utf-8")
    
    def write_segment(self, start: float, end: float, text: str):
        """Write a single segment to all open files"""
        self.segment_count += 1
        processed_text = process_vietnamese_text(text.strip())
        
        # Write to SRT
        if self.srt_file:
            self.srt_file.write(f"{self.segment_count}\n")
            self.srt_file.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            self.srt_file.write(processed_text + "\n\n")
            self.srt_file.flush()  # Immediate write
        
        # Write to VTT
        if self.vtt_file:
            vtt_start = format_srt_timestamp(start).replace(",", ".")
            vtt_end = format_srt_timestamp(end).replace(",", ".")
            self.vtt_file.write(f"{vtt_start} --> {vtt_end}\n{processed_text}\n\n")
            self.vtt_file.flush()  # Immediate write
        
        # Collect for JSON
        if self.json_segments is not None:
            self.json_segments.append({"start": start, "end": end, "text": processed_text})
        
        # Write to TXT (append)
        if self.txt_file:
            self.txt_file.write(processed_text + " ")
            self.txt_file.flush()  # Immediate write
    
    def close_files(self):
        """Close all files and finalize JSON"""
        if self.srt_file:
            self.srt_file.close()
        if self.vtt_file:
            self.vtt_file.close()
        if self.txt_file:
            self.txt_file.close()
        if self.json_segments is not None:
            # Write JSON file at the end
            import json
            with open(self.base_path + ".json", "w", encoding="utf-8") as f:
                json.dump(self.json_segments, f, ensure_ascii=False, indent=2)

def process_vietnamese_text(text: str) -> str:
    """Process Vietnamese text for better readability."""
    if not text:
        return text
    
    # Clean up common transcription issues
    text = text.strip()
    
    # Fix spacing around punctuation
    import re
    text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.!?])([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ])', r'\1 \2', text)  # Add space after sentence-ending punctuation
    
    # Fix common Vietnamese transcription errors
    replacements = {
        'quá mạnh': 'quá mạnh',
        'thét hiệu': 'test hiệu',
        'quá năng': 'hiệu năng',
        'con chip': 'chip',
        'anh em': 'anh em',
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    return text

def split_vietnamese_sentences(text: str) -> List[str]:
    """Split Vietnamese text into natural sentence boundaries."""
    import re
    
    # Vietnamese sentence endings
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ])', text)
    
    # Also split on conjunctions and natural pauses
    result = []
    for sentence in sentences:
        # Split long sentences on conjunctions
        if len(sentence) > 100:  # Long sentence
            parts = re.split(r'\s+(nhưng mà|tuy nhiên|ngoài ra|thậm chí|đặc biệt|tất nhiên|trong khi)\s+', sentence)
            result.extend([p.strip() for p in parts if p.strip() and not p in ['nhưng mà', 'tuy nhiên', 'ngoài ra', 'thậm chí', 'đặc biệt', 'tất nhiên', 'trong khi']])
        else:
            result.append(sentence.strip())
    
    return [s for s in result if s]

def _uniform_segments_from_text(total_seconds: float, text: str):
    import re
    cleaned = process_vietnamese_text(text.strip())
    if not cleaned:
        return None
    
    # Use Vietnamese sentence splitting for better segmentation
    parts = split_vietnamese_sentences(cleaned)
    if not parts:
        # Fallback to simple splitting
        parts = re.split(r"(?<=[\.\!\?…])\s+", cleaned)
        parts = [p.strip() for p in parts if p.strip()]
    
    if not parts:
        parts = [cleaned]
    
    n = len(parts)
    slice_len = max(total_seconds / n, 1.0)  # Minimum 1 second per segment
    segments = []
    t = 0.0
    
    for p in parts:
        start = t
        # Adjust segment length based on text length (Vietnamese words tend to be shorter)
        text_factor = max(0.5, min(2.0, len(p) / 50))  # Scale based on text length
        segment_duration = slice_len * text_factor
        end = min(total_seconds, start + segment_duration)
        segments.append((start, end, p))
        t = end
    
    if segments:
        segments[-1] = (segments[-1][0], total_seconds, segments[-1][2])
    
    return segments

def resolve_ffmpeg_path() -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    local_ffmpeg = os.path.join(project_root, "tools", "ffmpeg", "bin", "ffmpeg.exe")
    # Use vendored .exe only on Windows; on macOS/Linux prefer native ffmpeg
    if os.name == "nt" and os.path.exists(local_ffmpeg):
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
    
    import platform
    if platform.machine() == "arm64":  # Apple Silicon optimizations
        # Use hardware-accelerated decoding on Apple Silicon
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error",
            "-hwaccel", "videotoolbox",  # Hardware acceleration
            "-i", input_file, 
            "-ac", "1",  # Mono
            "-ar", "16000",  # 16kHz sample rate
            "-acodec", "pcm_s16le",  # Optimal codec for Apple Silicon
            "-af", "volume=1.0",  # Ensure consistent volume
            "-y",  # Overwrite output file
            wav_path
        ]
    else:  # Intel Mac fallback
        cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error",
               "-i", input_file, "-ac", "1", "-ar", "16000", wav_path]
    
    subprocess.run(cmd, check=True)
    return wav_path

# ------------------------------
# Subtitle splitting
# ------------------------------
def split_for_subtitles(segments, max_len=8.0, max_chars=80):
    """Split segments for subtitles with Vietnamese-specific optimizations."""
    new_segments = []
    for start, end, text in tqdm(segments, desc="Splitting subtitles", unit="segment"):
        duration = end - start
        text = text.strip()
        
        # If segment is already good length and not too long text-wise
        if duration <= max_len and len(text) <= max_chars:
            new_segments.append((start, end, text))
            continue
            
        # Split by natural Vietnamese phrase boundaries
        import re
        
        # First try to split by punctuation
        phrases = re.split(r'([,.;:]\s+)', text)
        phrases = [p.strip() for p in phrases if p.strip() and p.strip() not in ',.;:']
        
        if len(phrases) <= 1:
            # Fallback to word splitting
            phrases = text.split()
        
        if len(phrases) <= 1:
            # Single word/phrase - just keep as is
            new_segments.append((start, end, text))
            continue
            
        # Distribute phrases across time
        total_chars = sum(len(p) for p in phrases)
        chunk_start = start
        current_phrases = []
        current_chars = 0
        
        for i, phrase in enumerate(phrases):
            current_phrases.append(phrase)
            current_chars += len(phrase)
            
            # Check if we should make a segment
            should_split = False
            
            # Split if too many characters
            if current_chars >= max_chars:
                should_split = True
            
            # Split if time would exceed max_len
            estimated_duration = (current_chars / total_chars) * duration
            if estimated_duration >= max_len:
                should_split = True
                
            # Always split on last phrase
            if i == len(phrases) - 1:
                should_split = True
                
            if should_split and current_phrases:
                # Calculate end time based on character proportion
                time_proportion = current_chars / total_chars
                chunk_end = min(end, start + (time_proportion * duration))
                
                # Ensure minimum segment length
                if chunk_end - chunk_start < 0.5:
                    chunk_end = chunk_start + 0.5
                
                segment_text = ' '.join(current_phrases).strip()
                if segment_text:
                    new_segments.append((chunk_start, chunk_end, segment_text))
                
                chunk_start = chunk_end
                current_phrases = []
                current_chars = 0
                
    return new_segments

# ------------------------------
# Backends
# ------------------------------
def load_backend(backend: str, model_name: str, allow_downloads: bool):
    if backend == "openai-whisper":
        import whisper, torch
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        cache_dir = os.environ.get("WHISPER_CACHE_DIR") or os.path.join(PROJECT_ROOT, ".cache", "whisper")
        os.makedirs(cache_dir, exist_ok=True)
        if not allow_downloads:
            expected = os.path.join(cache_dir, f"{model_name}.pt")
            if not os.path.exists(expected):
                raise RuntimeError(f"Model '{model_name}' not found in cache ({expected}). Re-run with --allow-downloads to fetch it.")
        return whisper.load_model(model_name, device="cpu", download_root=cache_dir)
    
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel
        import platform
        
        # Optimize for Apple Silicon M1 Pro
        if platform.machine() == "arm64":  # Apple Silicon
            # Use optimized settings for Apple Silicon
            # Set CPU threads to use performance cores (6) + 1 efficiency core
            return WhisperModel(
                model_name, 
                device="cpu", 
                compute_type="int8",  # Stable performance on Apple Silicon
                cpu_threads=7,  # 6 performance + 1 efficiency core
                num_workers=1,  # Single worker per model instance
                local_files_only=not allow_downloads
            )
        else:  # Intel Mac fallback
            return WhisperModel(model_name, device="cpu", compute_type="int8", 
                               local_files_only=not allow_downloads)

    if backend == "pho-whisper":
        # detect if it's ct2 or not
        if model_name.endswith("-ct2"):
            from faster_whisper import WhisperModel
            return ("pho-ct2", WhisperModel(model_name, device="cpu", compute_type="int8",
                                            local_files_only=not allow_downloads))
        else:
            try:
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            except ImportError:
                print("\nInstalling 'transformers' package...")
                import subprocess as _subprocess
                _subprocess.check_call(["pip", "install", "transformers"])
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

def transcribe_faster_whisper(model, file_path: str, lang: str = None):
    """Transcribe using faster-whisper with optimized settings for Vietnamese."""
    # Extract audio to WAV format for better compatibility
    print(f"🎵 Extracting audio from {os.path.basename(file_path)}...")
    wav_path = extract_wav(file_path)
    print(f"✅ Audio extracted to WAV format")
    
    # Optimize settings for Vietnamese and Apple Silicon performance
    print(f"🧠 Running faster-whisper transcription...")
    
    import platform
    if platform.machine() == "arm64":  # Apple Silicon optimizations
        segments, info = model.transcribe(
            wav_path,
            language=lang or None,
            beam_size=3,  # Reduced for speed on Apple Silicon
            best_of=1,    # Single candidate for speed
            temperature=0.0,  # Single temperature for consistency
            vad_filter=True,  # Keep VAD for efficiency
            vad_parameters=dict(
                min_silence_duration_ms=300,  # Shorter silence detection
                threshold=0.5,  # Lower threshold for better detection
            ),
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt=None,
            # Apple Silicon specific optimizations
            chunk_length=30,  # Process in 30s chunks for efficiency
            without_timestamps=False,  # Keep timestamps
        )
    else:  # Intel Mac fallback with original settings
        segments, info = model.transcribe(
            wav_path,
            language=lang or None,
            beam_size=5,
            best_of=3,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt=None,
        )
    
    print(f"🔄 Processing transcription results...")
    # Convert segments to our format with progress bar
    print(f"📄 Converting segments to list format...")
    segments_list = []
    # Use tqdm without total since we don't know the count upfront (lazy iterator)
    with tqdm(desc="Processing segments", unit="seg", leave=False) as pbar:
        for seg in segments:
            segments_list.append((seg.start, seg.end, seg.text))
            # Update progress bar less frequently for better performance
            if len(segments_list) % 10 == 0 or len(segments_list) == 1:
                pbar.set_postfix({"count": len(segments_list), "duration": f"{seg.end:.1f}s"})
            pbar.update(1)
    print(f"✅ Converted {len(segments_list)} segments")
    # Join all text and process Vietnamese
    full_text = "".join(seg.text for seg in segments).strip()
    full_text = process_vietnamese_text(full_text)
    
    # Process individual segments with minimal logging
    processed_segments = [(start, end, process_vietnamese_text(text)) 
                         for start, end, text in tqdm(segments_list, desc="Processing Vietnamese", unit="segment", leave=False)]
    return full_text, processed_segments

def transcribe_faster_whisper_streaming(model, file_path: str, writer: StreamingFileWriter, lang: str = None, max_sub_len: float = 8.0):
    """Transcribe using faster-whisper with streaming output to files"""
    # Extract audio to WAV format for better compatibility
    print(f"🎵 Extracting audio from {os.path.basename(file_path)}...")
    wav_path = extract_wav(file_path)
    print(f"✅ Audio extracted to WAV format")
    
    # Optimize settings for Vietnamese and Apple Silicon performance
    print(f"🧠 Running faster-whisper transcription with streaming...")
    
    import platform
    if platform.machine() == "arm64":  # Apple Silicon optimizations
        segments, info = model.transcribe(
            wav_path,
            language=lang or None,
            beam_size=3,  # Reduced for speed on Apple Silicon
            best_of=1,    # Single candidate for speed
            temperature=0.0,  # Single temperature for consistency
            vad_filter=True,  # Keep VAD for efficiency
            vad_parameters=dict(
                min_silence_duration_ms=300,  # Shorter silence detection
                threshold=0.5,  # Lower threshold for better detection
            ),
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt=None,
            # Apple Silicon specific optimizations
            chunk_length=30,  # Process in 30s chunks for efficiency
            without_timestamps=False,  # Keep timestamps
        )
    else:  # Intel Mac fallback with original settings
        segments, info = model.transcribe(
            wav_path,
            language=lang or None,
            beam_size=5,
            best_of=3,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt=None,
        )
    
    print(f"🔄 Streaming segments to files...")
    # Stream segments directly to files with progress bar
    segments_for_splitting = []  # Collect for subtitle splitting only
    
    with tqdm(desc="Streaming segments", unit="seg", leave=False) as pbar:
        for seg in segments:
            # Write segment immediately to files
            writer.write_segment(seg.start, seg.end, seg.text)
            
            # Also collect for subtitle splitting later if needed
            segments_for_splitting.append((seg.start, seg.end, seg.text))
            
            # Update progress
            pbar.set_postfix({"count": len(segments_for_splitting), "duration": f"{seg.end:.1f}s"})
            pbar.update(1)
    
    print(f"✅ Streamed {len(segments_for_splitting)} segments to files")
    return segments_for_splitting

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
    decoded = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    duration_sec = float(len(audio) / 16000.0)
    segments = _uniform_segments_from_text(duration_sec, decoded) if decoded else None
    return decoded, segments

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
    decoded = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    duration_sec = float(len(audio) / 16000.0)
    segments = _uniform_segments_from_text(duration_sec, decoded) if decoded else None
    return decoded, segments

def transcribe(backend, model_data, file_path, lang):
    if backend == "openai-whisper":
        return transcribe_openai(model_data, file_path, lang)
    elif backend == "faster-whisper":
        return transcribe_faster_whisper(model_data, file_path, lang)
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
def process_file(file_path: str, backend: str, model_name: str, lang: str,
                 save_srt=False, save_json=False, save_vtt=False, max_sub_len: float = 8.0, allow_downloads: bool = True):
    # Optimize process priority on macOS
    import platform
    if platform.machine() == "arm64":
        try:
            import os
            # Set higher priority for transcription process
            os.nice(-5)  # Higher priority (requires admin on some systems)
        except (OSError, PermissionError):
            pass  # Continue if can't set priority
    
    video_dir = os.path.dirname(file_path)
    out_dir = os.path.join(video_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    base = os.path.join(out_dir, base_name)
    fname = os.path.basename(file_path)
    print(f"🎬 Processing {fname} ...")

    # Load model in this worker process to avoid pickling issues
    print(f"⚙️ Loading {backend} model '{model_name}' for {fname}...")
    model_data = load_backend(backend, model_name, allow_downloads)
    print(f"✅ Model loaded for {fname}")
    
    # Create streaming writer for real-time output
    writer = StreamingFileWriter(base)
    writer.open_files(save_srt=save_srt, save_vtt=save_vtt, save_json=save_json)
    
    try:
        print(f"🎬 Starting streaming transcription for {fname}...")
        
        if backend == "faster-whisper":
            # Use streaming transcription for faster-whisper
            segments = transcribe_faster_whisper_streaming(model_data, file_path, writer, lang, max_sub_len)
            print(f"✅ Streaming transcription complete for {fname} - {len(segments)} segments")
            
            # Optional: Create split subtitles if needed (but basic files are already written)
            if segments and max_sub_len < 8.0:  # Only if custom split length requested
                print(f"🔄 Creating split subtitles...")
                split_segments = split_for_subtitles(segments, max_len=max_sub_len)
                if save_srt:
                    write_srt(base + "_split.srt", split_segments)
                if save_vtt:
                    write_vtt(base + "_split.vtt", split_segments)
        else:
            # Fall back to regular transcription for other backends
            text, segments = transcribe(backend, model_data, file_path, lang)
            print(f"✅ Transcription complete for {fname} - {len(segments) if segments else 0} segments")
            
            # Write to files manually for non-streaming backends
            if segments:
                for start, end, text_seg in segments:
                    writer.write_segment(start, end, text_seg)
    
    finally:
        # Always close files
        writer.close_files()
    
    print(f"✅ All output files saved for {fname}")

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
                      save_srt: bool = False, allow_downloads: bool = True, save_json: bool = False,
                      save_vtt: bool = False, workers: int = None, max_sub_len: float = 8.0):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp4")]
    if not files:
        print("❌ No MP4 files found")
        return
    # Don't load model here - let each worker process load it to avoid pickling issues
    total_start = time.time()
    
    import platform
    if platform.machine() == "arm64":  # Apple Silicon optimizations
        # On Apple Silicon, use fewer workers but each with more CPU power
        # M1 Pro: 6 performance cores, so use 2-3 workers to avoid context switching
        workers = workers or min(3, len(files))  # Max 3 workers or number of files
    else:
        workers = workers or max(1, os.cpu_count() - 1)
    
    print(f"📺 Processing {len(files)} video file(s)...")
    
    # Set optimal multiprocessing method for macOS
    import multiprocessing as mp
    if platform.machine() == "arm64" and hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method('fork', force=True)  # Faster than spawn on macOS
        except RuntimeError:
            pass  # Already set
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, f, backend, model_name, lang,
                                   save_srt, save_json, save_vtt, max_sub_len, allow_downloads): f for f in files}
        
        # Use tqdm for overall progress tracking
        with tqdm(total=len(files), desc="Processing videos", unit="video") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"❌ Error processing {futures[future]}: {e}")
                    pbar.update(1)
    print(f"\n🎉 All done in {time.time() - total_start:.2f}s")


# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch transcribe MP4 files (CPU + FFmpeg optimized, interactive)")
    parser.add_argument("--folder", default=HARDCODED_FOLDER, help="Folder with MP4 files")
    parser.add_argument("--youtube", help="YouTube URL to download and transcribe")
    parser.add_argument("--backend", choices=["openai-whisper","pho-whisper","hf-whisper","faster-whisper"], help="ASR backend")
    parser.add_argument("--model", help="Model name/id")
    parser.add_argument("--lang", default="auto", help="Language code or 'auto' (en, vi, ja, ko, zh, auto)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU-1)")
    parser.add_argument("--srt", action="store_true",default=True, help="Also save .srt subtitle file")
    parser.add_argument("--vtt", action="store_true",default=True, help="Also save .vtt subtitle file")
    parser.add_argument("--json", action="store_true",default=True, help="Also save .json timeline file")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe",
                        help="Whisper task (for hf-whisper only)")
    parser.add_argument("--max-sub-len", type=float, default=8.0, help="Max subtitle segment length in seconds (default 8)")
    parser.add_argument("--allow-downloads", dest="allow_downloads", action="store_true", default=True, help="Allow model downloads if missing")
    parser.add_argument("--no-downloads", dest="allow_downloads", action="store_false", help="Do not download models; require local cache")
    args = parser.parse_args()

    # Handle YouTube download if URL is provided
    if args.youtube:
        print(f"🎬 YouTube URL provided: {args.youtube}") 
        try:
            downloaded_file = download_youtube_video(args.youtube, args.folder)
            print(f"✅ Video downloaded to: {downloaded_file}")
        except Exception as e:
            print(f"❌ Failed to download video: {e}")
            return

    # Map 'auto' to None for backend auto-detection
    lang = None if (args.lang and args.lang.lower() == "auto") else args.lang
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
        allow_downloads=args.allow_downloads,
        save_json=args.json,
        save_vtt=args.vtt,
        workers=args.workers,
        max_sub_len=args.max_sub_len,
    )

if __name__ == "__main__":
    main()
