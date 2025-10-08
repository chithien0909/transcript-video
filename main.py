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
TRANSCRIPTS_FOLDER = r"./transcripts"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")

# Performance optimization settings
import psutil
import multiprocessing as mp

# System resource detection
SYSTEM_CORES = os.cpu_count() or 4
PHYSICAL_CORES = psutil.cpu_count(logical=False) or SYSTEM_CORES // 2
TOTAL_MEMORY_GB = psutil.virtual_memory().total / (1024**3)
AVAILABLE_MEMORY_GB = psutil.virtual_memory().available / (1024**3)

# Optimized worker configuration based on system resources
def get_optimal_workers():
    """Calculate optimal number of workers based on system resources."""
    # For Intel i7-1260P (12 cores, 16 threads): Use 8-10 workers for transcription
    # Leave some cores for system and I/O operations
    if SYSTEM_CORES >= 16:
        return min(10, SYSTEM_CORES - 4)  # Leave 4 cores for system
    elif SYSTEM_CORES >= 8:
        return min(6, SYSTEM_CORES - 2)   # Leave 2 cores for system
    else:
        return max(1, SYSTEM_CORES - 1)    # Leave 1 core for system

OPTIMAL_WORKERS = get_optimal_workers()

# Memory optimization settings
MAX_MEMORY_PER_WORKER_GB = 2.0  # Maximum memory per worker process
OPTIMAL_CHUNK_SIZE = min(30, max(10, int(AVAILABLE_MEMORY_GB / OPTIMAL_WORKERS)))  # Audio chunk size in seconds

# Get ffmpeg path for yt-dlp
def get_ffmpeg_path():
    """Get the path to ffmpeg executable for yt-dlp."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    local_ffmpeg = os.path.join(project_root, "tools", "ffmpeg", "bin", "ffmpeg.exe")
    
    # Use vendored .exe on Windows if it exists
    if os.name == "nt" and os.path.exists(local_ffmpeg):
        return local_ffmpeg
    
    # Fallback to system ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    
    return "ffmpeg"  # System ffmpeg

FFMPEG_PATH = get_ffmpeg_path()

# yt-dlp Configuration Presets (will be initialized after ffmpeg path is determined)
def create_ytdlp_presets():
    """Create yt-dlp presets with ffmpeg path included."""
    return {
        "default": {
            "format": "best[ext=mp4]/mp4/best",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ffmpeg_location": FFMPEG_PATH,
        },
        "high_quality": {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/mp4",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ffmpeg_location": FFMPEG_PATH,
        },
        "fast_download": {
            "format": "worst[ext=mp4]/mp4/worst",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ffmpeg_location": FFMPEG_PATH,
        },
        "with_subtitles": {
            "format": "best[ext=mp4]/mp4/best",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "vi"],
            "ffmpeg_location": FFMPEG_PATH,
        },
        "audio_only": {
            "format": "bestaudio[ext=m4a]/bestaudio",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "extractaudio": True,
            "audioformat": "mp3",
            "audioquality": "192K",
            "ffmpeg_location": FFMPEG_PATH,
        },
        "playlist": {
            "format": "best[ext=mp4]/mp4/best",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "playlistend": 10,  # Limit to first 10 videos
            "ffmpeg_location": FFMPEG_PATH,
        },
        # Quality-specific presets
        "4k_ultra": {
            "format": "bestvideo[height<=2160][ext=mp4]+bestaudio[ext=m4a]/best[height<=2160][ext=mp4]/mp4",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ffmpeg_location": FFMPEG_PATH,
        },
        "1080p_hd": {
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/mp4",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ffmpeg_location": FFMPEG_PATH,
        },
        "720p_hd": {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/mp4",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ffmpeg_location": FFMPEG_PATH,
        },
        "480p_sd": {
            "format": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/mp4",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ffmpeg_location": FFMPEG_PATH,
        },
        "360p_low": {
            "format": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/mp4",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ffmpeg_location": FFMPEG_PATH,
        },
        "240p_minimal": {
            "format": "bestvideo[height<=240][ext=mp4]+bestaudio[ext=m4a]/best[height<=240][ext=mp4]/mp4",
            "restrictfilenames": True,
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ffmpeg_location": FFMPEG_PATH,
        }
    }

# Initialize presets after ffmpeg path is determined
YTDLP_PRESETS = create_ytdlp_presets()

# Default yt-dlp configuration file path
YTDLP_CONFIG_FILE = os.path.join(PROJECT_ROOT, "ytdlp_config.json")

def configure_cache_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(CACHE_DIR, "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(CACHE_DIR, "hf", "hub"))
    os.environ.setdefault("WHISPER_CACHE_DIR", os.path.join(CACHE_DIR, "whisper"))
    os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)
    
    # Performance optimizations for Intel i7-1260P
    import platform
    
    # CPU optimization for Intel processors
    os.environ.setdefault("OMP_NUM_THREADS", str(PHYSICAL_CORES))
    os.environ.setdefault("MKL_NUM_THREADS", str(PHYSICAL_CORES))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(PHYSICAL_CORES))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(PHYSICAL_CORES))
    
    # Memory optimization
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Avoid tokenizer warnings
    
    # Intel-specific optimizations
    if platform.machine() == "AMD64":  # Intel x64
        # Enable Intel MKL optimizations
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
        # Memory allocation optimization
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    
    # macOS performance optimizations (if running on Mac)
    if platform.machine() == "arm64":  # Apple Silicon
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(PHYSICAL_CORES))
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

configure_cache_dirs()

# ------------------------------
# yt-dlp Configuration Management
# ------------------------------
def load_ytdlp_config():
    """Load yt-dlp configuration from file or return default preset."""
    if os.path.exists(YTDLP_CONFIG_FILE):
        try:
            with open(YTDLP_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config
        except (json.JSONDecodeError, IOError) as e:
            print(f" Warning: Could not load yt-dlp config file: {e}")
            print("Using default configuration.")
    
    return YTDLP_PRESETS["default"]

def save_ytdlp_config(config):
    """Save yt-dlp configuration to file."""
    try:
        with open(YTDLP_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f" yt-dlp configuration saved to {YTDLP_CONFIG_FILE}")
    except IOError as e:
        print(f" Error saving yt-dlp config: {e}")

def get_ytdlp_config(preset_name=None, custom_config=None):
    """Get yt-dlp configuration based on preset or custom config."""
    if custom_config:
        # Ensure ffmpeg path is included in custom config
        config = custom_config.copy()
        if "ffmpeg_location" not in config:
            config["ffmpeg_location"] = FFMPEG_PATH
        return config
    
    if preset_name and preset_name in YTDLP_PRESETS:
        return YTDLP_PRESETS[preset_name]
    
    # Load from file and ensure ffmpeg path is included
    config = load_ytdlp_config()
    if "ffmpeg_location" not in config:
        config["ffmpeg_location"] = FFMPEG_PATH
    return config

def interactive_ytdlp_config():
    """Interactive menu for configuring yt-dlp settings."""
    print("\n yt-dlp Configuration")
    print("-" * 30)
    print("Select configuration preset:")
    print(" 1. Default (balanced quality/speed)")
    print(" 2. High Quality (best video + audio)")
    print(" 3. Fast Download (lower quality, faster)")
    print(" 4. With Subtitles (download subtitles too)")
    print(" 5. Audio Only (extract audio)")
    print(" 6. Playlist (limit playlist downloads)")
    print(" 7.  Quality Selection")
    print(" 8. Custom Configuration")
    print(" 9. Load from file")
    print("10. Save current config")
    
    while True:
        choice = input("\nSelect option (1-10): ").strip()
        
        if choice == "1":
            return "default", YTDLP_PRESETS["default"]
        elif choice == "2":
            return "high_quality", YTDLP_PRESETS["high_quality"]
        elif choice == "3":
            return "fast_download", YTDLP_PRESETS["fast_download"]
        elif choice == "4":
            return "with_subtitles", YTDLP_PRESETS["with_subtitles"]
        elif choice == "5":
            return "audio_only", YTDLP_PRESETS["audio_only"]
        elif choice == "6":
            return "playlist", YTDLP_PRESETS["playlist"]
        elif choice == "7":
            return interactive_quality_selection()
        elif choice == "8":
            return "custom", create_custom_ytdlp_config()
        elif choice == "9":
            config = load_ytdlp_config()
            return "file", config
        elif choice == "10":
            save_ytdlp_config(YTDLP_PRESETS["default"])
            print(" Default configuration saved to file.")
            continue
        else:
            print(" Invalid choice. Please enter 1-10.")

def interactive_quality_selection():
    """Interactive menu for selecting video quality."""
    print("\n Video Quality Selection")
    print("-" * 30)
    print("Select video quality:")
    print(" 1. 4K Ultra (2160p) - Highest quality, largest file")
    print(" 2. 1080p HD - High quality, good balance")
    print(" 3. 720p HD - Medium quality, smaller file")
    print(" 4. 480p SD - Standard quality, compact file")
    print(" 5. 360p Low - Lower quality, small file")
    print(" 6. 240p Minimal - Lowest quality, tiny file")
    print(" 7. Custom resolution")
    print(" 8. Back to main menu")
    
    while True:
        choice = input("\nSelect quality option (1-8): ").strip()
        
        if choice == "1":
            return "4k_ultra", YTDLP_PRESETS["4k_ultra"]
        elif choice == "2":
            return "1080p_hd", YTDLP_PRESETS["1080p_hd"]
        elif choice == "3":
            return "720p_hd", YTDLP_PRESETS["720p_hd"]
        elif choice == "4":
            return "480p_sd", YTDLP_PRESETS["480p_sd"]
        elif choice == "5":
            return "360p_low", YTDLP_PRESETS["360p_low"]
        elif choice == "6":
            return "240p_minimal", YTDLP_PRESETS["240p_minimal"]
        elif choice == "7":
            return create_custom_quality_config()
        elif choice == "8":
            return interactive_ytdlp_config()  # Go back to main config menu
        else:
            print(" Invalid choice. Please enter 1-8.")

def create_custom_quality_config():
    """Create custom quality configuration."""
    print("\n Custom Quality Configuration")
    print("-" * 30)
    
    # Get resolution
    while True:
        resolution = input("Enter maximum resolution height (e.g., 1080, 720, 480): ").strip()
        if resolution.isdigit() and int(resolution) > 0:
            resolution = int(resolution)
            break
        print(" Please enter a valid positive number.")
    
    # Get additional options
    print(f"\nAdditional options for {resolution}p:")
    print(" 1. Video + Audio (recommended)")
    print(" 2. Video only")
    print(" 3. Audio only")
    
    while True:
        option = input("Select option (1-3): ").strip()
        if option == "1":
            format_str = f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}][ext=mp4]/mp4"
            break
        elif option == "2":
            format_str = f"bestvideo[height<={resolution}][ext=mp4]/best[height<={resolution}][ext=mp4]/mp4"
            break
        elif option == "3":
            format_str = "bestaudio[ext=m4a]/bestaudio"
            break
        else:
            print(" Invalid choice. Please enter 1-3.")
    
    config = {
        "format": format_str,
        "restrictfilenames": True,
        "writeinfojson": False,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "ffmpeg_location": FFMPEG_PATH,
    }
    
    # Add audio extraction if audio only
    if option == "3":
        extract_audio = input("Extract to MP3? (y/n): ").strip().lower()
        if extract_audio in ['y', 'yes']:
            config["extractaudio"] = True
            config["audioformat"] = "mp3"
            quality = input("Audio quality (e.g., 192K, 320K): ").strip() or "192K"
            config["audioquality"] = quality
    
    print(f"\n Custom quality configuration created:")
    print(f"  Resolution: {resolution}p")
    print(f"  Format: {format_str}")
    
    return f"custom_{resolution}p", config

def create_custom_ytdlp_config():
    """Create custom yt-dlp configuration through interactive prompts."""
    config = {}
    
    print("\n Custom yt-dlp Configuration")
    print("-" * 30)
    
    # Format selection with quality options
    print("\nVideo format options:")
    print(" 1. best[ext=mp4]/mp4/best (default)")
    print(" 2. bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/mp4 (high quality)")
    print(" 3. worst[ext=mp4]/mp4/worst (fast download)")
    print(" 4. bestaudio[ext=m4a]/bestaudio (audio only)")
    print(" 5.  Quality-specific formats")
    print(" 6. Custom format string")
    
    format_choice = input("Select format (1-6): ").strip()
    format_options = {
        "1": "best[ext=mp4]/mp4/best",
        "2": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/mp4",
        "3": "worst[ext=mp4]/mp4/worst",
        "4": "bestaudio[ext=m4a]/bestaudio",
    }
    
    if format_choice in format_options:
        config["format"] = format_options[format_choice]
    elif format_choice == "5":
        # Quality-specific selection
        print("\nQuality-specific formats:")
        print(" 1. 4K Ultra (2160p)")
        print(" 2. 1080p HD")
        print(" 3. 720p HD")
        print(" 4. 480p SD")
        print(" 5. 360p Low")
        print(" 6. Custom resolution")
        
        quality_choice = input("Select quality (1-6): ").strip()
        quality_formats = {
            "1": "bestvideo[height<=2160][ext=mp4]+bestaudio[ext=m4a]/best[height<=2160][ext=mp4]/mp4",
            "2": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/mp4",
            "3": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/mp4",
            "4": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/mp4",
            "5": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/mp4",
        }
        
        if quality_choice in quality_formats:
            config["format"] = quality_formats[quality_choice]
        elif quality_choice == "6":
            resolution = input("Enter maximum resolution height (e.g., 1080, 720): ").strip()
            if resolution.isdigit():
                config["format"] = f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}][ext=mp4]/mp4"
            else:
                config["format"] = "best[ext=mp4]/mp4/best"
        else:
            config["format"] = "best[ext=mp4]/mp4/best"
    elif format_choice == "6":
        custom_format = input("Enter custom format string: ").strip()
        if custom_format:
            config["format"] = custom_format
        else:
            config["format"] = "best[ext=mp4]/mp4/best"
    else:
        config["format"] = "best[ext=mp4]/mp4/best"
    
    # Additional options
    config["restrictfilenames"] = True
    config["ffmpeg_location"] = FFMPEG_PATH
    
    # Subtitles
    subtitles = input("Download subtitles? (y/n): ").strip().lower()
    if subtitles in ['y', 'yes']:
        config["writesubtitles"] = True
        config["writeautomaticsub"] = True
        langs = input("Subtitle languages (comma-separated, e.g., en,vi): ").strip()
        if langs:
            config["subtitleslangs"] = [lang.strip() for lang in langs.split(",")]
    else:
        config["writesubtitles"] = False
        config["writeautomaticsub"] = False
    
    # Audio extraction
    if config["format"].startswith("bestaudio"):
        extract_audio = input("Extract to MP3? (y/n): ").strip().lower()
        if extract_audio in ['y', 'yes']:
            config["extractaudio"] = True
            config["audioformat"] = "mp3"
            quality = input("Audio quality (e.g., 192K, 320K): ").strip() or "192K"
            config["audioquality"] = quality
    
    # Playlist options
    playlist_limit = input("Playlist video limit (number or 'none'): ").strip()
    if playlist_limit.isdigit():
        config["playlistend"] = int(playlist_limit)
    
    # Output template
    template = input("Output filename template (or press Enter for default): ").strip()
    if template:
        config["outtmpl"] = template
    
    print(f"\n Custom configuration created:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

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
def create_video_folder(video_title: str, base_dir: str) -> str:
    """Create a folder for the video based on title and timestamp."""
    import re
    from datetime import datetime
    
    # Clean video title for folder name
    clean_title = re.sub(r'[<>:"/\\|?*]', '', video_title)  # Remove invalid characters
    clean_title = re.sub(r'\s+', '_', clean_title.strip())  # Replace spaces with underscores
    clean_title = clean_title[:50]  # Limit length
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create folder name
    folder_name = f"{clean_title}_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    
    # Create the folder
    os.makedirs(folder_path, exist_ok=True)
    
    return folder_path

def get_video_info(url: str) -> dict:
    """Get video information without downloading."""
    try:
        import yt_dlp
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'uploader': info.get('uploader', 'Unknown'),
                'duration': info.get('duration', 0),
                'upload_date': info.get('upload_date', ''),
                'view_count': info.get('view_count', 0),
            }
    except Exception as e:
        print(f" Could not get video info: {e}")
        return {
            'title': 'Unknown_Video',
            'uploader': 'Unknown',
            'duration': 0,
            'upload_date': '',
            'view_count': 0,
        }

def save_video_info(video_folder: str, video_info: dict, url: str):
    """Save video information to a JSON file in the video folder."""
    import json
    from datetime import datetime
    
    info_file = os.path.join(video_folder, "video_info.json")
    info_data = {
        "url": url,
        "title": video_info['title'],
        "uploader": video_info['uploader'],
        "duration": video_info['duration'],
        "upload_date": video_info['upload_date'],
        "view_count": video_info['view_count'],
        "download_time": datetime.now().isoformat(),
        "folder_name": os.path.basename(video_folder)
    }
    
    try:
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        print(f" Video info saved to: video_info.json")
    except Exception as e:
        print(f" Could not save video info: {e}")

def download_youtube_video(url: str, output_dir: str, preset_name: str = None, custom_config: dict = None) -> str:
    """Download YouTube video and return the path to the downloaded file."""
    # Get video info first to create proper folder structure
    print(f" Getting video information...")
    video_info = get_video_info(url)
    video_title = video_info['title']
    
    # Create video-specific folder
    video_folder = create_video_folder(video_title, output_dir)
    print(f" Created folder: {os.path.basename(video_folder)}")
    
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt_dlp is required for --youtube downloads. Install via: python3 -m pip install yt-dlp") from e
    
    # Get configuration
    config = get_ytdlp_config(preset_name, custom_config)
    
    # Configure yt-dlp options
    ydl_opts = config.copy()
    
    # Set output template to use video-specific folder
    if 'outtmpl' not in ydl_opts:
        ydl_opts['outtmpl'] = os.path.join(video_folder, '%(title)s.%(ext)s')
    
    # Add progress hook for better user feedback
    def progress_hook(d):
        if d['status'] == 'downloading':
            percent = d.get('_percent_str', 'N/A')
            speed = d.get('_speed_str', 'N/A')
            print(f"\r Downloading: {percent} at {speed}", end='', flush=True)
        elif d['status'] == 'finished':
            print(f"\r Downloaded: {os.path.basename(d['filename'])}")
    
    ydl_opts['progress_hooks'] = [progress_hook]
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f" Downloading video from: {url}")
        print(f" Using configuration: {preset_name or 'custom' if custom_config else 'default'}")
        print(f" Using ffmpeg: {ydl_opts.get('ffmpeg_location', 'system default')}")
        
        try:
            # Extract info to get the filename that will be used
            info = ydl.extract_info(url, download=False)
            filename = ydl.prepare_filename(info)
            
            # Download the video
            ydl.download([url])
            
            # Save video information
            save_video_info(video_folder, video_info, url)
            
            print(f" Download complete: {os.path.basename(filename)}")
            return filename
            
        except Exception as e:
            print(f" Error downloading video: {e}")
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
    
    # Get video duration for progress tracking
    duration_cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-i", input_file,
        "-f", "null", "-"
    ]
    
    try:
        # Get duration first
        result = subprocess.run(duration_cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
        duration_match = None
        for line in result.stdout.split('\n'):
            if 'Duration:' in line:
                duration_str = line.split('Duration:')[1].split(',')[0].strip()
                h, m, s = duration_str.split(':')
                duration_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                duration_match = duration_seconds
                break
        
        import platform
        if platform.machine() == "arm64":  # Apple Silicon optimizations
            # Use hardware-accelerated decoding on Apple Silicon
            cmd = [
                ffmpeg_bin, "-hide_banner", "-loglevel", "info",
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
            cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "info",
                   "-i", input_file, "-ac", "1", "-ar", "16000", wav_path]
        
        if duration_match:
            print(f" Extracting audio (duration: {duration_match:.1f}s)...")
            with tqdm(total=duration_match, desc="Extracting audio", unit="s", leave=False) as pbar:
                process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
                
                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        # Parse ffmpeg progress
                        if 'time=' in output:
                            try:
                                time_part = output.split('time=')[1].split()[0]
                                h, m, s = time_part.split(':')
                                current_time = int(h) * 3600 + int(m) * 60 + float(s)
                                pbar.n = min(current_time, duration_match)
                                pbar.refresh()
                            except (ValueError, IndexError):
                                pass
                
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
        else:
            # Fallback without progress if duration can't be determined
            print(f" Extracting audio...")
            subprocess.run(cmd, check=True)
        
        return wav_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr.decode()}")

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
    print(f" Loading {backend} model '{model_name}'...")
    
    if backend == "openai-whisper":
        import whisper, torch
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        cache_dir = os.environ.get("WHISPER_CACHE_DIR") or os.path.join(PROJECT_ROOT, ".cache", "whisper")
        os.makedirs(cache_dir, exist_ok=True)
        if not allow_downloads:
            expected = os.path.join(cache_dir, f"{model_name}.pt")
            if not os.path.exists(expected):
                raise RuntimeError(f"Model '{model_name}' not found in cache ({expected}). Re-run with --allow-downloads to fetch it.")
        
        with tqdm(desc="Loading OpenAI Whisper model", unit="step", leave=False) as pbar:
            pbar.set_description("Downloading model files...")
            pbar.update(1)
            model = whisper.load_model(model_name, device="cpu", download_root=cache_dir)
            pbar.set_description("Model loaded successfully")
            pbar.update(1)
        return model
    
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel
        import platform
        
        with tqdm(desc="Loading Faster-Whisper model", unit="step", leave=False) as pbar:
            pbar.set_description("Initializing model...")
            pbar.update(1)
            
            # Optimize for Intel i7-1260P (12 cores, 16 threads)
            if platform.machine() == "AMD64":  # Intel x64
                pbar.set_description("Configuring for Intel i7-1260P...")
                pbar.update(1)
                model = WhisperModel(
                    model_name, 
                    device="cpu", 
                    compute_type="int8",  # Optimal for Intel performance cores
                    cpu_threads=PHYSICAL_CORES,  # Use all physical cores
                    num_workers=1,  # Single worker per model instance
                    local_files_only=not allow_downloads
                )
            elif platform.machine() == "arm64":  # Apple Silicon
                pbar.set_description("Configuring for Apple Silicon...")
                pbar.update(1)
                model = WhisperModel(
                    model_name, 
                    device="cpu", 
                    compute_type="int8",  # Stable performance on Apple Silicon
                    cpu_threads=7,  # 6 performance + 1 efficiency core
                    num_workers=1,  # Single worker per model instance
                    local_files_only=not allow_downloads
                )
            else:  # Fallback
                pbar.set_description("Configuring for generic CPU...")
                pbar.update(1)
                model = WhisperModel(model_name, device="cpu", compute_type="int8", 
                                   local_files_only=not allow_downloads)
            
            pbar.set_description("Model loaded successfully")
            pbar.update(1)
        return model

    if backend == "pho-whisper":
        # detect if it's ct2 or not
        if model_name.endswith("-ct2"):
            from faster_whisper import WhisperModel
            with tqdm(desc="Loading Pho-Whisper CT2 model", unit="step", leave=False) as pbar:
                pbar.set_description("Initializing Pho-Whisper CT2...")
                pbar.update(1)
                model = WhisperModel(model_name, device="cpu", compute_type="int8",
                                    local_files_only=not allow_downloads)
                pbar.set_description("Model loaded successfully")
                pbar.update(1)
            return ("pho-ct2", model)
        else:
            try:
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            except ImportError:
                print("\nInstalling 'transformers' package...")
                import subprocess as _subprocess
                _subprocess.check_call(["pip", "install", "transformers"])
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            import torch
            
            with tqdm(desc="Loading Pho-Whisper HF model", unit="step", leave=False) as pbar:
                pbar.set_description("Loading processor...")
                pbar.update(1)
                processor = AutoProcessor.from_pretrained(model_name, local_files_only=not allow_downloads)
                
                pbar.set_description("Loading model weights...")
                pbar.update(1)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, dtype=torch.float32,
                                                                  local_files_only=not allow_downloads).to("cpu")
                
                pbar.set_description("Model loaded successfully")
                pbar.update(1)
            return ("pho-hf", (processor, model))


    if backend == "hf-whisper":
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        import torch
        
        with tqdm(desc="Loading HuggingFace Whisper model", unit="step", leave=False) as pbar:
            pbar.set_description("Loading processor...")
            pbar.update(1)
            processor = AutoProcessor.from_pretrained(model_name, local_files_only=not allow_downloads)
            
            pbar.set_description("Loading model weights...")
            pbar.update(1)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, torch_dtype=torch.float32, local_files_only=not allow_downloads
            ).to("cpu")
            
            pbar.set_description("Model loaded successfully")
            pbar.update(1)
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
    """Transcribe using faster-whisper with optimized settings for Intel i7-1260P."""
    # Extract audio to WAV format for better compatibility
    print(f" Extracting audio from {os.path.basename(file_path)}...")
    wav_path = extract_wav(file_path)
    print(f" Audio extracted to WAV format")
    
    # Optimize settings for Intel i7-1260P performance
    print(f" Running faster-whisper transcription with Intel optimizations...")
    
    import platform
    
    # Intel i7-1260P optimized settings (12 cores, 16 threads)
    if platform.machine() == "AMD64":  # Intel x64
        segments, info = model.transcribe(
            wav_path,
            language=lang or None,
            beam_size=5,  # Balanced for Intel performance cores
            best_of=3,    # Multiple candidates for better accuracy
            temperature=[0.0, 0.2, 0.4],  # Reduced temperature range for speed
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=400,  # Optimized for Intel
                threshold=0.5,
            ),
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt=None,
            # Intel-specific optimizations
            chunk_length=OPTIMAL_CHUNK_SIZE,  # Dynamic chunk size based on memory
            without_timestamps=False,
        )
    elif platform.machine() == "arm64":  # Apple Silicon optimizations
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
    else:  # Fallback settings
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
    
    print(f" Processing transcription results...")
    # Convert segments to our format with progress bar
    print(f" Converting segments to list format...")
    segments_list = []
    # Use tqdm without total since we don't know the count upfront (lazy iterator)
    with tqdm(desc="Processing segments", unit="seg", leave=False) as pbar:
        for seg in segments:
            segments_list.append((seg.start, seg.end, seg.text))
            # Update progress bar less frequently for better performance
            if len(segments_list) % 10 == 0 or len(segments_list) == 1:
                pbar.set_postfix({"count": len(segments_list), "duration": f"{seg.end:.1f}s"})
            pbar.update(1)
    print(f" Converted {len(segments_list)} segments")
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
    print(f" Extracting audio from {os.path.basename(file_path)}...")
    wav_path = extract_wav(file_path)
    print(f" Audio extracted to WAV format")
    
    # Optimize settings for Vietnamese and Apple Silicon performance
    print(f" Running faster-whisper transcription with streaming...")
    
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
    
    print(f" Streaming segments to files...")
    # Stream segments directly to files with progress bar
    segments_for_splitting = []  # Collect for subtitle splitting only
    
    # Get total duration for progress tracking
    total_duration = 0
    if hasattr(info, 'duration') and info.duration:
        total_duration = info.duration
    elif segments:
        # Estimate from last segment if info not available
        for seg in segments:
            total_duration = max(total_duration, seg.end)
    
    if total_duration > 0:
        print(f" Total duration: {total_duration:.1f}s")
        with tqdm(total=total_duration, desc="Transcribing", unit="s", leave=False) as pbar:
            for seg in segments:
                # Write segment immediately to files
                writer.write_segment(seg.start, seg.end, seg.text)
                
                # Also collect for subtitle splitting later if needed
                segments_for_splitting.append((seg.start, seg.end, seg.text))
                
                # Update progress based on segment end time
                pbar.n = min(seg.end, total_duration)
                pbar.set_postfix({
                    "segments": len(segments_for_splitting), 
                    "time": f"{seg.end:.1f}s",
                    "speed": f"{seg.end/pbar.n*100:.1f}%" if pbar.n > 0 else "0%"
                })
                pbar.refresh()
    else:
        # Fallback to segment-based progress if duration unknown
        with tqdm(desc="Streaming segments", unit="seg", leave=False) as pbar:
            for seg in segments:
                # Write segment immediately to files
                writer.write_segment(seg.start, seg.end, seg.text)
                
                # Also collect for subtitle splitting later if needed
                segments_for_splitting.append((seg.start, seg.end, seg.text))
                
                # Update progress
                pbar.set_postfix({"count": len(segments_for_splitting), "duration": f"{seg.end:.1f}s"})
                pbar.update(1)
    
    print(f" Streamed {len(segments_for_splitting)} segments to files")
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
def check_existing_transcripts(file_path: str, save_srt: bool = False, save_json: bool = False, save_vtt: bool = False, transcript_folder: str = "./transcripts") -> tuple[bool, str]:
    """
    Check if transcription files already exist for a video in the transcripts folder.
    Returns (exists, existing_folder_path)
    """
    video_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Look in the transcripts folder
    if not os.path.isabs(transcript_folder):
        transcripts_dir = os.path.join(PROJECT_ROOT, transcript_folder.lstrip('./'))
    else:
        transcripts_dir = transcript_folder
    
    # If transcripts folder doesn't exist yet, no existing transcripts
    if not os.path.exists(transcripts_dir):
        return False, ""
    
    # Look for existing transcription folders
    for item in os.listdir(transcripts_dir):
        if item.startswith(f"{video_name}_transcription_") and os.path.isdir(os.path.join(transcripts_dir, item)):
            existing_folder = os.path.join(transcripts_dir, item)
            
            # Check if required output files exist
            base_name = os.path.join(existing_folder, video_name)
            files_exist = []
            
            if save_srt and os.path.exists(f"{base_name}.srt"):
                files_exist.append("SRT")
            if save_vtt and os.path.exists(f"{base_name}.vtt"):
                files_exist.append("VTT")
            if save_json and os.path.exists(f"{base_name}.json"):
                files_exist.append("JSON")
            if os.path.exists(f"{base_name}.txt"):
                files_exist.append("TXT")
            
            # If we have at least the TXT file and any requested format files, consider it complete
            if "TXT" in files_exist and len(files_exist) > 1:
                return True, existing_folder
    
    return False, ""


def process_file(file_path: str, backend: str, model_name: str, lang: str,
                 save_srt=False, save_json=False, save_vtt=False, max_sub_len: float = 8.0, allow_downloads: bool = True, skip_existing: bool = True, transcript_folder: str = "./transcripts"):
    # Check if transcription already exists
    if skip_existing:
        exists, existing_folder = check_existing_transcripts(file_path, save_srt, save_json, save_vtt, transcript_folder)
        if exists:
            fname = os.path.basename(file_path)
            print(f" Skipping {fname} - transcription already exists in {os.path.basename(existing_folder)}")
            return
    
    # Optimize process priority on macOS
    import platform
    if platform.machine() == "arm64":
        try:
            # Set higher priority for transcription process
            os.nice(-5)  # Higher priority (requires admin on some systems)
        except (OSError, PermissionError):
            pass  # Continue if can't set priority
    
    fname = os.path.basename(file_path)
    file_start_time = time.time()
    
    print(f" Processing {fname} ...")
    
    # Only create transcript folder structure when we're actually going to transcribe
    # Create output directory structure in transcripts folder
    video_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create transcripts folder if it doesn't exist (only when processing)
    if not os.path.isabs(transcript_folder):
        transcripts_dir = os.path.join(PROJECT_ROOT, transcript_folder.lstrip('./'))
    else:
        transcripts_dir = transcript_folder
    os.makedirs(transcripts_dir, exist_ok=True)
    
    # Create video-specific output folder in transcripts directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Shorten video name to avoid Windows path length issues
    short_video_name = video_name[:30] if len(video_name) > 30 else video_name
    output_folder_name = f"{short_video_name}_transcription_{timestamp}"
    out_dir = os.path.join(transcripts_dir, output_folder_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Use shortened name for base file name as well
    base_name = os.path.join(out_dir, short_video_name)
    
    print(f" Output folder: transcripts/{output_folder_name}")

    # Load model in this worker process to avoid pickling issues
    print(f" Loading {backend} model '{model_name}' for {fname}...")
    model_data = load_backend(backend, model_name, allow_downloads)
    print(f" Model loaded for {fname}")
    
    # Create streaming writer for real-time output
    writer = StreamingFileWriter(base_name)
    writer.open_files(save_srt=save_srt, save_vtt=save_vtt, save_json=save_json)
    
    try:
        print(f" Starting streaming transcription for {fname}...")
        
        if backend == "faster-whisper":
            # Use streaming transcription for faster-whisper
            segments = transcribe_faster_whisper_streaming(model_data, file_path, writer, lang, max_sub_len)
            print(f" Streaming transcription complete for {fname} - {len(segments)} segments")
            
            # Optional: Create split subtitles if needed (but basic files are already written)
            if segments and max_sub_len < 8.0:  # Only if custom split length requested
                print(f" Creating split subtitles...")
                split_segments = split_for_subtitles(segments, max_len=max_sub_len)
                if save_srt:
                    write_srt(base_name + "_split.srt", split_segments)
                if save_vtt:
                    write_vtt(base_name + "_split.vtt", split_segments)
        else:
            # Fall back to regular transcription for other backends
            text, segments = transcribe(backend, model_data, file_path, lang)
            print(f" Transcription complete for {fname} - {len(segments) if segments else 0} segments")
            
            # Write to files manually for non-streaming backends
            if segments:
                for start, end, text_seg in segments:
                    writer.write_segment(start, end, text_seg)
    
    finally:
        # Always close files
        writer.close_files()
    
    file_time = time.time() - file_start_time
    print(f" All output files saved for {fname} (completed in {file_time:.2f}s)")

# ------------------------------
# Interactive Menus
# ------------------------------
def interactive_main_menu():
    """Main menu to choose operation mode"""
    print("\n" + "="*60)
    print("VIDEO TRANSCRIPTION TOOL")
    print("="*60)
    print("\nWhat would you like to do?")
    print(" 1. Download YouTube video and transcribe")
    print(" 2. Transcribe existing video files")
    print(" 3. Download YouTube video only (no transcription)")
    print(" 4. Advanced settings")
    print(" 5. Configure yt-dlp settings")
    print(" 6. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        if choice == "1":
            return "download_and_transcribe"
        elif choice == "2":
            return "transcribe_only"
        elif choice == "3":
            return "download_only"
        elif choice == "4":
            return "advanced_settings"
        elif choice == "5":
            return "ytdlp_config"
        elif choice == "6":
            return "exit"
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")

def interactive_youtube_url_input():
    """Get YouTube URL(s) from user"""
    print("\n YouTube Video Download")
    print("-" * 30)
    urls = []
    
    while True:
        if not urls:
            url = input("Enter YouTube URL: ").strip()
        else:
            url = input("Enter another YouTube URL (or press Enter to continue): ").strip()
            if not url:
                break
        
        if url:
            # Basic URL validation
            if "youtube.com" in url or "youtu.be" in url:
                urls.append(url)
                print(f" Added: {url}")
            else:
                print(" Invalid YouTube URL. Please enter a valid YouTube link.")
        
        if not urls:
            print(" Please enter at least one valid YouTube URL.")
    
    return urls

def interactive_ytdlp_selection():
    """Let user select yt-dlp configuration for downloads"""
    print("\n yt-dlp Configuration Selection")
    print("-" * 30)
    print("Choose download configuration:")
    print(" 1. Use default configuration")
    print(" 2. Select preset configuration")
    print(" 3.  Select quality preset")
    print(" 4. Use custom configuration")
    print(" 5. Load from saved file")
    
    while True:
        choice = input("\nSelect option (1-5): ").strip()
        if choice == "1":
            return None, None  # Use default
        elif choice == "2":
            preset_name, config = interactive_ytdlp_config()
            return preset_name, config
        elif choice == "3":
            return interactive_quality_selection()
        elif choice == "4":
            return "custom", create_custom_ytdlp_config()
        elif choice == "5":
            config = load_ytdlp_config()
            return "file", config
        else:
            print(" Invalid choice. Please enter 1-5.")

def interactive_folder_selection():
    """Let user select or specify folder for video files"""
    print("\n Video Folder Selection")
    print("-" * 30)
    print(f" 1. Use default folder: {HARDCODED_FOLDER}")
    print(" 2. Specify custom folder")
    print(" 3. Use current directory")
    
    while True:
        choice = input("\nSelect folder option (1-3): ").strip()
        if choice == "1":
            folder = HARDCODED_FOLDER
            break
        elif choice == "2":
            folder = input("Enter folder path: ").strip()
            if not folder:
                print(" Please enter a valid folder path.")
                continue
            break
        elif choice == "3":
            folder = "."
            break
        else:
            print(" Invalid choice. Please enter 1, 2, or 3.")
    
    # Check if folder exists
    if not os.path.exists(folder):
        create = input(f"\n❓ Folder '{folder}' doesn't exist. Create it? (y/n): ").strip().lower()
        if create in ['y', 'yes']:
            os.makedirs(folder, exist_ok=True)
            print(f" Created folder: {folder}")
        else:
            print(" Cannot proceed without a valid folder.")
            return interactive_folder_selection()
    
    return folder

def interactive_skip_existing_selection():
    """Ask user whether to skip existing transcriptions"""
    print("\nSkip existing transcriptions?")
    print(" 1. Yes - Skip files that already have transcriptions (recommended)")
    print(" 2. No - Re-transcribe all files")
    
    while True:
        choice = input("\nEnter your choice (1-2): ").strip()
        if choice == "1":
            return True
        elif choice == "2":
            return False
        else:
            print("Invalid choice, try again.")


def interactive_output_format_selection():
    """Let user choose output formats"""
    print("\n Output Format Selection")
    print("-" * 30)
    print("Select which formats to generate:")
    print(" 1. SRT (SubRip subtitles) - Standard subtitle format")
    print(" 2. VTT (WebVTT subtitles) - Web-compatible format")
    print(" 3. JSON (Timeline data) - Machine-readable format")
    print(" 4. TXT (Plain text) - Always included")
    print(" 5. All formats")
    print(" 6. SRT + VTT (most common)")
    
    while True:
        choice = input("\nSelect format option (1-6): ").strip()
        if choice == "1":
            return True, False, False  # srt, vtt, json
        elif choice == "2":
            return False, True, False
        elif choice == "3":
            return False, False, True
        elif choice == "4":
            return False, False, False  # TXT only
        elif choice == "5":
            return True, True, True  # All formats
        elif choice == "6":
            return True, True, False  # SRT + VTT
        else:
            print(" Invalid choice. Please enter 1-6.")

def interactive_advanced_settings():
    """Advanced settings configuration with performance optimization"""
    print("\n Advanced Settings")
    print("-" * 30)
    
    # Show system information
    print(f"System: Intel i7-1260P ({SYSTEM_CORES} cores, {PHYSICAL_CORES} physical)")
    print(f"Memory: {AVAILABLE_MEMORY_GB:.1f}GB available of {TOTAL_MEMORY_GB:.1f}GB total")
    print(f"Optimal workers: {OPTIMAL_WORKERS}")
    print(f"Optimal chunk size: {OPTIMAL_CHUNK_SIZE}s")
    
    settings = {}
    
    # Max subtitle length
    while True:
        max_len = input(f"Max subtitle length in seconds (default: 8.0): ").strip()
        if not max_len:
            settings['max_sub_len'] = 8.0
            break
        try:
            settings['max_sub_len'] = float(max_len)
            if settings['max_sub_len'] <= 0:
                print(" Please enter a positive number.")
                continue
            break
        except ValueError:
            print(" Please enter a valid number.")
    
    # Number of workers with optimization recommendations
    print(f"\nWorker Configuration:")
    print(f"  Recommended: {OPTIMAL_WORKERS} workers (optimized for Intel i7-1260P)")
    print(f"  Maximum: {SYSTEM_CORES} workers (all cores)")
    print(f"  Conservative: {max(1, PHYSICAL_CORES)} workers (physical cores only)")
    
    while True:
        workers = input(f"Number of parallel workers (default: {OPTIMAL_WORKERS}): ").strip()
        if not workers:
            settings['workers'] = OPTIMAL_WORKERS
            break
        try:
            settings['workers'] = int(workers)
            if settings['workers'] <= 0:
                print(" Please enter a positive number.")
                continue
            if settings['workers'] > SYSTEM_CORES:
                print(f" Warning: Using more workers ({settings['workers']}) than CPU cores ({SYSTEM_CORES}) may reduce performance.")
                confirm = input(" Continue anyway? (y/n): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    continue
            break
        except ValueError:
            print(" Please enter a valid number.")
    
    # Transcript folder location
    while True:
        transcript_folder = input(f"Transcript output folder (default: ./transcripts): ").strip()
        if not transcript_folder:
            settings['transcript_folder'] = "./transcripts"
            break
        # Validate folder path
        if os.path.isabs(transcript_folder) or transcript_folder.startswith('./') or transcript_folder.startswith('../'):
            settings['transcript_folder'] = transcript_folder
            break
        else:
            print(" Please enter a valid folder path (e.g., ./transcripts, /path/to/folder)")
    
    return settings

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
                      save_vtt: bool = False, workers: int = None, max_sub_len: float = 8.0, skip_existing: bool = True, transcript_folder: str = "./transcripts"):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp4")]
    if not files:
        print(" No MP4 files found")
        return
    
    # Don't load model here - let each worker process load it to avoid pickling issues
    total_start = time.time()
    
    # Use optimized worker configuration
    workers = workers or OPTIMAL_WORKERS
    workers = min(workers, len(files))  # Don't use more workers than files
    
    print(f"Processing {len(files)} video file(s) with {workers} worker(s)...")
    print(f"Backend: {backend}, Model: {model_name}, Language: {lang or 'auto'}")
    print(f"System: {SYSTEM_CORES} cores, {AVAILABLE_MEMORY_GB:.1f}GB RAM available")
    print(f"Optimized for Intel i7-1260P: {PHYSICAL_CORES} physical cores")
    
    # Set optimal multiprocessing method
    import multiprocessing as mp
    import platform
    
    # Use spawn method on Windows (default) for better stability
    if platform.system() == "Windows":
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    elif platform.machine() == "arm64" and hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method('fork', force=True)  # Faster than spawn on macOS
        except RuntimeError:
            pass  # Already set
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, f, backend, model_name, lang,
                                   save_srt, save_json, save_vtt, max_sub_len, allow_downloads, skip_existing, transcript_folder): f for f in files}
        
        # Enhanced progress tracking with resource monitoring
        completed_files = []
        failed_files = []
        
        def get_system_stats():
            """Get current system resource usage."""
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                return f"CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%"
            except:
                return "CPU: N/A | RAM: N/A"
        
        with tqdm(total=len(files), desc="Processing videos", unit="video", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for future in as_completed(futures):
                file_path = futures[future]
                file_name = os.path.basename(file_path)
                
                try:
                    future.result()
                    completed_files.append(file_name)
                    pbar.set_postfix({
                        'completed': len(completed_files),
                        'failed': len(failed_files),
                        'current': file_name[:20] + '...' if len(file_name) > 20 else file_name,
                        'resources': get_system_stats()
                    })
                except Exception as e:
                    failed_files.append((file_name, str(e)))
                    pbar.set_postfix({
                        'completed': len(completed_files),
                        'failed': len(failed_files),
                        'error': file_name[:15] + '...' if len(file_name) > 15 else file_name,
                        'resources': get_system_stats()
                    })
                    print(f"\n Error processing {file_name}: {e}")
                
                pbar.update(1)
    
    # Summary
    total_time = time.time() - total_start
    print(f"\n" + "="*60)
    print(f"PROCESSING COMPLETE")
    print(f"="*60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Files processed: {len(completed_files)}")
    print(f"Files failed: {len(failed_files)}")
    
    if completed_files:
        print(f"\nSuccessfully processed:")
        for file in completed_files:
            print(f"  [OK] {file}")
    
    if failed_files:
        print(f"\nFailed to process:")
        for file, error in failed_files:
            print(f"  [FAIL] {file}: {error}")
    
    if len(completed_files) > 0:
        avg_time = total_time / len(completed_files)
        print(f"\nAverage time per file: {avg_time:.2f}s")


# ------------------------------
# CLI
# ------------------------------
def main():
    # Support both CLI and interactive modes
    parser = argparse.ArgumentParser(description="Video Transcription Tool with Interactive Menu", add_help=False)
    parser.add_argument("--help", "-h", action="store_true", help="Show this help message")
    parser.add_argument("--cli", action="store_true", help="Use CLI mode (skip interactive menu)")
    
    # CLI-only arguments (for backwards compatibility)
    parser.add_argument("--folder", default=HARDCODED_FOLDER, help="Folder with MP4 files")
    parser.add_argument("--youtube", help="YouTube URL to download and transcribe")
    parser.add_argument("--backend", choices=["openai-whisper","pho-whisper","hf-whisper","faster-whisper"], help="ASR backend")
    parser.add_argument("--model", help="Model name/id")
    parser.add_argument("--lang", default="vi", help="Language code (en, vi, ja, ko, zh)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--srt", action="store_true", help="Save .srt subtitle file")
    parser.add_argument("--vtt", action="store_true", help="Save .vtt subtitle file")
    parser.add_argument("--json", action="store_true", help="Save .json timeline file")
    parser.add_argument("--max-sub-len", type=float, default=8.0, help="Max subtitle segment length in seconds")
    parser.add_argument("--allow-downloads", dest="allow_downloads", action="store_true", default=True, help="Allow model downloads")
    parser.add_argument("--no-downloads", dest="allow_downloads", action="store_false", help="Disable model downloads")
    
    # yt-dlp configuration arguments
    parser.add_argument("--ytdlp-preset", choices=list(YTDLP_PRESETS.keys()), help="yt-dlp configuration preset")
    parser.add_argument("--ytdlp-config", help="Path to custom yt-dlp configuration JSON file")
    parser.add_argument("--quality", choices=["4k_ultra", "1080p_hd", "720p_hd", "480p_sd", "360p_low", "240p_minimal"], 
                       help="Quick quality selection preset")
    
    args = parser.parse_args()
    
    if args.help:
        parser.print_help()
        return
    
    # Use CLI mode if specified or if YouTube URL provided directly
    if args.cli or args.youtube:
        run_cli_mode(args)
    else:
        run_interactive_mode()

def run_cli_mode(args):
    """Run in CLI mode with performance optimizations"""
    # Handle YouTube download if URL is provided
    if args.youtube:
        print(f" YouTube URL provided: {args.youtube}") 
        
        # Load yt-dlp configuration
        custom_config = None
        preset_name = args.ytdlp_preset
        
        # Handle quality argument (takes precedence over preset)
        if args.quality:
            preset_name = args.quality
            print(f" Using quality preset: {args.quality}")
        elif args.ytdlp_config:
            try:
                with open(args.ytdlp_config, 'r', encoding='utf-8') as f:
                    custom_config = json.load(f)
                print(f" Loaded custom yt-dlp config from: {args.ytdlp_config}")
            except Exception as e:
                print(f" Error loading yt-dlp config: {e}")
                return
        
        try:
            downloaded_file = download_youtube_video(args.youtube, args.folder, preset_name, custom_config)
            print(f" Video downloaded to: {downloaded_file}")
        except Exception as e:
            print(f" Failed to download video: {e}")
            return

    lang = args.lang or interactive_language_menu()
    if not args.backend or not args.model:
        backend, model = interactive_model_menu(lang or "en")
    else:
        backend, model = args.backend, args.model

    # Use optimized worker count if not specified
    workers = args.workers or OPTIMAL_WORKERS

    print(f"\n=== PERFORMANCE CONFIGURATION ===")
    print(f"System: Intel i7-1260P ({SYSTEM_CORES} cores, {PHYSICAL_CORES} physical)")
    print(f"Memory: {AVAILABLE_MEMORY_GB:.1f}GB available")
    print(f"Workers: {workers} (optimized for your system)")
    print(f"Chunk size: {OPTIMAL_CHUNK_SIZE}s")
    
    print(f"\n=== TRANSCRIPTION SETTINGS ===")
    print(f"Backend: {backend}")
    print(f"Model:   {model}")
    print(f"Lang:    {lang or 'auto'}")
    print(f"Folder:  {args.folder}")
    print(f"Outputs: srt={'on' if args.srt else 'off'}, vtt={'on' if args.vtt else 'off'}, json={'on' if args.json else 'off'}")
    print(f"Max sub len: {args.max_sub_len}s")
    print(f"Output dir: transcripts/ (centralized transcript folder)")

    transcribe_videos(
        args.folder,
        backend,
        model,
        lang=lang,
        save_srt=args.srt,
        allow_downloads=args.allow_downloads,
        save_json=args.json,
        save_vtt=args.vtt,
        workers=workers,
        max_sub_len=args.max_sub_len,
        transcript_folder="./transcripts"
    )

def run_interactive_mode():
    """Run in interactive menu mode"""
    advanced_settings = {
        'max_sub_len': 8.0,
        'workers': None
    }
    
    while True:
        operation = interactive_main_menu()
        
        if operation == "exit":
            print(" Thank you for using the Video Transcription Tool!")
            break
            
        elif operation == "advanced_settings":
            advanced_settings.update(interactive_advanced_settings())
            print(f"\n Advanced settings updated:")
            print(f"  - Max subtitle length: {advanced_settings['max_sub_len']}s")
            print(f"  - Workers: {advanced_settings['workers'] or 'auto'}")
            print(f"  - Transcript folder: {advanced_settings.get('transcript_folder', './transcripts')}")
            input("\nPress Enter to continue...")
            continue
            
        elif operation == "download_only":
            # Download YouTube videos without transcription
            urls = interactive_youtube_url_input()
            folder = interactive_folder_selection()
            preset_name, ytdlp_config = interactive_ytdlp_selection()
            
            print(f"\n Downloading {len(urls)} video(s) to {folder}...")
            for url in urls:
                try:
                    downloaded_file = download_youtube_video(url, folder, preset_name, ytdlp_config)
                    print(f" Downloaded: {os.path.basename(downloaded_file)}")
                except Exception as e:
                    print(f" Failed to download {url}: {e}")
            
            input("\nPress Enter to continue...")
            
        elif operation == "ytdlp_config":
            # Configure yt-dlp settings
            preset_name, config = interactive_ytdlp_config()
            if preset_name and config:
                print(f"\n yt-dlp configuration set to: {preset_name}")
                save_choice = input("Save this configuration to file? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes']:
                    save_ytdlp_config(config)
            input("\nPress Enter to continue...")
            
        elif operation in ["download_and_transcribe", "transcribe_only"]:
            # Get folder and optionally download videos
            if operation == "download_and_transcribe":
                urls = interactive_youtube_url_input()
                folder = interactive_folder_selection()
                preset_name, ytdlp_config = interactive_ytdlp_selection()
                
                # Download videos first
                print(f"\n Downloading {len(urls)} video(s)...")
                for url in urls:
                    try:
                        downloaded_file = download_youtube_video(url, folder, preset_name, ytdlp_config)
                        print(f" Downloaded: {os.path.basename(downloaded_file)}")
                    except Exception as e:
                        print(f" Failed to download {url}: {e}")
            else:
                folder = interactive_folder_selection()
            
            # Get transcription settings
            lang = interactive_language_menu()
            backend, model = interactive_model_menu(lang or "en")
            save_srt, save_vtt, save_json = interactive_output_format_selection()
            skip_existing = interactive_skip_existing_selection()
            
            # Show configuration
            print(f"\nConfiguration Summary:")
            print(f"Backend: {backend}")
            print(f"Model: {model}")
            print(f"Language: {lang or 'auto'}")
            print(f"Folder: {folder}")
            print(f"Formats: SRT={save_srt}, VTT={save_vtt}, JSON={save_json}, TXT=True")
            print(f"Skip existing: {skip_existing}")
            print(f"Max subtitle length: {advanced_settings['max_sub_len']}s")
            print(f"Workers: {advanced_settings['workers'] or 'auto'}")
            print(f"Transcript folder: {advanced_settings.get('transcript_folder', './transcripts')}")
            
            confirm = input("\nStart transcription? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                try:
                    transcribe_videos(
                        folder,
                        backend,
                        model,
                        lang=lang,
                        save_srt=save_srt,
                        save_json=save_json,
                        save_vtt=save_vtt,
                        workers=advanced_settings['workers'],
                        max_sub_len=advanced_settings['max_sub_len'],
                        skip_existing=skip_existing,
                        transcript_folder=advanced_settings.get('transcript_folder', './transcripts')
                    )
                except KeyboardInterrupt:
                    print("\n Transcription interrupted by user.")
                except Exception as e:
                    print(f"\n Error during transcription: {e}")
            else:
                print(" Transcription cancelled.")
            
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
