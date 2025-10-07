## SAA-K22 Transcription Tool

Batch transcribe MP4 recordings in a folder using Whisper or VINAI PhoWhisper.

### Prerequisites
- Python 3.9+
- FFmpeg available (the app will auto-bootstrap a local copy if missing)
- Optional GPU with CUDA for faster inference

### Quick Start
```bash
python main.py --help
```

Defaults:
- Folder: `C:\Users\HP\Documents\docs\SAA-K22\videos`
- Backend: `openai-whisper`
- Whisper model: `medium`
- Language: `en` (set empty for auto-detect)

### Backend Selection
- `--backend openai-whisper`  Use OpenAI Whisper (openai-whisper package)
- `--backend pho-whisper`     Use VINAI PhoWhisper CTranslate2 (faster-whisper)
- `--backend hf-whisper-vi`   Use Hugging Face Transformers VINAI model

### Model Options by Backend
- openai-whisper:
  - `--model <size>` where size is one of: `tiny`, `base`, `small`, `medium`, `large`
- pho-whisper:
  - `--pho-model <HF-ct2-id>` e.g. `vinai/PhoWhisper-small-ct2`, `vinai/PhoWhisper-medium-ct2`, `vinai/PhoWhisper-large-ct2`
- hf-whisper-vi:
  - `--hf-model <HF-id>` e.g. `vinai/PhoWhisper-small`, `vinai/PhoWhisper-medium`, `vinai/PhoWhisper-large`

### Language and Downloads
- `--lang <code>` set preferred language (e.g., `en`, `vi`). Leave empty to auto-detect.
- `--allow-downloads` enable fetching models from the internet if not cached.
- `--no-downloads` require models to be present in the local cache.

### Examples
OpenAI Whisper, English:
```bash
python main.py --backend openai-whisper --model small --lang en
```

VINAI PhoWhisper (CTranslate2), Vietnamese:
```bash
python main.py --backend pho-whisper --pho-model vinai/PhoWhisper-medium-ct2 --lang vi
```

HF Transformers VINAI model, Vietnamese, allow downloads:
```bash
python main.py --backend hf-whisper-vi --hf-model vinai/PhoWhisper-medium --allow-downloads --lang vi
```

### Frame Extraction (auto + manual)
- Auto on startup (default). Skips if cached frames exist.
- Manual extraction only:
```bash
python main.py --extract-frames --fps 1.0 --frames-output frames --image-format jpg --naming sequential
```
Caching:
- Reuse if `_frames_meta.json` and any images exist.
- Force re-extract with `--force-extract`.

### Timeline logging and subtitles
- Print per-segment timeline during/after transcription:
```bash
python main.py --log-timeline
```
- Save SRT alongside transcript:
```bash
python main.py --save-srt
```

### Run everything at once
Auto-extract frames (cached reuse), transcribe, print timeline, and save SRT:
```bash
python main.py --all --folder "C:\Users\HP\Documents\docs\SAA-K22\videos" --frames-output frames --fps 1.0 --image-format jpg --naming sequential --backend openai-whisper --model medium
```

### Outputs
- For each `*.mp4` file, a matching `*.txt` transcript is created in the same folder.
- When timeline segments are available and `--save-srt` is set, a `*.srt` file is also saved.
- Existing transcripts are skipped.

### Notes
- The tool maintains caches under `.cache/` inside the project (including Whisper and HF caches).
- On Windows, FFmpeg is vendored into `tools/ffmpeg/bin/` when auto-bootstrapped.
- If a file path issue occurs, the tool will print a directory listing for debugging.

### Model selection (interactive)
Run without arguments to choose a backend/model interactively, then the tool will (by default) auto-extract frames if missing and transcribe:
```bash
python main.py
```


# transcript-video
