# Quality Selection Guide

This guide explains how to use the quality selection features in the Video Transcription Tool.

## Quick Quality Selection

### Interactive Mode
1. Run the tool: `python main.py`
2. Select "🔧 Configure yt-dlp settings"
3. Choose "🎯 Quality Selection" (option 7)
4. Select your desired quality:
   - **4K Ultra (2160p)**: Highest quality, largest file size
   - **1080p HD**: High quality, good balance
   - **720p HD**: Medium quality, smaller file
   - **480p SD**: Standard quality, compact file
   - **360p Low**: Lower quality, small file
   - **240p Minimal**: Lowest quality, tiny file
   - **Custom resolution**: Enter any resolution you want

### CLI Mode
Use the `--quality` argument for quick quality selection:

```bash
# Download in 1080p HD
python main.py --youtube "https://youtube.com/watch?v=..." --quality 1080p_hd

# Download in 720p HD
python main.py --youtube "https://youtube.com/watch?v=..." --quality 720p_hd

# Download in 4K Ultra
python main.py --youtube "https://youtube.com/watch?v=..." --quality 4k_ultra
```

## Quality Presets Explained

### 4K Ultra (2160p)
- **Best for**: High-end displays, professional use
- **File size**: Very large (several GB)
- **Download time**: Longest
- **Format**: `bestvideo[height<=2160][ext=mp4]+bestaudio[ext=m4a]`

### 1080p HD
- **Best for**: Most modern displays, good balance
- **File size**: Large (1-3 GB typically)
- **Download time**: Moderate to long
- **Format**: `bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]`

### 720p HD
- **Best for**: Standard HD displays, mobile devices
- **File size**: Medium (500MB-1GB typically)
- **Download time**: Moderate
- **Format**: `bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]`

### 480p SD
- **Best for**: Older displays, bandwidth-limited connections
- **File size**: Small (100-500MB typically)
- **Download time**: Fast
- **Format**: `bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]`

### 360p Low
- **Best for**: Very slow connections, mobile data
- **File size**: Very small (50-200MB typically)
- **Download time**: Very fast
- **Format**: `bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]`

### 240p Minimal
- **Best for**: Extremely slow connections, preview purposes
- **File size**: Tiny (10-50MB typically)
- **Download time**: Fastest
- **Format**: `bestvideo[height<=240][ext=mp4]+bestaudio[ext=m4a]`

## Custom Quality Selection

### Interactive Custom Quality
1. Select "🎯 Quality Selection" → "Custom resolution"
2. Enter your desired resolution height (e.g., 1440 for 1440p)
3. Choose content type:
   - **Video + Audio**: Full video with audio
   - **Video only**: Video without audio
   - **Audio only**: Extract audio only

### Custom Format Strings
For advanced users, you can create custom format strings:

```bash
# Custom 1440p (2K) quality
python main.py --youtube "URL" --ytdlp-preset custom --ytdlp-config custom_config.json
```

Where `custom_config.json` contains:
```json
{
  "format": "bestvideo[height<=1440][ext=mp4]+bestaudio[ext=m4a]/best[height<=1440][ext=mp4]/mp4",
  "restrictfilenames": true
}
```

## Quality vs File Size Guidelines

| Quality | Resolution | Typical File Size | Best For |
|---------|------------|-------------------|----------|
| 4K Ultra | 2160p | 2-8 GB | Professional, high-end displays |
| 1080p HD | 1080p | 500MB-2GB | Most modern displays |
| 720p HD | 720p | 200MB-800MB | Standard HD, mobile |
| 480p SD | 480p | 100MB-400MB | Older displays, slow connections |
| 360p Low | 360p | 50MB-200MB | Mobile data, very slow connections |
| 240p Minimal | 240p | 10MB-50MB | Preview, extremely slow connections |

## Tips for Quality Selection

### For Transcription
- **720p HD** or **1080p HD** are usually sufficient
- Higher quality doesn't improve transcription accuracy
- Consider file size for storage and processing time

### For Archival
- **1080p HD** or **4K Ultra** for long-term storage
- Higher quality preserves more detail for future use

### For Mobile/Portable Use
- **480p SD** or **720p HD** for mobile devices
- Balance between quality and storage space

### For Bandwidth-Limited Connections
- **360p Low** or **480p SD** for slow connections
- Consider downloading during off-peak hours

## Integration with Transcription

Quality selection works seamlessly with transcription:

1. **Download**: Choose your preferred quality
2. **Transcribe**: Quality doesn't affect transcription accuracy
3. **Output**: Generate subtitles in multiple formats

The tool automatically handles different video qualities and extracts audio for transcription regardless of the original video quality.
