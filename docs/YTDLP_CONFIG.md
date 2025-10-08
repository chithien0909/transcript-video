# yt-dlp Configuration Guide

This document explains how to configure yt-dlp settings for the Video Transcription Tool.

## Configuration Methods

### 1. Interactive Configuration
Run the tool and select "🔧 Configure yt-dlp settings" from the main menu to access the interactive configuration system.

### 2. Configuration Presets
The tool includes several built-in presets:

#### General Presets:
- **default**: Balanced quality and speed
- **high_quality**: Best video + audio quality
- **fast_download**: Lower quality, faster downloads
- **with_subtitles**: Downloads subtitles along with video
- **audio_only**: Extracts audio only
- **playlist**: Limits playlist downloads to first 10 videos

#### Quality-Specific Presets:
- **4k_ultra**: 4K Ultra HD (2160p) - Highest quality, largest file
- **1080p_hd**: Full HD (1080p) - High quality, good balance
- **720p_hd**: HD (720p) - Medium quality, smaller file
- **480p_sd**: Standard Definition (480p) - Standard quality, compact file
- **360p_low**: Low quality (360p) - Lower quality, small file
- **240p_minimal**: Minimal quality (240p) - Lowest quality, tiny file

### 3. Configuration File
Create a `ytdlp_config.json` file in the project root with your custom settings:

```json
{
  "format": "best[ext=mp4]/mp4/best",
  "restrictfilenames": true,
  "writeinfojson": false,
  "writesubtitles": false,
  "writeautomaticsub": false,
  "outtmpl": "%(title)s.%(ext)s"
}
```

### 4. CLI Arguments
Use command-line arguments for quick configuration:

```bash
# Use a preset
python main.py --youtube "https://youtube.com/watch?v=..." --ytdlp-preset high_quality

# Use quality preset (quick selection)
python main.py --youtube "https://youtube.com/watch?v=..." --quality 1080p_hd

# Use custom config file
python main.py --youtube "https://youtube.com/watch?v=..." --ytdlp-config my_config.json
```

## Configuration Options

### Format Selection
- `best[ext=mp4]/mp4/best`: Prefer MP4 format
- `bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/mp4`: High quality video + audio
- `worst[ext=mp4]/mp4/worst`: Fast download, lower quality
- `bestaudio[ext=m4a]/bestaudio`: Audio only

### Subtitle Options
- `writesubtitles`: Download manual subtitles
- `writeautomaticsub`: Download auto-generated subtitles
- `subtitleslangs`: List of subtitle languages (e.g., ["en", "vi"])

### Audio Extraction
- `extractaudio`: Extract audio from video
- `audioformat`: Audio format (mp3, m4a, etc.)
- `audioquality`: Audio quality (192K, 320K, etc.)

### Playlist Options
- `playlistend`: Limit number of videos from playlist
- `playliststart`: Start from specific video number

### Output Options
- `outtmpl`: Output filename template
- `restrictfilenames`: Remove special characters from filenames
- `writeinfojson`: Save video metadata as JSON
- `ffmpeg_location`: Path to ffmpeg executable (automatically detected)

## FFmpeg Integration

The tool automatically detects and uses the local ffmpeg executable:

- **Windows**: Uses `./tools/ffmpeg/bin/ffmpeg.exe` if available
- **macOS/Linux**: Falls back to system ffmpeg or imageio-ffmpeg
- **Automatic**: All presets include the correct ffmpeg path

This ensures that video+audio merging works correctly for quality-specific downloads.

## Examples

### Quality-Specific Downloads

#### 4K Ultra HD (2160p)
```json
{
  "format": "bestvideo[height<=2160][ext=mp4]+bestaudio[ext=m4a]/best[height<=2160][ext=mp4]/mp4",
  "restrictfilenames": true,
  "writeinfojson": false
}
```

#### Full HD (1080p)
```json
{
  "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/mp4",
  "restrictfilenames": true,
  "writeinfojson": false
}
```

#### HD (720p)
```json
{
  "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/mp4",
  "restrictfilenames": true,
  "writeinfojson": false
}
```

#### Standard Definition (480p)
```json
{
  "format": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/mp4",
  "restrictfilenames": true,
  "writeinfojson": false
}
```

### High Quality Video Download (Best Available)
```json
{
  "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/mp4",
  "restrictfilenames": true,
  "writeinfojson": false
}
```

### Audio Only with MP3 Extraction
```json
{
  "format": "bestaudio[ext=m4a]/bestaudio",
  "extractaudio": true,
  "audioformat": "mp3",
  "audioquality": "320K",
  "restrictfilenames": true
}
```

### Download with Subtitles
```json
{
  "format": "best[ext=mp4]/mp4/best",
  "writesubtitles": true,
  "writeautomaticsub": true,
  "subtitleslangs": ["en", "vi"],
  "restrictfilenames": true
}
```

### Playlist Download (First 5 Videos)
```json
{
  "format": "best[ext=mp4]/mp4/best",
  "playlistend": 5,
  "restrictfilenames": true
}
```

## Advanced Usage

### Custom Format Strings
You can use advanced yt-dlp format selectors:

```json
{
  "format": "best[height<=720][ext=mp4]/best[height<=480][ext=mp4]/best[ext=mp4]/mp4",
  "restrictfilenames": true
}
```

### Multiple Output Formats
```json
{
  "format": "best[ext=mp4]/mp4/best",
  "outtmpl": "%(uploader)s - %(title)s.%(ext)s",
  "restrictfilenames": true,
  "writeinfojson": true
}
```

## Troubleshooting

### Common Issues
1. **Download fails**: Check if the URL is valid and accessible
2. **Format not available**: Try a more flexible format string
3. **Filename issues**: Ensure `restrictfilenames` is set to `true`
4. **Subtitles not downloading**: Check if subtitles are available for the video

### Debug Mode
Add `"verbose": true` to your configuration to see detailed download information.

## Integration with Transcription

The yt-dlp configuration works seamlessly with the transcription features:

1. Download videos with your preferred settings
2. Automatically transcribe downloaded videos
3. Generate subtitles in multiple formats (SRT, VTT, JSON, TXT)

The tool will use your yt-dlp configuration for all YouTube downloads, whether downloading only or downloading and transcribing.
