# Output Folder Structure

This document explains the new organized folder structure for downloaded videos and transcriptions.

## Folder Organization

### YouTube Downloads
When downloading YouTube videos, the tool now creates organized folders:

```
videos/
├── Video_Title_20241215_143022/
│   ├── Video Title.mp4
│   └── video_info.json
├── Another_Video_20241215_143155/
│   ├── Another Video.mp4
│   └── video_info.json
└── ...
```

### Transcription Output
When transcribing videos, the tool creates timestamped folders:

```
videos/
├── Video_Title_20241215_143022/
│   ├── Video Title.mp4
│   ├── video_info.json
│   └── Video_Title_transcription_20241215_144530/
│       ├── Video Title.srt
│       ├── Video Title.vtt
│       ├── Video Title.json
│       ├── Video Title.txt
│       └── Video Title_split.srt (if custom split length)
└── ...
```

## Folder Naming Convention

### Download Folders
- **Format**: `{Clean_Title}_{YYYYMMDD_HHMMSS}`
- **Example**: `How_to_Use_Python_20241215_143022`
- **Rules**:
  - Invalid characters (`<>:"/\|?*`) are removed
  - Spaces replaced with underscores
  - Title limited to 50 characters
  - Timestamp added for uniqueness

### Transcription Folders
- **Format**: `{Video_Name}_transcription_{YYYYMMDD_HHMMSS}`
- **Example**: `How_to_Use_Python_transcription_20241215_144530`
- **Purpose**: Separate transcription runs with timestamps

## Video Info File

Each download folder contains a `video_info.json` file with metadata:

```json
{
  "url": "https://youtube.com/watch?v=...",
  "title": "How to Use Python",
  "uploader": "TechChannel",
  "duration": 1200,
  "upload_date": "20241201",
  "view_count": 50000,
  "download_time": "2024-12-15T14:30:22.123456",
  "folder_name": "How_to_Use_Python_20241215_143022"
}
```

## Benefits

### 1. **Organization**
- Each video gets its own folder
- Easy to find specific downloads
- Clear separation between downloads and transcriptions

### 2. **Timestamping**
- Download time recorded
- Transcription time recorded
- Easy to track when content was processed

### 3. **Metadata Preservation**
- Video information saved with each download
- URL, title, uploader, duration preserved
- Useful for reference and organization

### 4. **No Conflicts**
- Timestamps prevent folder name conflicts
- Multiple downloads of same video possible
- Safe for batch processing

## Usage Examples

### Download Single Video
```bash
python main.py --youtube "https://youtube.com/watch?v=..." --quality 1080p_hd
```
**Result**: Creates `Video_Title_20241215_143022/` with video and info file

### Download and Transcribe
```bash
python main.py --youtube "https://youtube.com/watch?v=..." --quality 720p_hd --srt --vtt
```
**Result**: 
1. Creates `Video_Title_20241215_143022/` with video
2. Creates `Video_Title_transcription_20241215_144530/` with subtitles

### Batch Processing
```bash
python main.py --folder ./videos --srt --vtt
```
**Result**: Each video gets its own transcription folder with timestamp

## Folder Structure Comparison

### Before (Old Structure)
```
videos/
├── Video Title.mp4
├── Video Title 2.mp4
└── output/
    ├── Video Title.srt
    ├── Video Title.vtt
    └── Video Title 2.srt
```

### After (New Structure)
```
videos/
├── Video_Title_20241215_143022/
│   ├── Video Title.mp4
│   ├── video_info.json
│   └── Video_Title_transcription_20241215_144530/
│       ├── Video Title.srt
│       ├── Video Title.vtt
│       ├── Video Title.json
│       └── Video Title.txt
└── Video_Title_2_20241215_143155/
    ├── Video Title 2.mp4
    ├── video_info.json
    └── Video_Title_2_transcription_20241215_144700/
        ├── Video Title 2.srt
        ├── Video Title 2.vtt
        ├── Video Title 2.json
        └── Video Title 2.txt
```

## Migration

The new folder structure is automatically applied to:
- New YouTube downloads
- New transcription runs
- Existing videos will use new structure when transcribed

No migration of existing files is needed - the tool will create new organized folders for future operations.
