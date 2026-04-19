import os
import pandas as pd
import subprocess
import yt_dlp

# =========================
# CONFIG
# =========================
YOUTUBE_ID = "ID"
EXCEL_FILE = "Untitled spreadsheet (1).xlsx"

VIDEO_DIR = "videos"
CLIP_DIR = "clips"
AUDIO_DIR = "audio"
MANIFEST_FILE = "manifest.csv"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# =========================
# STEP 1: DOWNLOAD VIDEO
# =========================
def download_video(youtube_id):
    output_path = os.path.join(VIDEO_DIR, f"{youtube_id}.mp4")
    
    if os.path.exists(output_path):
        print("✅ Video already downloaded")
        return output_path

    url = f"https://www.youtube.com/watch?v={youtube_id}"

    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4'
    }

    print("⬇️ Downloading video...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path

# =========================
# STEP 2: TIME NORMALIZATION
# =========================
def normalize_time(t):
    if not isinstance(t, str):
        return None

    parts = t.strip().split(":")

    try:
        if len(parts) == 2:
            # MM:SS
            mm, ss = parts
            hh = 0
            ms = 0

        elif len(parts) == 3:
            p1, p2, p3 = parts

            # 🔥 Treat as MM:SS:MS if first < 60
            if int(p1) < 60:
                hh = 0
                mm = int(p1)
                ss = int(p2)
                ms = int(p3)
            else:
                # fallback HH:MM:SS
                hh = int(p1)
                mm = int(p2)
                ss = int(p3)
                ms = 0

        else:
            return None

        return f"{hh:02d}:{mm:02d}:{ss:02d}.{int(ms):03d}"

    except:
        return None

# =========================
# STEP 3: EXTRACT VIDEO CLIP
# =========================
def extract_video(input_video, start, end, output_video):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-ss", start,
        "-to", end,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac",
        output_video
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# =========================
# STEP 4: EXTRACT AUDIO
# =========================
def extract_audio(input_video, output_audio):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-acodec", "pcm_s16le",
        output_audio
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# =========================
# STEP 5: MAIN PIPELINE
# =========================
def process():
    video_path = download_video(YOUTUBE_ID)

    df = pd.read_excel(EXCEL_FILE)

    # Clean data
    df = df.dropna(subset=["Clip Name", "Start Time", "End Time"])

    manifest = []

    for idx, row in df.iterrows():
        clip_name = str(row["Clip Name"]).strip()

        start = normalize_time(str(row["Start Time"]))
        end = normalize_time(str(row["End Time"]))

        if not start or not end:
            print(f"⚠️ Skipping invalid time at row {idx}")
            continue

        video_out = os.path.join(CLIP_DIR, f"{clip_name}.mp4")
        audio_out = os.path.join(AUDIO_DIR, f"{clip_name}.wav")

        print(f"🎬 Processing: {clip_name}")

        # Extract video clip
        extract_video(video_path, start, end, video_out)

        # Extract audio
        extract_audio(video_out, audio_out)

        manifest.append({
            "clip_name": clip_name,
            "text": row.get("Hinglish Text", ""),
            "intent": row.get("INTENT", ""),
            "video_path": video_out,
            "audio_path": audio_out
        })

    # Save manifest
    pd.DataFrame(manifest).to_csv(MANIFEST_FILE, index=False)
    print(f"✅ Manifest saved: {MANIFEST_FILE}")


import pandas as pd

INPUT_FILE = "Untitled spreadsheet (1).xlsx"
OUTPUT_FILE = "Untitled spreadsheet (1).xlsx"

def preprocess():
    df = pd.read_excel(INPUT_FILE)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Ensure Clip Name exists
    if "Clip Name" not in df.columns:
        raise ValueError("❌ 'Clip Name' column not found")

    # Counter per scene
    scene_counter = {}

    new_clip_names = []

    for idx, row in df.iterrows():
        base_name = str(row.get("Clip Name", "")).strip()

        # Skip empty rows
        if base_name == "" or base_name.lower() == "nan":
            new_clip_names.append(None)
            continue

        if base_name not in scene_counter:
            scene_counter[base_name] = 1
        else:
            scene_counter[base_name] += 1

        new_name = f"{base_name}-u{scene_counter[base_name]}"
        new_clip_names.append(new_name)

    # Replace column
    df["Clip Name"] = new_clip_names

    # Drop rows where Clip Name became None
    df = df.dropna(subset=["Clip Name"])

    # Save new file
    df.to_excel(OUTPUT_FILE, index=False)

    print(f"✅ Processed file saved as: {OUTPUT_FILE}")

# if __name__ == "__main__":
#     preprocess()
# # =========================
# RUN
# =========================
if __name__ == "__main__":
    # preprocess()
    process()