import yt_dlp
import ffmpeg
import os

SONGS = [
    ("Opening", 0),
    ("RIP Fredo (Notice Me)", 1 * 60 + 29),
    ("Cancun", 3 * 60 + 8),
    ("Let It Go", 4 * 60 + 29),
    ("Wokeuplikethis*", 7 * 60 + 30),
    ("Half & Half", 10 * 60 + 47),
    ("FlatBed Freestyle", 11 * 60 + 39),
    ("Shoota", 14 * 60 + 51),
    ("Dothatshit!", 17 * 60 + 26),
    ("Of Course We Ghetto Flowers", 20 * 60 + 31),
    ("RIP", 21 * 60 + 31),
    ("Home (KOD)", 24 * 60 + 52),
    ("Yah Mean", 26 * 60 + 41),
    ("Mileage", 30 * 60 + 1),
    ("Fetti", 31 * 60 + 25),
    ("Minute", 32 * 60 + 50),
    ("Lookin", 34 * 60 + 35),
    ("Telephone Calls", 35 * 60 + 19),
    ("Choppa Won't Miss", 36 * 60 + 16),
    ("Love Hurts", 38 * 60 + 28),
    ("Magnolia", 40 * 60 + 20),
    ("Long Time (Intro)", 43 * 60 + 15),
    ("Broke Boi", 46 * 60 + 41),
    ("Take a Step Back (#llj)", 49 * 60 + 16),
]


CROP_TIMES = {
    "RIP Fredo (Notice Me)": (1, 21),
    "Cancun": (1, 30),
    "Let It Go": (2, 60 + 36),
    "Wokeuplikethis*": (13, 27),
    "Half & Half": (1, 0),
    "FlatBed Freestyle": (1, 17),
    # "Shoota": (0.25,),
    # "Dothatshit!": (0.5, 0),
    "Of Course We Ghetto Flowers": (9, 16),
    "Home (KOD)": (1, 23),
    # "Yah Mean": (1, 0),
    "Mileage": (1, 8),
    # "Fetti": (1, 0),
    # "Minute": (1, 0),
    # "Lookin": (1, 0),
    # "Telephone Calls": (1, 0),
    # "Choppa Won't Miss": (0.5, 0),
    "Love Hurts": (1, 6),
    "Magnolia": (0.5, 3.5),
    "Long Time (Intro)": (0.5, 0),
}


def download_full_audio(youtube_url, output_path="full_audio.mp3"):
    """Download full audio from YouTube as MP3."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path.replace(".mp3", ".%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path


def extract_segment(
    input_file, start_time, end_time, crop_start, crop_end, output_file
):
    """Extract a segment from audio with optional cropping."""
    actual_start = start_time + crop_start
    actual_end = end_time - crop_end
    duration = actual_end - actual_start

    if duration <= 0:
        print(f"  Skipping {output_file}: duration <= 0 after cropping")
        return

    (
        ffmpeg.input(input_file, ss=actual_start, t=duration)
        .output(output_file, acodec="copy", loglevel="quiet")
        .overwrite_output()
        .run()
    )
    print(f"  Saved: {output_file} ({duration:.1f}s)")


def sanitize_filename(name):
    """Convert song name to safe filename."""
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name)


if __name__ == "__main__":
    youtube_url = (
        "https://youtu.be/COwiRA5XkHQ?si=ocXmhiZ5bIkTBr8F"  # Replace with actual URL
    )

    print("Downloading full audio from YouTube...")
    full_audio = download_full_audio(youtube_url)

    output_dir = "live_recordings"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExtracting {len(SONGS)} songs...\n")

    for i, (song_name, start_time) in enumerate(SONGS):
        if i < len(SONGS) - 1:
            end_time = SONGS[i + 1][1]
        else:
            end_time = start_time + 5 * 60

        crop_start, crop_end = CROP_TIMES.get(song_name, (0, 0))

        safe_name = sanitize_filename(song_name)
        output_file = os.path.join(output_dir, f"{safe_name}.mp3")

        print(f"[{i+1}/{len(SONGS)}] {song_name}")
        print(
            f"  Time: {start_time//60}:{start_time%60:02d} - {end_time//60}:{end_time%60:02d}"
        )

        extract_segment(
            full_audio, start_time, end_time, crop_start, crop_end, output_file
        )

    print(f"\nDone! Songs saved to '{output_dir}/' folder.")
