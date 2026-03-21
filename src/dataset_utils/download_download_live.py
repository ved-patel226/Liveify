# *NOTE - ONLY WORKS WITH YT PREMIUM

import yt_dlp
import ffmpeg
import os

include_list = {
    "Marvin's Room": {"crop": (0, 0)},
    "Teenage Fever": {"crop": (20, 0)},
    "Feel No Ways": {"crop": (0, 0)},
    "Headlines": {"crop": (13, 0)},
    "The Motto": {"crop": (1, 9)},
    "Started From the Bottom": {"crop": (0, 0)},
    "Energy": {"crop": (0, 0)},
    "Know Yourself": {"crop": (0, 0)},
    "Nonstop": {"crop": (0, 0)},
    # "SICKO MODE": {"crop": (14, 0)},
    "No Face": {"crop": (0, 0)},
    "Circadian Rhythm": {"crop": (0.5, 0)},
    # Remixes roughly start =====
    # "Laugh Now Cry Later": {"crop": (0, 0)},
    # "God's Plan (Performed In The Crowd)": {"crop": (0, 0)},
    # "Controlla": {"crop": (17.5, 0)},
    "Hold On, We're Going Home": {"crop": (16, 2)},
    # "No Guidance": {"crop": (0, 0)},
    "One Dance": {"crop": (0, 0)},  # if it doesn't work, try removing
    # Remixes roughly end =====
    "Massive": {"crop": (0, 20)},
    "Fake Love": {"crop": (2, 20)},
    # "Hotline Bling": {"crop": (0, 0)},
    # "Child's Play": {"crop": (0, 0)},
    "In My Feelings": {"crop": (1, 0)},
    "Nice for What": {"crop": (0, 0)},
    "Girls Want Girls": {"crop": (2, 0)},
    "Search & Rescue": {"crop": (1, 2)},
    # "You Broke My Heart": {"crop": (1, 0)},
    "What's Next": {"crop": (2, 0)},
    # "IDGAF": {"crop": (0.5, 0)},
    "Jimmy Cooks": {"crop": (1.5, 0)},
    # "Knife Talk": {"crop": (0.5, 0)},
    "Rich Flex": {"crop": (0, 30)},
    # "Yebba's Heartbreak": {"crop": (0.5, 0)},
}


def get_chapters(youtube_url):
    ydl_opts = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info.get("chapters", [])


def download_full_audio(youtube_url, output_path="full_audio.mp3"):
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


def crop_chapter(input_file, start, end, crop_start, crop_end, output_file):
    actual_start = start + crop_start
    actual_end = end - crop_end
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
    print(f"  Saved: {output_file}")


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=MGEWtssL4IA"

    print("Fetching chapters...")
    chapters = get_chapters(url)

    print("Downloading full audio...")
    full_audio = download_full_audio(url)

    os.makedirs("chapters", exist_ok=True)

    for chapter in chapters:
        title = chapter["title"]
        if title not in include_list:
            continue

        crop_start, crop_end = include_list[title]["crop"]
        start = chapter["start_time"]
        end = chapter["end_time"]

        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
        output_file = os.path.join("chapters", f"{safe_title}.mp3")

        print(
            f"Processing: {title} ({start:.1f}s - {end:.1f}s, crop {crop_start}s/{crop_end}s)"
        )
        crop_chapter(full_audio, start, end, crop_start, crop_end, output_file)

    print("\nDone! Check the 'chapters' folder.")
