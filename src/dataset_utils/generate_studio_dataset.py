import os
import yt_dlp

from songs import YT_SONG_LINKS


def save_songs_and_titles():
    os.makedirs("./dataset/studio", exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "./dataset/studio/%(title)s.%(ext)s",
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in YT_SONG_LINKS:
            try:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "unknown")
                print(f"Downloaded: {title}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")


def main() -> None:
    save_songs_and_titles()
    print(f"Total songs downloaded: {len(os.listdir('./dataset/studio'))}")


if __name__ == "__main__":
    main()
