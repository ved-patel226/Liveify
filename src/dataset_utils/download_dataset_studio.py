"""Download a list of YouTube videos as MP3s with optional cookies support.

Some videos require authentication (e.g. Music Premium).  Export a
cookies.txt file from your browser and save it alongside this repository
(or adjust :func:`default_cookies_path` to point elsewhere).  The script
will automatically use it if present.

Usage:

    python -m src.dataset_utils.dataset_studio

The MP3 files are written to ``downloads/studio_mp3s`` by default.
"""

import os
from yt_dlp import YoutubeDL


URLS = [
    "https://www.youtube.com/watch?v=Jy64gM-bI1Q",
    "https://www.youtube.com/watch?v=UQNebFZnEQQ",
    "https://www.youtube.com/watch?v=j1vEsY67YBo",
    "https://www.youtube.com/watch?v=tVthyPOWc-E",
    "https://www.youtube.com/watch?v=b8M6N0FTpNc",
    "https://www.youtube.com/watch?v=El2KPBO-_uk",
    "https://www.youtube.com/watch?v=az6m9IE8h4o",
    "https://www.youtube.com/watch?v=jz_01KVkOBI",
    "https://www.youtube.com/watch?v=hzS5TP-Qh5c",
    "https://www.youtube.com/watch?v=5LFB3qdmZBM",
    "https://www.youtube.com/watch?v=KnkDL9lkbX8",
    "https://www.youtube.com/watch?v=_zq1Squ-Uqs",
    "https://www.youtube.com/watch?v=bdg3Zxb9r4g",
    "https://www.youtube.com/watch?v=XNpGNykVZ6U",
    "https://www.youtube.com/watch?v=IV-XT27UOHo",
    "https://www.youtube.com/watch?v=f22c5UkbaOY",
    "https://www.youtube.com/watch?v=SLFN-XfeYXw",
    "https://www.youtube.com/watch?v=xMGBOcUhqu0",
    "https://www.youtube.com/watch?v=0db5QhK9buw",
    "https://www.youtube.com/watch?v=pMaogWC5TEQ",
    "https://www.youtube.com/watch?v=e8HtwsnuTIw",
    "https://www.youtube.com/watch?v=JDb3ZZD4bA0",
    "https://www.youtube.com/watch?v=5LFB3qdmZBM"
    "https://www.youtube.com/watch?v=ReQi9NFgM34",
]

URLS = ["https://www.youtube.com/watch?v=I4DjHHVHWAE"]


def download_mp3(urls, outdir="downloads/studio_mp3s", cookies=None):
    """Download each URL as an MP3 into the target directory.

    If ``cookies`` is a path to a cookies.txt file, it will be passed to
    yt-dlp so that authenticated downloads can succeed.
    """
    os.makedirs(outdir, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(outdir, "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    if cookies:
        ydl_opts["cookiefile"] = cookies

    with YoutubeDL(ydl_opts) as ydl:
        for u in urls:
            try:
                ydl.download([u])
            except Exception as e:  # noqa: BLE001 - broad is okay for logging
                print(f"warning: failed to download {u}: {e}")
                # continue with next URL


def default_cookies_path():
    """Return the path where a cookies.txt file is expected.

    Currently this is simply ``cookies.txt`` in the workspace root; you
    can change the string here if your cookie export lives elsewhere.
    """
    return os.path.join(os.getcwd(), "cookies.txt")


if __name__ == "__main__":
    cookies = default_cookies_path() if os.path.isfile(default_cookies_path()) else None
    download_mp3(URLS, outdir="downloads/studio_mp3s", cookies=cookies)
