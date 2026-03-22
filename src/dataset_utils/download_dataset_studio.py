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

# ===== DRAKE =====
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

# ===== Carti =====

URLS = [
    "https://youtu.be/t9KTSbKuxi4?si=DDv7pDAwICbPYv0A",  # RIP Fredo (Notice Me)
    "https://youtu.be/rQs81CJAWSc?si=ziF62WeV2uo_zeTr",  # Cancun
    "https://youtu.be/2t5gC28Z_PI?si=UBTQh5PzEtR9oKFo",  # Let it Go
    "https://youtu.be/FnD72g8urKI?si=WrGfZq19Glpws72M",  # Wokeuplikethis*
    "https://youtu.be/jxMfGNzwhVo?si=fcrDnXX9xRoHQlhP",  # Half & Half
    "https://youtu.be/sV_Tlfvt6ig?si=mXQoXOqL3eBQ4gar",  # FlatBed Freestyle
    "https://youtu.be/ruwgaLd83g4?si=VJ6gairIcNn-awtB",  # Of Course We Ghetto Flowers
    "https://youtu.be/-2pjiKmhlAI?si=pt_ONA-nvgkkiRTH",  # Home (KOD)
    "https://youtu.be/5ep6XUPtbHM?si=Ql8dVt6riHfN-NpK",  # Mileage
    "https://youtu.be/8pC-A0KMztk?si=uHXVA-Pnwfcm7qmP",  # Love Hurts
    "https://youtu.be/RLYksQvr5zY?si=OnO36TjH3AOEIqbo",  # Magnolia
    "https://youtu.be/tkPoOvVnbRk?si=qUlbOSY-i5imi1dF",  # Long Time (Intro)
]

URLS = [  # teca
    "https://youtu.be/coqEyxNcJmA?si=JyclWDZldaIQxTWS",  # 500 Lbs
    "https://youtu.be/Csst0G-QfkU?si=UED-taqLY1UEM0cK",  # Dark Thoughts
    "https://youtu.be/0MDHAz_P8lA?si=uVhiZiVa6pOLRZUn",  # Dead or Alive
    "https://youtu.be/AH85mAXA-30?si=akDU1D6mzfTuVmKZ",  # Down with me
    "https://youtu.be/MwYMUSmbEQQ?si=9NNs9xpCnHDWb48O",  # Fell in love
    "https://youtu.be/dLUI4US3Agk?si=lX4MsoPRv9Ag85Z6",  # Half the plot
    "https://youtu.be/_5vDT2e1tRk?si=LXZH8bgvI3Z3GB4E",  # Lot of me
    "https://youtu.be/fLICuTGQMYA?si=g4ZYYuGH-hHio0wk",  # Love me
    "https://youtu.be/EsjDJ8XpIXw?si=JPkLoTf_L6iRYGmo",  # Never Left
    "https://youtu.be/-qBgaVRGzfQ?si=kHytCFsonpM_GsHZ",  # On Your Own
    "https://youtu.be/wTy5dra6CzM?si=TEfek063rKFCScb7",  # OWA OWA
    "https://youtu.be/AALQwjjk85Y?si=TKWQD4OwN8W-zFZX",  # Ransom
    "https://youtu.be/W3bLGdKNxB4?si=p7e1bhQxIkfAPKxi",  # Tic tac toe
]


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
