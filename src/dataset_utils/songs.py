YT_SONG_LINKS = [
    "https://www.youtube.com/watch?v=JH398xAYpZA",
    "https://www.youtube.com/watch?v=0T4UykXuJnI",
    "https://www.youtube.com/watch?v=E6zblNbGXA4",
    "https://www.youtube.com/watch?v=IB_FP_rEih4",
    "https://www.youtube.com/watch?v=dqt8Z1k0oWQ",
    "https://www.youtube.com/watch?v=-ZuS0p2qRYo",
    "https://www.youtube.com/watch?v=GfiJowcJiVw",
    "https://www.youtube.com/watch?v=B3J6tQTuubc",
    "https://www.youtube.com/watch?v=WL-3zvnUomk",
    "https://www.youtube.com/watch?v=Rif-RTvmmss",
    "https://www.youtube.com/watch?v=-uj9b9JCIJM",
    "https://www.youtube.com/watch?v=j3OzH8URrng",
    "https://www.youtube.com/watch?v=plRha7E_2y4",
    "https://www.youtube.com/watch?v=j9Hije4z6O4",
    "https://www.youtube.com/watch?v=RcS_8-a-sMg",
    "https://www.youtube.com/watch?v=ygTZZpVkmKg",
    "https://www.youtube.com/watch?v=kxgj5af8zg4",
    "https://www.youtube.com/watch?v=5v1TOFULOWA",
    "https://www.youtube.com/watch?v=g8TLU_JxCjc",
    "https://www.youtube.com/watch?v=u6lihZAcy4s",
    "https://www.youtube.com/watch?v=LKsgDcckur0",
    "https://www.youtube.com/watch?v=fHI8X4OXluQ",
    "https://www.youtube.com/watch?v=i4ZuseKFBF0",
    "https://www.youtube.com/watch?v=cJVpXPSXGtk",
    "https://www.youtube.com/watch?v=mTLQhPFx2nM",
]


def main() -> None:
    """
    Compare files in ./dataset/studio and ./dataset/live and print names
    that are only in one of the two directories.
    """

    import os

    def names_without_ext(path):
        items = []
        for entry in os.listdir(path):
            full = os.path.join(path, entry)
            if os.path.isfile(full):
                base, _ = os.path.splitext(entry)
                items.append(base)
        return set(items)

    live_names = names_without_ext("./dataset/live")
    studio_names = names_without_ext("./dataset/studio")

    only_live = sorted(live_names - studio_names)
    only_studio = sorted(studio_names - live_names)

    print("Only in live:")
    for n in only_live:
        print(n)

    print("\nOnly in studio:")
    for n in only_studio:
        print(n)


if __name__ == "__main__":
    main()
