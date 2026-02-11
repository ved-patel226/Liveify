import os
import sys
import ffmpeg


def convert_wav_to_mp3(input_path: str, output_path: str, quality: str = "192") -> bool:
    """WAV file -> MP3 using ffmpeg."""
    try:
        (
            ffmpeg.input(input_path)
            .output(output_path, audio_bitrate=f"{quality}k", codec="libmp3lame")
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except ffmpeg.Error as e:
        print(f"Error converting {input_path}: {e}")
        return False


def main(root=None, quality="192", delete_wav=False):
    if root is None:
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
        )

    converted = 0
    failed = 0

    for sub in ("studio", "live"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue

        for name in os.listdir(d):
            if not name.lower().endswith(".wav"):
                continue

            src = os.path.join(d, name)
            if not os.path.isfile(src):
                continue

            # Create output filename
            base = os.path.splitext(name)[0]
            dst = os.path.join(d, base + ".mp3")

            print(f"Converting: {src} -> {dst}")
            if convert_wav_to_mp3(src, dst, quality):
                converted += 1
                if delete_wav:
                    print(f"Deleting: {src}")
                    os.remove(src)
            else:
                failed += 1

    print(f"\nConversion complete!")
    print(f"Converted: {converted}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    delete_wav = "--delete" in sys.argv
    main(quality="192", delete_wav=delete_wav)
