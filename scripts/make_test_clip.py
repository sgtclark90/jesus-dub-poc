"""Generate a synthetic English test clip so the full pipeline can be exercised
end-to-end without a real video. Speaks a few Luke-flavored lines (with key terms),
then renders them over a solid-color video via ffmpeg.

    python scripts/make_test_clip.py
    -> data/input/clip.mp4
"""
import asyncio
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import audioio  # noqa: E402

TEXT = ("Do not be afraid. Today in the town of David a Savior has been born to you. "
        "He is the Messiah, the Lord. This is the good news about Jesus the Christ.")
VOICE = "en-US-GuyNeural"
OUT = "data/input/clip.mp4"


async def _tts(path: str) -> None:
    import edge_tts
    await edge_tts.Communicate(TEXT, VOICE).save(path)


def main() -> None:
    os.makedirs("data/input", exist_ok=True)
    mp3 = "data/input/_voice.mp3"
    wav = "data/input/_voice.wav"
    asyncio.run(_tts(mp3))
    audioio.to_wav(mp3, wav, sr=24000)
    dur = audioio.duration(wav)
    subprocess.run([
        audioio.ffmpeg(), "-y",
        "-f", "lavfi", "-i", f"color=c=midnightblue:s=854x480:rate=25:d={dur:.2f}",
        "-i", wav,
        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        OUT,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(mp3); os.remove(wav)
    print(f"wrote {OUT}  ({dur:.1f}s)")


if __name__ == "__main__":
    main()
