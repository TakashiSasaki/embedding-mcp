import base64
import io
import math
import struct
import subprocess
import time
import wave
import zlib

from common import ASSETS, peak_rss, server
from text import cosine


def base64_text(data):
    return base64.b64encode(data).decode("ascii")


def image(rgb):
    def chunk(key, data):
        return struct.pack(">I", len(data)) + key + data + struct.pack(">I", zlib.crc32(key + data))

    header = struct.pack(">IIBBBBB", 224, 224, 8, 2, 0, 0, 0)
    pixels = (b"\0" + bytes(rgb) * 224) * 224
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


def audio(hz):
    result = io.BytesIO()
    with wave.open(result, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"".join(struct.pack("<h", round(8000 * math.sin(2 * math.pi * hz * i / 16000)))
                                 for i in range(16000)))
    return result.getvalue()


def video(color):
    return subprocess.check_output([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:s=224x224:r=4:d=1",
        "-an", "-threads", "1", "-c:v", "libx264", "-movflags", "frag_keyframe+empty_moov",
        "-f", "mp4", "pipe:1",
    ])


def part(kind, value):
    return {"content": [{"type": kind, kind: value}]}


def benchmark(out, threads):
    red = part("image_url", {"url": "data:image/png;base64," + base64_text(image((255, 0, 0)))})
    blue = part("image_url", {"url": "data:image/png;base64," + base64_text(image((0, 0, 255)))})
    wav_a, wav_b = (part("input_audio", {"data": base64_text(audio(hz)), "format": "wav"}) for hz in (440, 880))
    clip_a, clip_b = (part("input_video", {"data": base64_text(video(color)), "format": "mp4"})
                      for color in ("red", "blue"))
    mixed = {"content": [{"type": "text", "text": "赤い画像"}, *red["content"], *wav_a["content"]]}
    cases = {
        "text_ja": ("task: search result | query: 赤い四角形", "task: search result | query: 青い四角形"),
        "image": (red, blue),
        "audio": (wav_a, wav_b),
        "video": (clip_a, clip_b),
        "mixed_vs_image": (mixed, red),
    }
    model = ASSETS / "embeddinggemma-2-Q8_0.gguf"
    projector = ASSETS / "mmproj-embeddinggemma-2-Q8_0.gguf"
    results = {}
    with server(model, out / "multimodal.log", threads=threads, context=2048, projector=projector) as client:
        for name, items in cases.items():
            vectors, elapsed = [], []
            for item in items:
                start = time.perf_counter()
                vector = client.post("/v1/embeddings", {"input": [item]})["data"][0]["embedding"]
                elapsed.append(round(1000 * (time.perf_counter() - start), 2))
                assert len(vector) == 768 and all(map(math.isfinite, vector)), name
                assert math.isclose(cosine(vector, vector), 1, abs_tol=1e-4), name
                vectors.append(vector)
            similarity = cosine(*vectors)
            assert similarity < 0.99999, name
            results[name] = {"dimensions": 768, "elapsed_ms": elapsed, "cosine": round(similarity, 6)}
        return {"cases": results, "peak_rss_mib": peak_rss(client.pid)}
