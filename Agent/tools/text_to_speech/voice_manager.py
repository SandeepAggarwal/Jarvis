from pathlib import Path
import requests


VOICE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/"
    "resolve/main/en/en_US/lessac/medium/"
    "en_US-lessac-medium.onnx"
)

VOICE_CONFIG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/"
    "resolve/main/en/en_US/lessac/medium/"
    "en_US-lessac-medium.onnx.json"
)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return

    print(f"Downloading {destination.name}...")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    with destination.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def get_voice() -> Path:
    project_root = Path(__file__).resolve().parent
    voices_dir = project_root / "voices"

    model = voices_dir / "en_US-lessac-medium.onnx"
    config = voices_dir / "en_US-lessac-medium.onnx.json"

    download_file(VOICE_URL, model)
    download_file(VOICE_CONFIG_URL, config)

    return model
