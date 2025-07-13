import json
import os
from pathlib import Path

import requests

URL = "https://api.svsbusiness.com/engine/api/engine/2b_compose"
ACE_TOKEN = "2dQupMW6j3DsJHv8jWk/C8i6Y/9H9sr2McpV7A7P3CclRGx6sdBUVHLgxB1iQoAAJ+iN+i9qt1BpVJIyq8bsPVIi2OYPcJfe3q5OARzU3wY="
COOPERATOR = "shengmingzhihua"


def process_aces_to_audio(
    ace_path: Path,
    audio_path: Path,
    url=URL,
    ace_token=ACE_TOKEN,
    cooperator=COOPERATOR,
):
    """Process aces files to audio files

    Args:
        ace_path (Path): path to the aces files
        audio_path (Path): path to save the audio files

    Returns:
        None

    """
    audio_path.mkdir(parents=True, exist_ok=True)

    ace_files = sorted(
        ace_path.glob("*.aces"), key=lambda x: int(x.stem.split("_")[-1])
    )

    for i in range(0, len(ace_files), 4):
        batch_ace_files = ace_files[i : min(i + 4, len(ace_files))]
        files = [("file", open(ace_file, "rb")) for ace_file in batch_ace_files]

        data_dict = {
            "ace_token": ace_token,
            "cooperator": cooperator,
            "speaker_id": "0",
        }

        resp = requests.request(
            "POST", url=url, files=files, data=data_dict, timeout=30
        )
        print(resp.text)

        resp_json = json.loads(resp.text)

        for ace_file, audio in zip(batch_ace_files, resp_json["data"]):
            audio_url = audio["audio"]
            audio_file_name = f"{ace_file.stem}.ogg"
            audio_file_path = audio_path / audio_file_name

            if not audio_file_path.exists():
                audio_resp = requests.get(audio_url, timeout=30)
                with open(audio_file_path, "wb") as f:
                    f.write(audio_resp.content)
                print(f"Saved to {audio_file_path}")
            else:
                print(f"{audio_file_path} already exists")


if __name__ == "__main__":
    # Usage example:
    ace_path = Path(f"/home/john/DuiniutanqinSinger/data/aces/temp/1686672")
    audio_path = Path("/home/john/DuiniutanqinSinger/data/audio_ace/temp/1686672")
    # audio_path.parent.parent.mkdir(exist_ok=True)
    # audio_path.parent.mkdir(exist_ok=True)
    # audio_path.mkdir(exist_ok=True)
    # audio path 的 ogg进行清除
    # for f in audio_path.glob("*.ogg"):
    #     os.remove(f)
    process_aces_to_audio(ace_path, audio_path)
