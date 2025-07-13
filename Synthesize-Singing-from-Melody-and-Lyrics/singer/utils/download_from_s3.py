# Copyright (c) 2023 MusicBeing Project. All Rights Reserved.
#
# Author: Yongsheng Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import subprocess  # 用于执行外部命令
from pathlib import Path
import mysql.connector

from singer.configs import get_hparams
from .logging import get_logger

# Database configuration


logger = get_logger(__name__)


# Function to download lyrics, bpm midi, midi, and original mp3 from indexed_music table
def download_from_index_music(song_ids: list, rewrite: bool = False) -> None:
    """
    Download lyrics, bpm midi, midi, and original mp3 from indexed_music table.

    Args:
        song_ids (list): List of song IDs to download.
        rewrite (bool): Whether to rewrite the files if they already exist.

    Returns:
        None
    """
    hparams = get_hparams()
    config = hparams["s3_config"]
    mv_path = hparams["mv_path"]
    # Establish database connection
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor(dictionary=True)

    # Prepare SQL query
    query = "SELECT songId, mfa_lyric, s3_bucket, original_s3_path \
        FROM indexed_music WHERE songId IN (%s)"
    placeholders = ", ".join(["%s"] * len(song_ids))
    query = query % placeholders

    try:
        # Execute the query
        cursor.execute(query, tuple(song_ids))

        # Fetch all rows and parse lyrics into a list
        for row in cursor.fetchall():
            song_id = row["songId"]
            s3_bucket = row["s3_bucket"]
            mfa_lyric_jsonl_str = row["mfa_lyric"]
            original_s3_path = row["original_s3_path"]

            song_id_path = Path(f"data/vocal/{song_id}")
            song_id_path.mkdir(parents=True, exist_ok=True)

            midi_file_path = song_id_path / f"{song_id}_origin.mid"
            if rewrite or not midi_file_path.exists():
                mv_midi(song_id, Path(mv_path), midi_file_path)

            json_file_path = song_id_path / f"{song_id}_origin.json"
            if rewrite or not json_file_path.exists():
                save_lyrics_to_json(mfa_lyric_jsonl_str, song_id, json_file_path)

            origin_mp3_file_path = song_id_path / f"{song_id}_origin.mp3"
            if rewrite or not origin_mp3_file_path.exists():
                save_origin_mp3(
                    s3_bucket, song_id, original_s3_path, origin_mp3_file_path
                )

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        # Close the cursor and database connection
        cursor.close()
        cnx.close()


# Function to save lyrics to a JSON file
def save_lyrics_to_json(
    mfa_lyric_jsonl_str: str,
    song_id,
    file_path: Path
) -> None:
    """
    Save lyrics to a JSON file.

    Args:
        mfa_lyric_jsonl_str (str): Lyrics in JSONL format.
        song_id (str): Song ID.
        file_path (Path): File path to save the lyrics.

    Returns:
        None

    """
    # If mfa_lyric_jsonl_str is not empty, parse and append to list
    if mfa_lyric_jsonl_str is not None and mfa_lyric_jsonl_str != "":
        lyrics_list = []
        # Split the JSONL string into separate JSON object strings
        json_obj_strs = mfa_lyric_jsonl_str.strip().split("\n")

        # Parse each JSON object string and append to lyrics_list
        for json_obj_str in json_obj_strs:
            if json_obj_str:  # Skip empty lines
                lyrics_dict = json.loads(json_obj_str)
                lyrics_list.append(lyrics_dict)

        # Save the lyrics_list to a JSON file
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(lyrics_list, json_file, ensure_ascii=False, indent=4)

        print(f"Lyrics saved to {file_path}")
    else:
        print(f"No lyrics found for song {song_id}, skipping")


# Function to save lyrics to a JSON file
def save_origin_mp3(
    s3_bucket: str,
    song_id: str,
    original_s3_path: str,
    destination_path: Path
) -> None:
    """
    Save original mp3 file to a local directory.

    Args:
        s3_bucket (str): S3 bucket name.
        song_id (str): Song ID.
        original_s3_path (str): Original S3 path.
        destination_path (Path): Destination path to save the mp3 file.

    Returns:
        None
    
    """
    # If original_s3_path is not empty, copy file from S3 to local directory
    if original_s3_path:
        # Use awscli to copy from S3 to local directory
        command = f"aws s3 cp s3://{s3_bucket}/{original_s3_path} {destination_path}"
        subprocess_status = subprocess.run(command, shell=True , check=True)
        if subprocess_status.returncode == 0:
            print(f"File copied from S3 to {destination_path}")
        else:
            print(f"Error copying file from S3 to {destination_path}")
    else:
        print(f"No original S3 path found for song {song_id}, skipping")


def mv_midi(
    song_id,
    src: Path,
    dst: Path
) -> None:
    """
    Move midi file from source directory to destination directory.

    Args:
        song_id (str): Song ID.
        src (Path): Source directory.
        dst (Path): Destination directory.

    Returns:
        None

    """
    if not src.exists():
        print(f"Source directory {src} does not exist, skipping")
        return None
    src_midi = src / f"{song_id}_src.mp3_5b.mid"
    if src_midi.exists():
        subprocess_status = subprocess.run(f"cp {src_midi} {dst}", shell=True, check=True)
        if subprocess_status.returncode == 0:
            print(f"File copied from {src_midi} to {dst}")
        else:
            print(f"Error copying file from {src_midi} to {dst}")
    else:
        print(f"No midi file found for song {song_id}, skipping")
