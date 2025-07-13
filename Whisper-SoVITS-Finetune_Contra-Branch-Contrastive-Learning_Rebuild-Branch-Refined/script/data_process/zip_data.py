import os
import zipfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", default="Acapella", type=str)
args = ap.parse_args()
directory = args.directory
base_dir = Path("/home/john/svc/data/origin_data")
origin_dir = base_dir / directory
zip_file = base_dir / f"{directory}.zip"


def get_file_size(file_path):
    return file_path.stat().st_size


def zip_file_process(file_path):
    # Returning the relative path and file path as tuple
    return file_path.relative_to(origin_dir), file_path


def compress_files(file_list):
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as z:
        for relative_path, file_path in tqdm(
            file_list, desc="Compressing", unit="files"
        ):
            z.write(file_path, relative_path)


def main():

    # Get all files in the Acapella directory
    file_paths = [p for p in origin_dir.rglob("*") if p.is_file()]

    # Get the original size
    original_size = sum(get_file_size(file_path) for file_path in file_paths)

    # Use multiprocessing to prepare the file paths for compression
    with Pool(32) as p:
        file_list = list(
            tqdm(
                p.imap(zip_file_process, file_paths),
                total=len(file_paths),
                desc="Preparing",
                unit="files",
            )
        )

    # Compress the files
    compress_files(file_list)

    # Get the compressed file size
    compressed_size = zip_file.stat().st_size

    # Output the sizes
    print(f"Original size: {original_size / (1024 * 1024):.2f} MB")
    print(f"Compressed size: {compressed_size / (1024 * 1024):.2f} MB")


main()
