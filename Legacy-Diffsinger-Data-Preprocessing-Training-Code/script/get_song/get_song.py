from get_song_from_aces import change_new_midi_to_ace_vocal
from get_song_from_ds import change_new_midi_to_ds_vocal

# sudo /home/john/miniconda3/envs/diffsinger/bin/python /home/john/CaichongSinger/script/get_song/get_song.py
# --new_midi_file_path xxxx --vocal_file_path xxxx --method ace --cuda 0
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--new_midi_file_path", type=str, required=True)
    parser.add_argument("--vocal_file_path", type=str, required=True)
    parser.add_argument("--cuda", type=int, default=0, help="cuda id, default 0")
    parser.add_argument("--method", type=str, required=True, help="ace or ds")
    args = parser.parse_args()

    if "ace" in args.method:
        change_new_midi_to_ace_vocal(
            args.new_midi_file_path, args.vocal_file_path, args.cuda
        )
    if "ds" in args.method:
        change_new_midi_to_ds_vocal(
            args.new_midi_file_path, args.vocal_file_path, args.cuda
        )
