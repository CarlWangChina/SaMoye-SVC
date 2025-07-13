import os
import random
import argparse


def print_error(info):
    print(f"\033[31m File isn't existed: {info}\033[0m")


IndexBySinger = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="src_path", required=True)

    args = parser.parse_args()
    os.makedirs("./files/", exist_ok=True)

    src_path = args.src
    rootPath = f"./{src_path}/waves-32k/"
    all_items = []
    for spks in os.listdir(f"./{rootPath}"):
        if not os.path.isdir(f"./{rootPath}/{spks}"):
            continue
        print(f"./{rootPath}/{spks}")
        for file in os.listdir(f"./{rootPath}/{spks}"):
            if file.endswith(".wav"):
                file = file[:-4]

                if IndexBySinger == False:
                    path_spk = f"./{src_path}/speaker/{spks}/{file}.spk.npy"
                else:
                    path_spk = f"./{src_path}/singer/{spks}.spk.npy"

                path_wave = f"./{src_path}/waves-32k/{spks}/{file}.wav"
                path_spec = f"./{src_path}/specs/{spks}/{file}.pt"
                path_pitch = f"./{src_path}/pitch/{spks}/{file}.pit.npy"
                path_hubert = f"./{src_path}/hubert/{spks}/{file}.vec.npy"
                path_whisper = f"./{src_path}/whisper/{spks}/{file}.ppg.npy"
                has_error = 0
                if not os.path.isfile(path_spk):
                    print_error(path_spk)
                    has_error = 1
                if not os.path.isfile(path_wave):
                    print_error(path_wave)
                    has_error = 1
                if not os.path.isfile(path_spec):
                    print_error(path_spec)
                    has_error = 1
                if not os.path.isfile(path_pitch):
                    print_error(path_pitch)
                    has_error = 1
                if not os.path.isfile(path_hubert):
                    print_error(path_hubert)
                    has_error = 1
                if not os.path.isfile(path_whisper):
                    print_error(path_whisper)
                    has_error = 1
                if has_error == 0:
                    all_items.append(
                        f"{path_wave}|{path_spec}|{path_pitch}|{path_hubert}|{path_whisper}|{path_spk}"
                    )
        if "finetune" in src_path:
            fw = open(f"./files/train_{spks}.txt", "w", encoding="utf-8")
            all_items.sort()
            for strs in all_items:
                print(strs, file=fw)
            fw.close()
            all_items = []

    random.shuffle(all_items)
    if "finetune" not in src_path:
        valids = all_items[:10]
        valids.sort()
        trains = all_items[10:]
        # trains.sort()
        fw = open(f"./files/valid.txt", "w", encoding="utf-8")
        for strs in valids:
            print(strs, file=fw)
        fw.close()
        fw = open(f"./files/train.txt", "w", encoding="utf-8")
        for strs in trains:
            print(strs, file=fw)
        fw.close()
