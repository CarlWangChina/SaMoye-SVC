import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
from mq_service.pitch_infer import pitch_infer

from ipdb import set_trace


def main2():
    readDir = Path("data/beijing_male")

    with open("data/beijing_male.txt", "wt", encoding="utf8") as fw:
        for fileName in readDir.rglob("*.wav"):
            f0 = pitch_infer(fileName).numpy()
            logging.info(
                f"spk: {fileName},\tpitch min: {np.min(f0):.2f}, mean: {np.mean(f0):.2f}, max: {np.max(f0):.2f}, gap={np.max(f0)-np.mean(f0):.2f}, std={np.std(f0):.2f}")
            fw.write(
                f"{str(fileName)},{np.min(f0):.2f}, {np.mean(f0):.2f}, {np.max(f0):.2f}, {np.max(f0)-np.mean(f0):.2f}, {np.std(f0):.2f}\n")


def main():
    readName = Path(
        "infer_test20240612_234344/sing/henan_read_20s_15_m_-12.wav")
    f0 = pitch_infer(readName).numpy()

    saveName = readName.with_suffix(".npysvc")
    np.save(saveName, f0)


if __name__ == "__main__":
    # main2()
    main()
