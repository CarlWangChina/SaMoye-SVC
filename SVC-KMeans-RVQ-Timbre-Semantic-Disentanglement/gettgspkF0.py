import numpy as np
import logging
import sys
from tqdm import tqdm
from pathlib import Path
from mq_service.pitch_infer import pitch_infer

from ipdb import set_trace


def main():
    """ Read a directory contains a lot of different speakers.
    """

    # saveDir = Path("data_svc/singerF0")
    readDir = Path("data_svc/pitch")
    # saveDir.mkdir(exist_ok=True, parents=True)

    # for speaker in tqdm(readDir.iterdir(), total=len(list(readDir.iterdir()))):
    for speaker in readDir.iterdir():
        spkName = speaker.stem

        spkF0s = []
        spkF0smin = []
        spkF0smax = []
        for fileName in speaker.rglob("*.pit.npy"):
            f0data = np.load(fileName)
            f0data = f0data[f0data > 0.]
            spkF0s.append(f0data.mean())
            spkF0smin.append(f0data.min())
            spkF0smax.append(f0data.max())

        np.save(f"{speaker}/{(spkName+'.npy')}",
                [np.min(spkF0smin), np.mean(spkF0s), np.max(spkF0smax)])
        # print(
        #     f"spk: {spkName.ljust(25,' ')},\tpitch min: {np.min(spkF0smin):.2f}, mean: {np.mean(spkF0s):.2f}, max: {np.max(spkF0smax):.2f}, gap={np.max(spkF0smax)-np.mean(spkF0smin):.2f}, std={np.std(spkF0s):.2f}")
        print(
            f"{spkName},{np.min(spkF0smin):.2f},{np.mean(spkF0s):.2f},{np.max(spkF0smax):.2f},{np.max(spkF0smax)-np.mean(spkF0s):.2f},{np.std(spkF0s):.2f}")


def main3():
    """ Read a speaker specific directory.
    """

    # Read target dir from command line
    if len(sys.argv) != 2:
        print("Please specify the target directory.")
        exit(1)

    readDir = Path(sys.argv[1])
    # saveDir = Path("data_svc/singerF0")
    # readDir = Path("data_svc/pitch/using1_10s")
    # saveDir.mkdir(exist_ok=True, parents=True)

    spkName = readDir.stem

    spkF0s = []
    spkF0smin = []
    spkF0smax = []
    for fileName in readDir.rglob("*.pit.npy"):
        f0data = np.load(fileName)
        f0data = f0data[f0data > 0.]
        spkF0s.append(f0data.mean())
        spkF0smin.append(f0data.min())
        spkF0smax.append(f0data.max())

    set_trace()
    np.save(f"{readDir}/{(spkName+'.npy')}",
            [np.min(spkF0smin), np.mean(spkF0s), np.max(spkF0smax)])
    print(
        f"{spkName},{np.min(spkF0smin):.2f},{np.mean(spkF0s):.2f},{np.max(spkF0smax):.2f},{np.max(spkF0smax)-np.mean(spkF0s):.2f},{np.std(spkF0s):.2f}")


def main2():
    readDir = Path("test_data")

    for fileName in readDir.rglob("*.wav"):
        f0 = pitch_infer(fileName).numpy()
        logging.info(
            f"spk: {fileName},\tpitch min: {np.min(f0):.2f}, mean: {np.mean(f0):.2f}, max: {np.max(f0):.2f}, gap={np.max(f0)-np.mean(f0):.2f}, std={np.std(f0):.2f}")


if __name__ == "__main__":
    main3()
