import numpy as np
import matplotlib.pyplot as plt
import librosa
import sys
import os
from pathlib import Path

from ipdb import set_trace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mq_service.pitch_infer import pitch_infer


def main():
    # f0Read = Path("infer_test20240612_225733/sing/henan_read_20s_15_w_-12.npy")
    # wavRead = f0Read.with_suffix(".wav")

    f0_predictor = "crepe"
    wavReadDir = Path(f"test_data/vocal_{f0_predictor}")

    for wavRead in wavReadDir.glob("*.wav"):
        f0 = pitch_infer(wavRead, f0_predictor=f0_predictor)
        f0 = f0.numpy() if f0_predictor == "crepe" else f0

        wavData, sr = librosa.load(wavRead, sr=32000)

        D = librosa.amplitude_to_db(np.abs(librosa.stft(wavData)), ref=np.max)
        # sr from target wav not svc result wav
        times = librosa.times_like(f0, sr=36000)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(D, x_axis="time", y_axis="log", ax=ax)
        ax.set(
            title=f"F0 crepe fundamental frequency estimation {wavRead.stem}")
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.plot(times, f0, label="F0", color="cyan", linewidth=3)
        ax.legend(loc="upper right")
        # save to file
        plt.savefig(wavRead.with_suffix(".png"))
        np.save(wavRead.with_suffix(".npy"), f0)
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
