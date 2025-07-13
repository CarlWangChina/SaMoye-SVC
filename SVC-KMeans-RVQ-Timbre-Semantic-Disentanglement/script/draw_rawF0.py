import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mq_service.pitch_infer import pitch_infer

from ipdb import set_trace


def main():
    f0Up = Path("infer_test20240613_104457/sing/suno_test_w_30s_0.npy")
    f0Down = Path("infer_test20240613_104457/sing/suno_test_w_30s_-12.npy")
    svcResult = Path(
        "infer_test20240613_104457/sing/henan_read_20s_15_w_-12.wav")

    f0up = np.load(f0Up)
    f0down = np.load(f0Down)

    # Infer pitch
    svcF0 = pitch_infer(svcResult).numpy()

    # 用matplot 在一个画布上画两个图
    times = np.arange(f0up.shape[0]) / 1000.0  # 单位是秒

    # 创建一个画布和两个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    # 在第一个子图上绘制y1
    ax1.plot(times, f0up, label="f0up")
    ax1.set_title("-12 F0")

    # 在第二个子图上绘制y2
    ax2.plot(times, f0down, label="f0down")
    ax2.set_title("no shift F0")

    ax3.plot(times, svcF0, label="svcF0")
    ax3.set_title("svc F0")

    # 显示图形
    plt.savefig("compare_F0.png")
    # plt.show()
    plt.close()


def main2():
    """ Draw f0 from one dir
    """

    # f0Dir = Path("test_data/vocal_rmvpe")
    f0Dir = Path(sys.argv[1])
    for npyFile in f0Dir.glob("*.npy"):
        f0 = np.load(npyFile)
        times = np.arange(f0.shape[0]) / 1000.0
        fig, ax = plt.subplots(1, 1)
        ax.plot(times, f0, label=f"F0 of {npyFile.stem}")
        plt.savefig(npyFile.with_suffix(".f0.png"))
        plt.close()


if __name__ == "__main__":
    main2()
