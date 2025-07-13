import numpy as np
import onnxruntime as ort
import tqdm

n_frames = 1000
n_runs = 20
mel = np.random.randn(1, n_frames, 128).astype(np.float32)
f0 = np.random.randn(1, n_frames).astype(np.float32) + 440.
provider = 'DmlExecutionProvider'

session = ort.InferenceSession('nsf_hifigan.onnx', providers=[provider])
for _ in tqdm.tqdm(range(n_runs)):
    session.run(['waveform'], {
        'mel': mel,
        'f0': f0
    })
