import numpy as np
import onnxruntime as ort
import tqdm

n_tokens = 10
n_frames = 100
n_runs = 20
speedup = 20
provider = 'DmlExecutionProvider'

tokens = np.array([[1] * n_tokens], dtype=np.int64)
durations = np.array([[n_frames // n_tokens] * n_tokens], dtype=np.int64)
f0 = np.array([[440.] * n_frames], dtype=np.float32)
speedup = np.array(speedup, dtype=np.int64)

session = ort.InferenceSession('model1.onnx', providers=[provider])
for _ in tqdm.tqdm(range(n_runs)):
    session.run(['mel'], {
        'tokens': tokens,
        'durations': durations,
        'f0': f0,
        'speedup': speedup
    })

session = ort.InferenceSession('model2.onnx', providers=[provider])
for _ in tqdm.tqdm(range(n_runs)):
    session.run(['mel'], {
        'tokens': tokens,
        'durations': durations,
        'f0': f0,
        'speedup': speedup
    })
