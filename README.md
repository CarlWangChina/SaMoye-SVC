# SaMoye-SVC

Even dogs can sing a song.

In the root directory, we have included some audio samples as demos. These samples are outputs from the samoye-svc model, which uses the vocal timbres of cats and dogs to simulate singing. The current results are trained on a dataset of 1700 hours of clean, unprocessed vocal singing data. Feel free to listen and see if you get the sense that these animals have come to life with the ability to sing like humans. For those seeking improved outcomes, you are encouraged to contribute additional clean vocal singing recordings for continued training and scaling up of the model.

![image](pics/cover.jpg)


# Script
This model's code is modified and improved based on [whisper-vits-svc](https://github.com/PlayVoice/whisper-vits-svc).

```bash
cd model
```

## Training:
```bash
python svc_trainer.py -c configs/sovits_spk_1700h.yaml -n sovits_spk_1700h
```

## Tensorboard:
```bash
tensorboard --logdir=logs/sovits_spk_1700h --port 12345
```

## Infer script:
Download the ckpt from [this link](https://pan.baidu.com/s/1AxnLlmCSPaMAkEBwUyaI2g?pwd=9999).

```bash
python svc_inference.py --config configs/sovits_spk_1700h.yaml --model sovits_spk_1700h_0020.pt --spk spk.wav --wave content.wav
```
