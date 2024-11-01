# SaMoye-SVC

Even dogs can sing a song.

The code is as shown in the repository, and the weight checkpoint can be downloaded from the following Baidu Netdisk link: https://pan.baidu.com/s/1AxnLlmCSPaMAkEBwUyaI2g?pwd=9999

In the zero-shot scenario of the SaMoye model, we've used 'cat' and 'dog' as reference audios, as well as 'man' and 'woman'.
The results generated have all appeared on the homepage.

https://github.com/CarlWangChina/SaMoye-SVC/blob/main/unseen_man0_song2.wav

https://github.com/CarlWangChina/SaMoye-SVC/blob/main/unseen_woman0_song4.wav

https://github.com/CarlWangChina/SaMoye-SVC/blob/main/dog2_song2.wav

https://github.com/CarlWangChina/SaMoye-SVC/blob/main/cat3_song0.wav

https://github.com/CarlWangChina/SaMoye-SVC/blob/main/cat2_song4.wav

Each audio file starts with the target timbre, followed by the generated portion. Please do not skip to the middle to listen to the audio; you will miss the beginning reference audio.

In the root directory, we have included some audio samples as demos. These samples are outputs from the samoye-svc model, which uses the vocal timbres of cats and dogs to simulate singing. 

The current results are trained on a dataset of 1700 hours of clean, unprocessed vocal singing data. Feel free to listen and see if you get the sense that these animals have come to life with the ability to sing like humans. For those seeking improved outcomes, you are encouraged to contribute additional clean vocal singing recordings for continued training and scaling up of the model.

https://github.com/CarlWangChina/SaMoye-SVC/blob/main/test.ipynb

The purpose of test.ipynb is to serve as a script for organizing and processing test data. It is used to handle the test data by generating the necessary files or directories for the inference process. In this context, test.ipynb is a Jupyter Notebook that likely contains instructions or code for preparing test data in the correct format, such as converting or loading data into the appropriate structure that the svc_inference.py script expects.

https://github.com/CarlWangChina/SaMoye-SVC/blob/main/SaMoye-SVC-问题回答整理.docx

https://github.com/CarlWangChina/SaMoye-SVC/blob/main/SaMoye-SVC-Question%26Answer.docx

Regarding problems encountered while running the model code, they are documented in the Q&A document available on our homepage. 
We've prepared two versions: one in Chinese and another in English. If you encounter any bugs in your code, please first consult these documents. 
If these sources do not resolve your issue, feel free to contact us.


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
