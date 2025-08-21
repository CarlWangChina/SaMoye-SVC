# SaMoye-SVC

## Quick Start: Simple Voice Cloning

This guide covers the fastest way to run the SaMoye SVC model for voice cloning. For this simple inference task, you only need the code located in the `SaMoye-SVC/SaMoye-Model/` directory.

### Step 1: Download the Pre-trained Model

First, download the required model weights from the link below and place the `.pt` file inside the `SaMoye-SVC/SaMoye-Model/` directory.

-   **Download Link:** [https://huggingface.co/karl-wang/SaMoyeSVC/blob/main/sovits_spk_1700h_0020.pt](https://huggingface.co/karl-wang/SaMoyeSVC/blob/main/sovits_spk_1700h_0020.pt)

### Step 2: Run the Inference Command

Navigate into the `SaMoye-SVC/SaMoye-Model/` directory and execute the following command in your terminal:

```bash
python svc_inference.py --config configs/sovits_spk_1700h.yaml --model sovits_spk_1700h_0020.pt --spk path/to/your/speaker.wav --wave path/to/your/content.wav
```

### Important:

You must customize the paths for --spk and --wave.

--spk: This is the path to the audio file of the target voice you want to clone.

--wave: This is the path to the content audio you want to convert.

That's it! It's that simple. You can safely ignore all other folders, as they contain supplementary research materials not required to run the model.


## Even dogs can sing a song.

The code is as shown in the repository, and the weight checkpoint can be downloaded from the following:
I have uploaded the model weights here on Hugging Face: https://huggingface.co/karl-wang/SaMoyeSVC/tree/main
Baidu Netdisk link: https://pan.baidu.com/s/1AxnLlmCSPaMAkEBwUyaI2g?pwd=9999

In the zero-shot scenario of the SaMoye model, we've used 'cat' and 'dog' as reference audios, as well as 'man' and 'woman'.
The results generated have all appeared on the homepage.

https://github.com/CarlWangChina/SaMoye-SVC/PetVocolia-Demo
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
Download the ckpt from [this link](https://huggingface.co/karl-wang/SaMoyeSVC/tree/main).

```bash
python svc_inference.py --config configs/sovits_spk_1700h.yaml --model sovits_spk_1700h_0020.pt --spk spk.wav --wave content.wav
```
# Model Checkpoints Migration Notice

To comply with GitHub's file size limitations and optimize repository structure, all large model checkpoint files (e.g., `.pth`, `.pt`, `.npy`) have been migrated to **Hugging Face Hub** for centralized storage and management.


### Migration Reason
- GitHub restricts individual file sizes to ≤100MB and large files (>2GB) require special handling.
- Hugging Face provides dedicated Large File Storage (LFS) support for model checkpoints.


### Hugging Face Repository
All model files are hosted in the following repository:  
[**https://huggingface.co/karl-wang/SaMoyeSVC/tree/main**](https://huggingface.co/karl-wang/SaMoyeSVC/tree/main)  
Specific checkpoint directory:  
`https://huggingface.co/karl-wang/SaMoyeSVC/tree/main/checkpoints-for-samoye-experiments`


### Usage Guide
1. **Access the Repository**:  
   Visit the Hugging Face link above and navigate to `checkpoints-for-samoye-experiments`.

2. **Download Checkpoints**:  
   Select the required model files (see list below) and download them locally.

3. **File Placement**:  
   Place the downloaded files into the project's `experiments/` directory to ensure proper code execution.


### Migrated File List
The following files have been moved from this repository to Hugging Face:  **[https://huggingface.co/karl-wang/SaMoyeSVC/tree/main](https://huggingface.co/karl-wang/SaMoyeSVC/tree/main)**
- `3025_nanzhong_00057_005.ppg.npy`  
- `best_model.pth.tar`  
- `hubert-soft-0d54a1f4.pt`  
- `kmeans_10000.pt`  
- `large-v2.pt`  
- `mix2spk92_sing_100k1.pth`  
- `mix2spk92_sing_121k1.pth`  
- `mix2spk92_sing_13k8.pth`  
- `mix2spk92_sing_95k2.pth`  
- `pretrained_model_50.pth`  
- `selfrecord50_sing_100k.pth`  
- `selfrecord50_sing_105k2.pth`  
- `selfrecord50_sing_13k8.pth`  
- `selfrecord50_sing_90k.pth`  
- `sovits5.0_pretrain.pth`  
- `sovitsfrompretrainedOrigin170_100.pth`  
- `toymodel0621spk42_sing_120k.pth`  
- `toymodel0621spk42_sing_130k.pth`  
- `toymodel0621spk42_sing_138k.pth`  
- `trainKMeans10k_10.pth`  
- `trainKMeans10k_25.pth`  
- `trainKMeans10k_50.pth`  
- `trainKMeans10kNOPPG_726k.pth`  
- `trainKMeans900_10.pth`  
- `trainKMeans900_25.pth`  
- `trainKMeans900_50.pth`  
- `trainRVQIDXNOPPG_100epoch.pth`  
- `trainRVQIDXNOPPG_50epoch.pth`  
- `trainRVQNOPPG_71k.pth`  
- `trainRVQNOPPG_91k.pth`  
- `trainTNohubertsoft_50.pth`

# Large File and Dataset Migration Notice

To keep this GitHub repository lean and efficient, all large files, datasets, experimental configurations, and model weights have been migrated to Hugging Face Hub.

All required assets can be downloaded from our official Hugging Face repository:
**[https://huggingface.co/datasets/karl-wang/SaMoyeSVC/tree/main](https://huggingface.co/datasets/karl-wang/SaMoyeSVC/tree/main)**

### Migrated Content Overview

The migrated content includes the following:

1.  **Archived Experiment Directories:**
    The following complete directories have been archived. Please download the corresponding `.zip` files and extract them to their original location.
    - `experiments/configs`
    - `experiments/files`
    - `experiments/train_shpanxin`
    - `experiments/yongshengTestData`
    - `experiments/ExperimentResult_20240804_201310`

2.  **Specific Archived Packages from `SaMoye-2`:**
    These specific large packages have been archived. Please download and extract them to the original paths listed below.
    - `raw_data.zip` (original path: `Singing-Data-Auto-Labeling-and-Diffsinger/raw_data`)
    - `models.zip` (original path: `Whisper-SoVITS-Finetune_Contra-Branch-Contrastive-Learning_Rebuild-Branch-Refined/models`)
    - `singer.zip` (original path: `Legacy-Diffsinger-Data-Preprocessing-Training-Code/singer`)

3.  **All Other Large Files from `SaMoye-2`:**
    Additionally, all individual files larger than **5 MB** from the `SaMoye-2` directory have been moved. This includes audio samples (`.wav`, `.mp3`, `.aac`), model weights (`.jit`, `.npy`), dictionaries (`.txt`), and other auxiliary data.

> **Important:**
> These files are essential for running experiments, training models, or reproducing results. They are not included in this GitHub repository, so please ensure you download and restore them to their correct original paths to ensure full functionality.
