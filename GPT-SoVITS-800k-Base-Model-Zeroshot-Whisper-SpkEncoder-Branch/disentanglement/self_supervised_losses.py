import torch
import torch.nn as nn
import numpy as np
import torchaudio
import librosa



'''
Data augmentation for audio data
'''
def audio_augument_pitch(waveform):
    # waveform: (B, T)
    B, T = waveform.shape
    # pitch shift
    shift = np.random.randint(-12, 12)
    aug_waveform = librosa.effects.pitch_shift(y=waveform.numpy(), sr=16000, n_steps=shift)
    waveform = torch.tensor(aug_waveform).float().to(waveform.device)
    return waveform

def audio_augument_speed(waveform, sr, factors=[0.8, 0.9, 1.1, 1.2, 1.3]):
    transform = torchaudio.transforms.SpeedPerturbation(sr, factors)
    waveform, _ = transform(waveform)
    return waveform

def audio_augument(waveform):
    waveform_aug = waveform.clone()
    methods = [audio_augument_pitch, audio_augument_speed]
    for method in methods:
        if np.random.rand() > 0.5:
            waveform_aug = method(waveform_aug)
    return waveform_aug


'''
Contractive losses
'''
def cosine_loss(embed_x, embed_y):
    '''
    embed_x/y: (B, D)
    '''
    embed_x = F.normalize(embed_x, p=2, dim=1)
    embed_y = F.normalize(embed_y, p=2, dim=1)
    sim_xy = torch.matmul(embed_x, embed_y.T)
    loss = 1 - sim_xy
    return loss.mean()
    

def batch_nce_loss(embed_x, embed_y, aff_xy, tau=0.07, noramalize=False, num_pos_pair=1):
    '''
    embed_x/y: (B, D), (B, D)
    '''
    if normalize:
        embed_x = F.normalize(embed_x, p=2, dim=1)
        embed_y = F.normalize(embed_y, p=2, dim=1)
        
    sim_xy = torch.matmul(embed_x, embed_y.T)
    batch_size = aff_xy.shape[0]
    num_neg_pair = batch_size - num_pos_pair
    # 根据aff_xy关系选择正负样本对
    pos_pair_ndx = torch.where(aff_xy)
    neg_pair_ndx = torch.where(torch.logical_not(aff_xy))
    # 正负样本距离
    sim_pos_pair = sim_xy[pos_pair_ndx].reshape(batch_size, num_pos_pair)
    sim_neg_pair = sim_xy[neg_pair_ndx].reshape(batch_size, num_neg_pair)
    # 计算nce损失
    sim_pos_pair = torch.exp(sim_pos_pair / tau)
    sim_neg_pair = torch.exp(sim_neg_pair / tau)
    
    sim_pos_pair = sim_pos_pair.sum(dim=-1, keepdim=True)
    sim_neg_pair = sim_neg_pair.sum(dim=-1, keepdim=True)
    loss = torch.log(sim_pos_pair + sim_neg_pair) -  torch.log(sim_pos_pair)
    loss = loss.mean()
    return loss


if __name__ == '__main__':
    
    audio_path = 'example_refence_audio/qinghuaci.wav'
    test_audio, sr = torchaudio.load(audio_path)
    test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)

    audio_pitch_aug = audio_augument_pitch(test_audio)
    audio_speed_aug = audio_augument_speed(test_audio, 16000)
    audio_aug = audio_augument(test_audio)
    print('Done!')
    
    # save audio
    torchaudio.save('qinghuaci_pitch.wav', audio_pitch_aug, 16000)
    torchaudio.save('qinghuaci_speed.wav', audio_speed_aug, 16000)
    torchaudio.save('qinghuaci_aug.wav', audio_aug, 16000)