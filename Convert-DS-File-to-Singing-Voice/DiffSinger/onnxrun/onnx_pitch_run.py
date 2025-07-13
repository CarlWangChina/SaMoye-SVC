import onnxruntime
import numpy as np
import torch
from utils.hparams import set_hparams, hparams
import logging 

onnxruntime.set_default_logger_severity(3)

class DsPitch():
    def __init__(self):
        # Load phonemes list
        # phonemesPath = 'onnxrun/onnx_model/zhibin_pitch/0913_zhibin_melodyencoder128x4_nobasepitch.phonemes.txt'
        # with open(phonemesPath, 'r') as f:
        #     self.phonemes = f.read().splitlines()

        # 加载模型
        # linguisticModel_path = 'onnxrun/onnx_model/zhibin_pitch/0913_zhibin_melodyencoder128x4_nobasepitch.linguistic.onnx'
        linguisticModel_path = 'onnxrun/onnx_model/zhibin_pitch/batch_size_none.linguistic.onnx'
        self.linguisticModel = onnxruntime.InferenceSession(linguisticModel_path, providers=['CUDAExecutionProvider'])
        # pitchModel_path = 'onnxrun/onnx_model/zhibin_pitch/0913_zhibin_melodyencoder128x4_nobasepitch.pitch.onnx'
        pitchModel_path = 'onnxrun/onnx_model/zhibin_pitch/batch_size_none.pitch.onnx'
        self.pitchModel = onnxruntime.InferenceSession(pitchModel_path, providers=['CUDAExecutionProvider'])
        print("| ONNX",onnxruntime.get_device(),self.linguisticModel.get_providers(), self.pitchModel.get_providers())
        dsConfig_hop_size = hparams['hop_size']
        dsConfig_sample_rate = hparams['audio_sample_rate']
        self.frameMs = 1000 * dsConfig_hop_size / dsConfig_sample_rate
        self.batch_size = hparams['batch_size']

    def Process(self, sample):
        # print(onnxruntime.get_device())
        # 执行推理
        # 1.linguistic Model推理
        # 获取模型的所有输入和输出节点名称
        # 获取模型的输入信息
        # input_details = self.linguisticModel.get_inputs()
        # for input_detail in input_details:
        #     print(f"Input Name: {input_detail.name}, Input Shape: {input_detail.shape}")
        # Input Name: tokens, Input Shape: [1, 'n_tokens']
        # Input Name: ph_dur, Input Shape: [1, 'n_tokens']
        # 获取模型的输出信息
        # output_details = self.linguisticModel.get_outputs()
        # for output_detail in output_details:
        #     print(f"Output Name: {output_detail.name}, Output Shape: {output_detail.shape}")
        # Output Name: encoder_out, Output Shape: [1, 'n_tokens', 256]
        # Output Name: x_masks, Output Shape: [1, 'n_tokens']

        max_tokens_length = max(entry['tokens'].shape[1] for entry in sample)  # 计算tokens的最大长度
        max_ph_dur_length = max(entry['ph_dur'].shape[1] for entry in sample)  # 计算ph_dur的最大长度
        # 创建空的 NumPy 数组，用于存放整合后的数据
        tokens_array = np.zeros((len(sample), max_tokens_length), dtype=np.int64)
        ph_dur_array = np.zeros((len(sample), max_ph_dur_length), dtype=np.int64)
        # print(f"tokens: {tokens_array.shape}")
        # 将数据填充到 NumPy 数组中
        for i, entry in enumerate(sample):
            tokens_array[i, :entry['tokens'].shape[1]] = entry['tokens'].cpu().numpy()
            ph_dur_array[i, :entry['ph_dur'].shape[1]] = entry['ph_dur'].cpu().numpy()
        
        linguisticInputs = {
            'tokens': tokens_array,
            'ph_dur': ph_dur_array
        }
        # for name, tensor in linguisticInputs.items():
        #     print(f"Shape of {name}: {tensor.shape}")
        linguisticOutputs = self.linguisticModel.run(None, linguisticInputs)

        # 2.pitch 模型推理
        # 获取模型的所有输入和输出节点名称
        # input_details = self.pitchModel.get_inputs()
        # for input_detail in input_details:
        #     print(f"Input Name: {input_detail.name}, Input Shape: {input_detail.shape}")
        # Input Name: encoder_out, Input Shape: [1, 'n_tokens', 256]
        # Input Name: ph_dur, Input Shape: [1, 'n_tokens']
        # Input Name: note_midi, Input Shape: [1, 'n_notes']
        # Input Name: note_rest, Input Shape: [1, 'n_notes']
        # Input Name: note_dur, Input Shape: [1, 'n_notes']
        # Input Name: pitch, Input Shape: [1, 'n_frames']
        # Input Name: retake, Input Shape: [1, 'n_frames']
        # Input Name: speedup, Input Shape: []
        # 获取模型的输出信息
        # output_details = self.pitchModel.get_outputs()
        # for output_detail in output_details:
        #     print(f"Output Name: {output_detail.name}, Output Shape: {output_detail.shape}")
        # Output Name: pitch_pred, Output Shape: [1, 'n_frames']
        
        encoder_out = linguisticOutputs[0]
        max_ph_dur_length = max(entry['ph_dur'].shape[1] for entry in sample)
        max_note_midi_length = max(entry['note_midi'].shape[1] for entry in sample)
        max_note_rest_length = max(entry['note_rest'].shape[1] for entry in sample)
        max_note_dur_length = max(entry['note_dur'].shape[1] for entry in sample)
        max_base_pitch_length = max(entry['base_pitch'].shape[1] for entry in sample)

        ph_dur_array = np.zeros((len(sample), max_ph_dur_length), dtype=np.int64)
        note_midi_array = np.zeros((len(sample), max_note_midi_length), dtype=np.float32)
        note_rest_array = np.zeros((len(sample), max_note_rest_length), dtype=np.bool_)
        note_dur_array = np.zeros((len(sample), max_note_dur_length), dtype=np.int64)
        base_pitch_array = np.zeros((len(sample), max_base_pitch_length), dtype=np.float32)

        for i, entry in enumerate(sample):
            ph_dur_array[i, :entry['ph_dur'].shape[1]] = entry['ph_dur'].cpu().numpy()
            note_midi_array[i, :entry['note_midi'].shape[1]] = entry['note_midi'].cpu().numpy()
            note_rest_array[i, :entry['note_rest'].shape[1]] = entry['note_rest'].cpu().numpy()
            note_dur_array[i, :entry['note_dur'].shape[1]] = entry['note_dur'].cpu().numpy()
            base_pitch_array[i, :entry['base_pitch'].shape[1]] = entry['base_pitch'].cpu().numpy()

        # retake 的形状要和pitch相同，内容都是True
        retake = np.full_like(base_pitch_array, True).astype(bool)
        speedup = [50]

        pitchInputs = {
            'encoder_out': encoder_out,
            'ph_dur': ph_dur_array,
            'note_midi': note_midi_array,
            'note_rest': note_rest_array,
            'note_dur': note_dur_array,
            'pitch': base_pitch_array,
            'retake': retake,
            'speedup': speedup
        }
        # for name, tensor in pitchInputs.items():
        #     try:
        #         print(f"Shape of {name}: {tensor.shape}, content: {tensor}")
        #     except:
        #         print(f"Shape of {name}: {tensor}")
        output = self.pitchModel.run(['pitch_pred'], pitchInputs)
        # np.set_printoptions(precision=2, suppress=True)

        # for out in output:
        #     print(f"Shape of output: {out.shape}, content: {out}")
        #     print(f"Shape of pitch: {pitch.shape}, content: {pitch}")
        return output


def main():
    dsPitch = DsPitch()
    dsPitch.Process()

if __name__ == '__main__':
    main()

