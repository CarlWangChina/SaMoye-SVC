import onnxruntime
import numpy as np
import sys
sys.path.append('.')  # 添加上一级文件夹到路径中
from utils.hparams import set_hparams, hparams
from modules.fastspeech.tts_modules import (
    LengthRegulator, RhythmRegulator,
    mel2ph_to_dur
)

def prepare_data(sample, key=''):
    # 计算tokens和ph_dur的最大长度
    max_length = max(entry[key].shape[1] for entry in sample)
    
    # 创建空的 NumPy 数组，用于存放整合后的数据
    out_array = np.zeros((len(sample), max_length), dtype=np.int64)
    
    # 将数据填充到 NumPy 数组中
    for i, entry in enumerate(sample):
        out_array[i, :entry[key].shape[1]] = entry[key].cpu().numpy()
    
    return out_array

class DsDuration():
    def __init__(self):
        # 加载模型
        # linguisticModel_path = 'onnxrun/onnx_model/zhibin_ph_dur/0630_zhibin_v7.2_multivar.linguistic.onnx'
        linguisticModel_path = '/home/john/Muer_DS/DiffSinger/onnxrun/onnx_model/zhibin_ph_dur/batch_size_none.linguistic.onnx'
        self.linguisticModel = onnxruntime.InferenceSession(linguisticModel_path)
        # durModel_path = 'onnxrun/onnx_model/zhibin_ph_dur/0630_zhibin_v7.2_multivar.dur.onnx'
        durModel_path = "/home/john/Muer_DS/DiffSinger/onnxrun/onnx_model/zhibin_ph_dur/batch_size_none.dur.onnx"
        self.durModel = onnxruntime.InferenceSession(durModel_path)
        dsConfig_hop_size = hparams['hop_size']
        dsConfig_sample_rate = hparams['audio_sample_rate']
        self.frameMs = 1000 * dsConfig_hop_size / dsConfig_sample_rate
        self.batch_size = 1 #hparams['batch_size']
        self.rr = RhythmRegulator()
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
        # Input Name: word_div, Input Shape: [1, 'n_words']
        # Input Name: word_dur, Input Shape: [1, 'n_words']

        # 获取模型的输出信息
        # output_details = self.linguisticModel.get_outputs()
        # for output_detail in output_details:
        #     print(f"Output Name: {output_detail.name}, Output Shape: {output_detail.shape}")
        # Output Name: encoder_out, Output Shape: [1, 'n_tokens', 256]
        # Output Name: x_masks, Output Shape: [1, 'n_tokens']
        # tokens = sample['tokens'].cpu().numpy()
        
        
        # if self.batch_size > 1:
        #     tokens = np.repeat(tokens, self.batch_size, axis=0)
        #     ph_dur = np.repeat(ph_dur, self.batch_size, axis=0)
            
        token_array = prepare_data(sample,'tokens')
        word_div_array = prepare_data(sample,'ph_num')
        word_dur_array = prepare_data(sample,'word_dur')
        linguisticInputs = {
            'tokens': token_array,
            'word_div':word_div_array,
            'word_dur': word_dur_array
        }
        # for name, tensor in linguisticInputs.items():
        #     print(f"Shape of {name}: {tensor.shape}")
        linguisticOutputs = self.linguisticModel.run(None, linguisticInputs)
        # for out in linguisticOutputs:
        #     print(f"Shape of output: {out.shape}")
        # 2.dur 模型推理
        # 获取模型的所有输入和输出节点名称
        # input_details = self.durModel.get_inputs()
        # for input_detail in input_details:
        #     print(f"Input Name: {input_detail.name}, Input Shape: {input_detail.shape}")
        # Input Name: encoder_out, Input Shape: [1, 'n_tokens', 256]
        # Input Name: x_masks, Input Shape: [1, 'n_tokens']
        # Input Name: ph_midi, Input Shape: [1, 'n_tokens']
        # 获取模型的输出信息
        # output_details = self.durModel.get_outputs()
        # for output_detail in output_details:
        #     print(f"Output Name: {output_detail.name}, Output Shape: {output_detail.shape}")
        # Output Name: ph_dur_pred, Output Shape: [1, 'n_tokens']
        encoder_out = linguisticOutputs[0]
        x_masks = linguisticOutputs[1]
        ph_midi_array = prepare_data(sample,'midi')

        durInputs = {
            'encoder_out': encoder_out,
            'x_masks': x_masks,
            'ph_midi': ph_midi_array
        }
        # for name, tensor in pitchInputs.items():
        #     try:
        #         print(f"Shape of {name}: {tensor.shape}, content: {tensor}")
        #     except:
        #         print(f"Shape of {name}: {tensor}")
        output = self.durModel.run(None, durInputs)
        # np.set_printoptions(precision=2, suppress=True)
        # for out in output:
        #     print(f"Shape of output: {out.shape}, content: {out}")
        #     print(f"Shape of pitch: {pitch.shape}, content: {pitch}")
        # ph2word_array = prepare_data(sample,'ph2word')
        # if output is not None:
        #     output = self.rr(output, ph2word_array, word_dur_array)
        return output


def main():
    dsDur = DsDuration()
    dsDur.Process(sample = '')

if __name__ == '__main__':
    main()

