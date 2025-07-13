""" Class for svc
    Author: Xin Pan
    Date: 2024.06.19
"""

import torch
import numpy as np
import traceback
import random
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
from scipy.io import wavfile



from vits.models import (SynthesizerInfer)
from utils import (get_logger)
from src.utils import (load_whisper_model, pred_ppg,
                       load_hubert_model, pred_vec, compute_f0_sing, post_process, load_targer_speakers)

from vad.utils import (init_jit_model)
from feature_retrieval import (DummyRetrieval)


# Define logger for service log
logger = get_logger("svc_logger")


class SVC5():
    ppgModel = None
    hubertModel = None
    # vadModel = None

    def __init__(self,
                 modelConfig,
                 modelCkpt: str,
                 ppgModel_path: str = "pretrainModels/whisper_pretrain/large-v2.pt",
                 hubertModel_path: str = "pretrainModels/hubert_pretrain/hubert-soft-0d54a1f4.pt",
                 vadModel_path: str = "vad/assets/silero_vad.jit",
                 svcModel_path: str = "pretrainModels/svcModel",
                 device: str = "cuda"):

        logger.info(
            f"Initializing svc with config: {modelConfig}, model path: {modelCkpt}")
        # Define inference device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # SVC model config and ckpt
        self.svcModelConfig = OmegaConf.load(modelConfig)
        self.svcModel = SynthesizerInfer(self.svcModelConfig.data.filter_length // 2 + 1,
                                         self.svcModelConfig.data.segment_size // self.svcModelConfig.data.hop_length,
                                         self.svcModelConfig)
        self._load_svc_model(modelCkpt)

        if __class__.ppgModel is None:
            __class__.ppgModel = load_whisper_model(
                "pretrainModels/whisper_pretrain/large-v2.pt", self.device)

        if __class__.hubertModel is None:
            __class__.hubertModel = load_hubert_model(
                "pretrainModels/hubert_pretrain/hubert-soft-0d54a1f4.pt", self.device)

        self.speakers = load_targer_speakers(svcModel_path)
        self.speakersLoadBase = svcModel_path
        logger.info(f"Model speakers have {len(self.speakers)}")


        self.speakers = load_targer_speakers("pretrainModels/svcModel")
        logger.info(f"Svc obj init success!")

    def _load_svc_model(self, modelCkpt: str):
        # Load svc model from ckpt
        logger.info(f"Loading svc model from {modelCkpt}")

        ckptDict = torch.load(modelCkpt)["model_g"]
        stateDict = self.svcModel.state_dict()
        newStateDict = {}
        for k, v in stateDict.items():
            try:
                newStateDict[k] = ckptDict[k]
            except:
                logger.error(f"{k} is not in svc ckptDict")
                newStateDict[k] = v

        self.svcModel.load_state_dict(newStateDict)

        # Eval mode and to device
        self.svcModel.eval()
        self.svcModel.to(self.device)
        logger.info(f"Loading svc model success!")

    def whisper_inference(self, wav):
        return pred_ppg(__class__.ppgModel, wav, self.device)

    def hubert_inference(self, wav):
        return pred_vec(__class__.hubertModel, wav, self.device)

    def _spk_auto_select(self, targetPit, f0scale: float = 1.0):
        """ Random select target speaker depend on target pitch and target wav pitch
        Return:
            spkNpy: numpy array of selected speaker
            pit: torch.Tensor of selected speaker
        """

        randomSequence = random.sample(self.speakers, len(self.speakers))
        spkPit = np.load(
            f"pretrainModels/svcModel/pitch/{randomSequence[0]}.npy")
        spkNpy = np.load(
            f"pretrainModels/svcModel/singer/{randomSequence[0]}.spk.npy")

        logger.info(
            f"Default select spk: {randomSequence[0]}")

        for spkName in randomSequence:
            spkPit = np.load(f"pretrainModels/svcModel/pitch/{spkName}.npy")
            spkNpy = np.load(
                f"pretrainModels/svcModel/singer/{spkName}.spk.npy")

            targetWavF0Max = np.max(targetPit.numpy())
            if spkPit[6] > targetWavF0Max:
                logger.info(
                    f"Auto select spk: {spkName}, spk 90% F0={spkPit[6]:0.2f}, target wav F0 max={targetWavF0Max:0.2f}")
                break

        # Auto F0 shift depend on target F0 50percent compare to target wav F0 mean
        if spkPit[4] < np.mean(targetPit.numpy()) * f0scale:
            pit = targetPit.numpy()
            source = pit[pit > 0]
            source_mean = source.mean()
            source_min = source.min()
            source_max = source.max()
            shift = -12
            logger.info(f"Target wav pitch mean={source_mean:0.1f}, \
            min={source_min:0.1f}, max={source_max:0.1f}, f0 shift={shift}")
            targetPit = pit * (2**(shift / 12))

        pit = torch.FloatTensor(targetPit) if isinstance(
            targetPit, np.ndarray) else targetPit
        return spkNpy, pit

    def inference_with_auto_slice(self, tgtWav: str, savePath: str, spkName: str = None, f0scale: float = 1.0):
        bret = False
        logger.info(
            f"Input params: tgtWav={tgtWav}, savePath={savePath}, spkName={spkName}, f0scale={f0scale}")

        savePath = Path(savePath)
        savePath.parent.mkdir(parents=True, exist_ok=True)

        try:
            ppg = self.whisper_inference(tgtWav)
            vec = self.hubert_inference(tgtWav)
            pit = compute_f0_sing(tgtWav, self.device)

            # Auto speaker select depend on target spk F0 and target wav F0
            spkNpy, pit = self._spk_auto_select(pit, f0scale)
            spk = torch.FloatTensor(spkNpy)

            ppg = np.repeat(ppg, 2, 0)
            ppg = torch.FloatTensor(ppg)

            vec = np.repeat(vec, 2, 0)
            vec = torch.FloatTensor(vec)

            retrieval = DummyRetrieval()

            logger.info(f"Will run svc infer {tgtWav}")
            out_audio = self.svc_infer(retrieval, spk, pit, ppg, vec)
            logger.info(f"Finish run svc infer {tgtWav}")

            logger.info(f"Will run post_process {tgtWav}")

            new_wav, _ = post_process(
                __class__.vadModel, refWavPath=tgtWav, svcWav=out_audio)
            logger.info(f"Finish run post_process {tgtWav}")

            wavfile.write(savePath,
                          self.svcModelConfig.data.sampling_rate, new_wav)
            bret = True

        except Exception as e:
            logger.error(traceback.format_exc())
        finally:
            return bret

    def svc_infer(self, retrieval, spk, pit, ppg, vec):
        len_pit = pit.size()[0]
        len_vec = vec.size()[0]
        len_ppg = ppg.size()[0]
        len_min = min(len_pit, len_vec)
        len_min = min(len_min, len_ppg)
        pit = pit[:len_min]
        vec = vec[:len_min, :]
        ppg = ppg[:len_min, :]

        with torch.no_grad():
            spk = spk.unsqueeze(0).to(self.device)
            source = pit.unsqueeze(0).to(self.device)
            source = self.svcModel.pitch2source(source)

            hop_size = self.svcModelConfig.data.hop_length
            all_frame = len_min
            hop_frame = 10
            out_chunk = 2500  # 25 S
            out_index = 0
            out_audio = []

            while out_index < all_frame:
                if out_index == 0:  # start frame
                    cut_s = 0
                    cut_s_out = 0
                else:
                    cut_s = out_index - hop_frame
                    cut_s_out = hop_frame * hop_size

                if out_index + out_chunk + hop_frame > all_frame:  # end frame
                    cut_e = all_frame
                    cut_e_out = -1
                else:
                    cut_e = out_index + out_chunk + hop_frame
                    cut_e_out = -1 * hop_frame * hop_size

                sub_ppg = retrieval.retriv_whisper(ppg[cut_s:cut_e, :])
                sub_vec = retrieval.retriv_hubert(vec[cut_s:cut_e, :])
                sub_ppg = sub_ppg.unsqueeze(0).to(self.device)
                sub_vec = sub_vec.unsqueeze(0).to(self.device)
                sub_pit = pit[cut_s:cut_e].unsqueeze(0).to(self.device)
                sub_len = torch.LongTensor([cut_e - cut_s]).to(self.device)
                sub_har = source[:, :, cut_s *
                                 hop_size: cut_e * hop_size].to(self.device)
                sub_out = self.svcModel.inference(
                    sub_ppg, sub_vec, sub_pit, spk, sub_len, sub_har)
                sub_out = sub_out[0, 0].data.cpu().detach().numpy()

                sub_out = sub_out[cut_s_out:cut_e_out]
                out_audio.extend(sub_out)
                out_index = out_index + out_chunk

            out_audio = np.asarray(out_audio)
        return out_audio
