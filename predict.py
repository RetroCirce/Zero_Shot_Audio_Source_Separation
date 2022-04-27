import os
import types

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import htsat_config
from cog import BasePredictor, Input, Path
from data_processor import MusdbDataset
from models.asp_model import AutoTaggingWarpper, SeparatorModel, ZeroShotASP
from models.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper
from utils import prepprocess_audio

def get_inference_configs():
    config = types.SimpleNamespace()
    config.ckpt_path = "pretrained/zeroshot_asp_full.ckpt"
    config.sed_ckpt_path = "pretrained/htsat_audioset_2048d.ckpt"
    config.wave_output_path = "predict_outputs"
    config.test_key = "query_name"
    config.test_type = "mix"
    config.loss_type = "mae"
    config.infer_type = "mean"
    config.sample_rate = 32000
    config.segment_frames = 200
    config.hop_samples = 320
    config.energy_thres = 0.1
    config.using_whiting = False
    config.latent_dim = 2048
    config.classes_num = 527
    config.overlap_rate = 0.5
    config.num_workers = 1

    return config

def load_models(config):
    sed_model = HTSAT_Swin_Transformer(
        spec_size=htsat_config.htsat_spec_size,
        patch_size=htsat_config.htsat_patch_size,
        in_chans=1,
        num_classes=htsat_config.classes_num,
        window_size=htsat_config.htsat_window_size,
        config=htsat_config,
        depths=htsat_config.htsat_depth,
        embed_dim=htsat_config.htsat_dim,
        patch_stride=htsat_config.htsat_stride,
        num_heads=htsat_config.htsat_num_head,
    )
    at_model = SEDWrapper(sed_model=sed_model, config=htsat_config, dataset=None)

    ckpt = torch.load(config.sed_ckpt_path, map_location="cpu")
    at_model.load_state_dict(ckpt["state_dict"])

    at_wrapper = AutoTaggingWarpper(
        at_model=at_model, config=config, target_keys=[config.test_key]
    )

    asp_model = ZeroShotASP(channels=1, config=config, at_model=at_model, dataset=None)
    ckpt = torch.load(config.ckpt_path, map_location="cpu")
    asp_model.load_state_dict(ckpt["state_dict"], strict=False)

    return at_wrapper, asp_model

def get_dataloader_from_sound_file(sound_file_path, config):
    signal, sampling_rate = librosa.load(str(sound_file_path), sr=None)
    signal = prepprocess_audio(
        signal[:, None], sampling_rate, config.sample_rate, config.test_type
    )
    signal = np.array([signal, signal]) # Duplicate signal for later use
    dataset = MusdbDataset(tracks=[signal])
    data_loader = DataLoader(dataset, num_workers=config.num_workers, batch_size=1, shuffle=False)
    return data_loader


class Predictor(BasePredictor):
    def setup(self):
        self.config = get_inference_configs()
        os.makedirs(self.config.wave_output_path, exist_ok=True)
        self.at_wrapper, self.asp_model = load_models(self.config)

    def predict(
        self,
        mix_file: Path = Input(description="Reference sound to extract source from"),
        query_file: Path = Input(description="Query sound to be searched and extracted from mix"),
    ) -> Path:
        ref_loader = get_dataloader_from_sound_file(str(mix_file), self.config)

        query_loader = get_dataloader_from_sound_file(str(query_file), self.config)

        trainer = pl.Trainer(gpus=1)
        trainer.test(self.at_wrapper, test_dataloaders=query_loader)
        avg_at = self.at_wrapper.avg_at

        exp_model = SeparatorModel(
            model=self.asp_model,
            config=self.config,
            target_keys=[self.config.test_key],
            avg_at=avg_at,
            using_wiener=False,
            calc_sdr=False,
            output_wav=True,
        )
        trainer.test(exp_model, test_dataloaders=ref_loader)

        prediction_path = os.path.join(
            self.config.wave_output_path, f"0_{self.config.test_key}_pred_(0.0).wav"
        )
        return prediction_path
