# Ke Chen
# knutchen@ucsd.edu
# Zero-shot Audio Source Separation via Query-based Learning from Weakly-labeled Data
# The Main Script

import os

gpu_use = 0
# this is to avoid the sdr calculation from occupying all cpus
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import librosa
import numpy as np
import soundfile as sf
import pickle

import torch
from torch.utils.data import DataLoader
from utils import collect_fn, dump_config, create_folder, prepprocess_audio
from models.asp_model import ZeroShotASP, SeparatorModel, AutoTaggingWarpper, WhitingWarpper
from data_processor import LGSPDataset, MusdbDataset
import config
import htsat_config
from models.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper

import pytorch_lightning as pl

import time
import warnings
import shutil

warnings.filterwarnings("ignore")


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def process_file_with_vector(vector_path):
    avg_at = load_from_file_fast(vector_path)

    test_type = 'mix'
    inference_file = config.inference_file
    inference_query = config.inference_query
    # test_key = config.test_key
    # We must extract it from vector
    test_key = [list(avg_at.keys())[0]]
    wave_output_path = config.wave_output_path
    sample_rate = config.sample_rate
    resume_checkpoint_zeroshot = config.resume_checkpoint
    resume_checkpoint_htsat = htsat_config.resume_checkpoint
    print('Inference file: {}'.format(inference_file))
    print('Inference query folder: {}'.format(inference_query))
    print('Test key: {}'.format(test_key))
    print('Wave out folder: {}'.format(wave_output_path))
    print('Sample rate: {}'.format(sample_rate))
    print('Model 1 (zeroshot): {}'.format(resume_checkpoint_zeroshot))
    print('Model 2 (htsat): {}'.format(resume_checkpoint_htsat))

    # set exp settings
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda")
    create_folder(wave_output_path)
    test_track, fs = librosa.load(inference_file, sr=None)
    test_track = test_track[:, None]
    print(test_track.shape)
    print(fs)
    # convert the track into 32000 Hz sample rate
    test_track = prepprocess_audio(
        test_track,
        fs, sample_rate,
        test_type
    )
    test_tracks = []
    temp = [test_track]
    for dickey in test_key:
        temp.append(test_track)
    temp = np.array(temp)
    test_tracks.append(temp)
    dataset = MusdbDataset(tracks=test_tracks)  # the action is similar to musdbdataset, reuse it
    loader = DataLoader(
        dataset=dataset,
        num_workers=1,
        batch_size=1,
        shuffle=False
    )

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
        num_heads=htsat_config.htsat_num_head
    )
    at_model = SEDWrapper(
        sed_model=sed_model,
        config=htsat_config,
        dataset=None
    )
    ckpt = torch.load(resume_checkpoint_htsat, map_location="cpu")
    at_model.load_state_dict(ckpt["state_dict"])

    if device_name == 'cpu':
        trainer = pl.Trainer(
            accelerator="cpu", gpus=None
        )
    else:
        trainer = pl.Trainer(
            gpus=1
        )

    # import seapration model
    model = ZeroShotASP(
        channels=1,
        config=config,
        at_model=at_model,
        dataset=dataset
    )
    # resume checkpoint
    ckpt = torch.load(resume_checkpoint_zeroshot, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    exp_model = SeparatorModel(
        model=model,
        config=config,
        target_keys=test_key,
        avg_at=avg_at,
        using_wiener=False,
        calc_sdr=False,
        output_wav=True
    )
    trainer.test(exp_model, test_dataloaders=loader)
    time.sleep(0.01)
    out_file_in = wave_output_path + '/0_{}_pred_(0.0).wav'.format(test_key[0])
    out_file_q = wave_output_path + '/{}_{}.wav'.format(os.path.basename(inference_file)[:-4], os.path.basename(vector_path)[:-4])

    print('Try to copy from: {}'.format(out_file_in))
    print('Try to copy to: {}'.format(out_file_q))
    if not os.path.isfile(out_file_in):
        print('File {} doesnt exists for some reason! Please check path...')
    shutil.copy(
        out_file_in,
        out_file_q,
    )


if __name__ == '__main__':
    try:
        # Try to take from config
        vector_path = config.vector_path
    except:
        # If not provided use hardcoded path
        vector_path = './vectors/bass_vector.pkl'
        if not os.path.isfile(vector_path):
            print('Vector doesn\'t exists: {}'.format(vector_path))
            exit()
    process_file_with_vector(vector_path)
