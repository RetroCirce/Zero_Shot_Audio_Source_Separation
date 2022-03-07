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
from hashlib import md5

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
import tqdm
import warnings
import shutil
import pickle
warnings.filterwarnings("ignore")

# use the model to quickly separate a track given a query
# it requires four variables in config.py:
#   inference_file: the track you want to separate
#   inference_query: a **folder** containing all samples from the same source
#   test_key: ["name"] indicate the source name (just a name for final output, no other functions)
#   wave_output_path: the output folder

# make sure the query folder contain the samples from the same source
# each time, the model is able to separate one source from the track
# if you want to separate multiple sources, you need to change the query folder or write a script to help you do that


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'), protocol=4)


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def create_vector():
    test_type = 'mix'
    inference_file = config.inference_file
    inference_query = config.inference_query
    test_key = config.test_key
    wave_output_path = config.wave_output_path
    sample_rate = config.sample_rate
    resume_checkpoint_zeroshot = config.resume_checkpoint
    resume_checkpoint_htsat = htsat_config.resume_checkpoint
    print('Inference query folder: {}'.format(inference_query))
    print('Test key: {}'.format(test_key))
    print('Vector out folder: {}'.format(wave_output_path))
    print('Sample rate: {}'.format(sample_rate))
    print('Model 1 (zeroshot): {}'.format(resume_checkpoint_zeroshot))

    # set exp settings
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda")
    create_folder(wave_output_path)

    # obtain the samples for query
    queries = []
    query_names = []
    for query_file in tqdm.tqdm(os.listdir(inference_query)):
        f_path = os.path.join(inference_query, query_file)
        if query_file.endswith(".wav"):
            temp_q, fs = librosa.load(f_path, sr=None)
            temp_q = temp_q[:, None]
            temp_q = prepprocess_audio(
                temp_q, 
                fs,
                sample_rate,
                test_type
            )
            temp = [temp_q]
            for dickey in test_key:
                temp.append(temp_q)
            temp = np.array(temp)
            queries.append(temp)
            query_names.append(os.path.basename(query_file))

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

    print('Process: {}'.format(len(queries)))
    avg_dataset = MusdbDataset(
        tracks=queries
    )
    avg_loader = DataLoader(
        dataset=avg_dataset,
        num_workers=1,
        batch_size=1,
        shuffle=False
    )
    at_wrapper = AutoTaggingWarpper(
        at_model=at_model,
        config=config,
        target_keys=test_key
    )
    trainer.test(
        at_wrapper,
        test_dataloaders=avg_loader
    )
    avg_at = at_wrapper.avg_at

    md5_str = str(md5(str(queries).encode('utf-8')).hexdigest())
    out_vector_path = wave_output_path + '/{}_vector_{}.pkl'.format(test_key[0], md5_str)
    save_in_file_fast(avg_at, out_vector_path)
    print('Vector saved in: {}'.format(out_vector_path))


if __name__ == '__main__':
    create_vector()
