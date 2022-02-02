# Ke Chen
# knutchen@ucsd.edu
# Zero-shot Audio Source Separation via Query-based Learning from Weakly-labeled Data
# The Main Script

import os
# this is to avoid the sdr calculation from occupying all cpus
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "6" 

import sys
import librosa
import numpy as np
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import collect_fn, dump_config, create_folder, prepprocess_audio
import musdb

from models.asp_model import ZeroShotASP, SeparatorModel, AutoTaggingWarpper, WhitingWarpper
from data_processor import LGSPDataset, MusdbDataset
import config
import htsat_config
from models.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from htsat_utils import process_idc

import warnings
warnings.filterwarnings("ignore")



class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, device_num, config):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device_num = device_num
        self.config = config

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle = False) if self.device_num > 1 else None
        train_loader = DataLoader(
            dataset = self.train_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = train_sampler,
            collate_fn = collect_fn
        )
        return train_loader
    def val_dataloader(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        eval_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = eval_sampler,
            collate_fn = collect_fn
        )
        return eval_loader
    def test_dataloader(self):
        test_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        test_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size // self.device_num,
            shuffle = False,
            sampler = test_sampler,
            collate_fn = collect_fn
        )
        return test_loader

def save_idc():
    train_index_path = os.path.join(config.dataset_path, "hdf5s", "indexes", config.index_type + ".h5")
    eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
    process_idc(train_index_path, config.classes_num,  config.index_type + "_idc.npy")
    process_idc(eval_index_path, config.classes_num, "eval_idc.npy")

# Process the musdb tracks into the sample rate of 32000 Hz sample rate, the original is 44100 Hz
def process_musdb():
    # use musdb as testset
    test_data = musdb.DB(
        root = config.musdb_path,
        download = False,
        subsets = "test",
        is_wav = True
    )
    print(len(test_data.tracks))
    mus_tracks = []
    # in musdb, all fs is the same (44100)
    orig_fs = test_data.tracks[0].rate
    print(orig_fs)
    for track in test_data.tracks:
        temp = {}
        mixture = prepprocess_audio(
            track.audio,
            orig_fs, config.sample_rate, 
            config.test_type
        )
        temp["mixture" ]= mixture
        for dickey in config.test_key:
            source = prepprocess_audio(
                track.targets[dickey].audio, 
                orig_fs, config.sample_rate,
                config.test_type
            )
            temp[dickey] = source
        print(track.audio.shape, len(temp.keys()), temp["mixture"].shape)
        mus_tracks.append(temp)
    print(len(mus_tracks))
    # save the file to npy
    np.save("musdb-32000fs.npy", mus_tracks)

# weight average will perform in the given folder
# It will output one model checkpoint, which avergas the weight of all models in the folder
def weight_average():
    model_ckpt = []
    model_files = os.listdir(config.wa_model_folder)
    wa_ckpt = {
        "state_dict": {}
    }

    for model_file in model_files:
        model_file = os.path.join(config.esm_model_folder, model_file)
        model_ckpt.append(torch.load(model_file, map_location="cpu")["state_dict"])
    keys = model_ckpt[0].keys()
    for key in keys:
        model_ckpt_key = torch.cat([d[key].float().unsqueeze(0) for d in model_ckpt])
        model_ckpt_key = torch.mean(model_ckpt_key, dim = 0)
        assert model_ckpt_key.shape == model_ckpt[0][key].shape, "the shape is unmatched " + model_ckpt_key.shape + " " + model_ckpt[0][key].shape
        wa_ckpt["state_dict"][key] = model_ckpt_key
    torch.save(wa_ckpt, config.wa_model_path)


# use the model to quickly separate a track given a query
# it requires four variables in config.py:
#   inference_file: the track you want to separate
#   inference_query: a **folder** containing all samples from the same source
#   test_key: ["name"] indicate the source name (just a name for final output, no other functions)
#   wave_output_path: the output folder

# make sure the query folder contain the samples from the same source
# each time, the model is able to separate one source from the track
# if you want to separate multiple sources, you need to change the query folder or write a script to help you do that
def inference():
    # set exp settings
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda")
    assert config.test_key is not None, "there should be a separate key"
    create_folder(config.wave_output_path)
    test_track, fs = librosa.load(config.inference_file, sr = None)
    test_track = test_track[:,None]
    print(test_track.shape)
    print(fs)
    # convert the track into 32000 Hz sample rate
    test_track = prepprocess_audio(
        test_track, 
        fs, config.sample_rate,
        config.test_type
        )
    test_tracks = []
    temp = [test_track]
    for dickey in config.test_key:
        temp.append(test_track)
    temp = np.array(temp)
    test_tracks.append(temp)
    dataset = MusdbDataset(tracks = test_tracks) # the action is similar to musdbdataset, reuse it
    loader = DataLoader(
        dataset = dataset,
        num_workers = 1,
        batch_size = 1,
        shuffle = False
    )
    # obtain the samples for query
    queries = []
    for query_file in os.listdir(config.inference_query):
        f_path = os.path.join(config.inference_query, query_file)
        if query_file.endswith(".wav"):
            temp_q, fs = librosa.load(f_path, sr = None)
            temp_q = temp_q[:, None]
            temp_q = prepprocess_audio(
                temp_q, 
                fs, config.sample_rate,
                config.test_type
            )
            temp = [temp_q]
            for dickey in config.test_key:
                temp.append(temp_q)
            temp = np.array(temp)
            queries.append(temp)
        
    assert config.resume_checkpoint is not None, "there should be a saved model when inferring"
    
    sed_model = HTSAT_Swin_Transformer(
        spec_size=htsat_config.htsat_spec_size,
        patch_size=htsat_config.htsat_patch_size,
        in_chans=1,
        num_classes=htsat_config.classes_num,
        window_size=htsat_config.htsat_window_size,
        config = htsat_config,
        depths = htsat_config.htsat_depth,
        embed_dim = htsat_config.htsat_dim,
        patch_stride=htsat_config.htsat_stride,
        num_heads=htsat_config.htsat_num_head
    )
    at_model = SEDWrapper(
        sed_model = sed_model, 
        config = htsat_config,
        dataset = None
    )
    ckpt = torch.load(htsat_config.resume_checkpoint, map_location="cpu")
    at_model.load_state_dict(ckpt["state_dict"])
    
    trainer = pl.Trainer(
        gpus = 1
    )
    avg_at = None
    # obtain the latent embedding as query
    if config.infer_type == "mean":
        avg_dataset = MusdbDataset(tracks = queries)
        avg_loader = DataLoader(
            dataset = avg_dataset,
            num_workers = 1,
            batch_size = 1,
            shuffle = False
        )
        at_wrapper = AutoTaggingWarpper(
            at_model = at_model,
            config = config,
            target_keys = config.test_key
        )
        trainer.test(at_wrapper, test_dataloaders = avg_loader)
        avg_at = at_wrapper.avg_at

    # import seapration model
    model = ZeroShotASP(
        channels = 1, config = config, 
        at_model = at_model, 
        dataset = dataset
    )
    # resume checkpoint
    ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict= False)
    exp_model = SeparatorModel(
        model = model,
        config = config,
        target_keys = config.test_key,
        avg_at = avg_at,
        using_wiener = False,
        calc_sdr = False,
        output_wav = True
    )
    trainer.test(exp_model, test_dataloaders = loader)

# test the separation model, mainly in musdb
def test():
    # set exp settings
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda")
    assert config.test_key is not None, "there should be a separate key"
    create_folder(config.wave_output_path)
    # use musdb as testset
    test_data = np.load(config.testset_path, allow_pickle = True)
    print(len(test_data))
    mus_tracks = []
    # in musdb, all fs is the same (44100)
    # load the dataset
    for track in test_data:
        temp = []
        mixture = track["mixture"]
        temp.append(mixture)
        for dickey in config.test_key:
            source = track[dickey]
            temp.append(source)
        temp = np.array(temp)
        print(temp.shape)
        mus_tracks.append(temp)
    print(len(mus_tracks))
    dataset = MusdbDataset(tracks = mus_tracks)
    loader = DataLoader(
        dataset = dataset,
        num_workers = 1,
        batch_size = 1,
        shuffle = False
    )
    assert config.resume_checkpoint is not None, "there should be a saved model when inferring"
    
    sed_model = HTSAT_Swin_Transformer(
        spec_size=htsat_config.htsat_spec_size,
        patch_size=htsat_config.htsat_patch_size,
        in_chans=1,
        num_classes=htsat_config.classes_num,
        window_size=htsat_config.htsat_window_size,
        config = htsat_config,
        depths = htsat_config.htsat_depth,
        embed_dim = htsat_config.htsat_dim,
        patch_stride=htsat_config.htsat_stride,
        num_heads=htsat_config.htsat_num_head
    )
    at_model = SEDWrapper(
        sed_model = sed_model, 
        config = htsat_config,
        dataset = None
    )
    ckpt = torch.load(htsat_config.resume_checkpoint, map_location="cpu")
    at_model.load_state_dict(ckpt["state_dict"])
    trainer = pl.Trainer(
        gpus = 1
    )
    avg_at = None
    # obtain the query of four stems from the training set
    if config.infer_type == "mean":
        avg_data = np.load(config.testavg_path, allow_pickle = True)[:90]
        print(len(avg_data))
        avgmus_tracks = []
        # in musdb, all fs is the same (44100)
        # load the dataset
        for track in avg_data:
            temp = []
            mixture = track["mixture"]
            temp.append(mixture)
            for dickey in config.test_key:
                source = track[dickey]
                temp.append(source)
            temp = np.array(temp)
            print(temp.shape)
            avgmus_tracks.append(temp)
        print(len(avgmus_tracks))
        avg_dataset = MusdbDataset(tracks = avgmus_tracks)
        avg_loader = DataLoader(
            dataset = avg_dataset,
            num_workers = 1,
            batch_size = 1,
            shuffle = False
        )
        at_wrapper = AutoTaggingWarpper(
            at_model = at_model,
            config = config,
            target_keys = config.test_key
        )
        trainer.test(at_wrapper, test_dataloaders = avg_loader)
        avg_at = at_wrapper.avg_at
    
    model = ZeroShotASP(
        channels = 1, config = config, 
        at_model = at_model, 
        dataset = dataset
    )
    ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict= False)
    exp_model = SeparatorModel(
        model = model,
        config = config,
        target_keys = config.test_key,
        avg_at = avg_at,
        using_wiener = config.using_wiener
    )
    trainer.test(exp_model, test_dataloaders = loader)

def train():
    # set exp settings
    # device_name = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda")
    
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)

    train_index_path = os.path.join(config.dataset_path, "hdf5s","indexes", config.index_type + ".h5")
    train_idc = np.load(os.path.join(config.idc_path, config.index_type + "_idc.npy"), allow_pickle = True)
    
    eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
    eval_idc = np.load(os.path.join(config.idc_path, "eval_idc.npy"), allow_pickle = True)

    # set exp folder
    exp_dir = os.path.join(config.workspace, "results", config.exp_name)
    checkpoint_dir = os.path.join(config.workspace, "results", config.exp_name, "checkpoint")
    
    if not config.debug:
        create_folder(os.path.join(config.workspace, "results"))
        create_folder(exp_dir)
        create_folder(checkpoint_dir)
        dump_config(config, os.path.join(exp_dir, config.exp_name), False)
        
    # load data
    # import dataset LGSPDataset (latent general source separation) and sampler
    dataset = LGSPDataset(
        index_path = train_index_path,
        idc = train_idc,
        config = config,
        factor = 0.05,
        eval_mode = False
    )
    eval_dataset = LGSPDataset(
        index_path = eval_index_path,
        idc = eval_idc,
        config = config,
        factor = 0.05,
        eval_mode = True
    )

    audioset_data = data_prep(train_dataset=dataset,eval_dataset=eval_dataset,device_num=device_num, config=config)
    checkpoint_callback = ModelCheckpoint(
        monitor = "mixture_sdr",
        filename='l-{epoch:d}-{mixture_sdr:.3f}-{clean_sdr:.3f}-{silence_sdr:.3f}',
        save_top_k = 10,
        mode = "max"
    )
    # infer at model
    sed_model = HTSAT_Swin_Transformer(
        spec_size=htsat_config.htsat_spec_size,
        patch_size=htsat_config.htsat_patch_size,
        in_chans=1,
        num_classes=htsat_config.classes_num,
        window_size=htsat_config.htsat_window_size,
        config = htsat_config,
        depths = htsat_config.htsat_depth,
        embed_dim = htsat_config.htsat_dim,
        patch_stride=htsat_config.htsat_stride,
        num_heads=htsat_config.htsat_num_head
    )
    at_model = SEDWrapper(
        sed_model = sed_model, 
        config = htsat_config,
        dataset = None
    )
    # load the checkpoint
    ckpt = torch.load(htsat_config.resume_checkpoint, map_location="cpu")
    at_model.load_state_dict(ckpt["state_dict"])
    
    trainer = pl.Trainer(
        deterministic=True,
        default_root_dir = checkpoint_dir,
        gpus = device_num,
        val_check_interval = 0.2,
        # check_val_every_n_epoch = 1,
        max_epochs = config.max_epoch,
        auto_lr_find = True,
        sync_batchnorm = True,
        callbacks = [checkpoint_callback],
        accelerator = "ddp" if device_num > 1 else None,
        resume_from_checkpoint = None, #config.resume_checkpoint,
        replace_sampler_ddp = False,
        gradient_clip_val=1.0,
        num_sanity_val_steps = 0,
    )
    model = ZeroShotASP(
        channels = 1, config = config, 
        at_model = at_model,
        dataset = dataset
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
    # trainer.test(model, datamodule = audioset_data)
    trainer.fit(model, audioset_data)

def main():
    parser = argparse.ArgumentParser(description="latent genreal source separation parser")
    subparsers = parser.add_subparsers(dest = "mode")
    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")
    parser_musdb = subparsers.add_parser("musdb_process")
    parser_saveidc = subparsers.add_parser("save_idc")
    parser_wa = subparsers.add_parser("weight_average")
    parser_infer = subparsers.add_parser("inference")
    args = parser.parse_args()
    # default settings
    logging.basicConfig(level=logging.INFO) 
    pl.utilities.seed.seed_everything(seed = config.random_seed)

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "musdb_process":
        process_musdb()
    elif args.mode == "weight_average":
        weight_average()
    elif args.mode == "save_idc":
        save_idc()
    elif args.mode == "inference":
        inference()
    else:
        raise Exception("Error Mode!")
    

if __name__ == '__main__':
    main()

