# Ke Chen
# knutchen@ucsd.edu
# Zero-shot Audio Source Separation via Query-based Learning from Weakly-labeled Data
# The dataset classes

import numpy as np
import torch
import logging
import os
import sys
import h5py
import csv
import time
import random
import json
from datetime import datetime
from utils import int16_to_float32

from torch.utils.data import Dataset, Sampler

# output the dict["index"].key form to save the memory in multi-GPU training
def reverse_dict(data_path, sed_path, output_dir):
    # filename 
    waveform_dir = os.path.join(output_dir, "audioset_eval_waveform_balanced.h5")
    sed_dir = os.path.join(output_dir, "audioset_eval_sed_balanced.h5")
    # load data
    logging.info("Write Data...............")
    h_data = h5py.File(data_path, "r")
    h_sed = h5py.File(sed_path, "r")
    audio_num = len(h_data["waveform"])
    assert len(h_data["waveform"]) == len(h_sed["sed_vector"]), "waveform and sed should be in the same length"
    with h5py.File(waveform_dir, 'w') as hw:
        for i in range(audio_num):
            hw.create_dataset(str(i), data=int16_to_float32(h_data['waveform'][i]), dtype=np.float32)
    logging.info("Write Data Succeed...............")
    logging.info("Write Sed...............")
    with h5py.File(sed_dir, 'w') as hw:
        for i in range(audio_num):
            hw.create_dataset(str(i), data=h_sed['sed_vector'][i], dtype=np.float32)     
    logging.info("Write Sed Succeed...............")

# A dataset for handling musdb
class MusdbDataset(Dataset):
    def __init__(self, tracks):
        self.tracks = tracks
        self.dataset_len = len(tracks)
    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: 
            track: [mixture + n_sources, n_samples]
        """
        return self.tracks[index]
    def __len__(self):
        return self.dataset_len

class InferDataset(Dataset):
    def __init__(self, tracks):
        self.tracks = tracks
        self.dataset_len = len(tracks)
    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: 
            track: [mixture + n_sources, n_samples]
        """
        return self.tracks[index]
    def __len__(self):
        return self.dataset_len

# polished LGSPDataset, the main dataset for procssing the audioset files
class LGSPDataset(Dataset):
    def __init__(self, index_path, idc, config, factor = 3, eval_mode = False):
        self.index_path = index_path
        self.fp = h5py.File(index_path, "r")
        self.config = config
        self.idc = idc
        self.factor = factor
        self.classes_num = self.config.classes_num
        self.eval_mode = eval_mode
        self.total_size = int(len(self.fp["audio_name"]) * self.factor)
        self.generate_queue()
        logging.info("total dataset size: %d" %(self.total_size))
        logging.info("class num: %d" %(self.classes_num))

    def generate_queue(self):
        self.queue = []      
        self.class_queue = []
        if self.config.debug:
            self.total_size = 1000
        if self.config.balanced_data:
            while len(self.queue) < self.total_size * 2:
                if self.eval_mode:
                    if len(self.config.eval_list) == 0:
                        class_set = [*range(self.classes_num)]
                    else:
                        class_set = self.config.eval_list[:]
                else:
                    class_set = [*range(self.classes_num)]
                    class_set = list(set(class_set) - set(self.config.eval_list))
                random.shuffle(class_set)
                self.queue += [self.idc[d][random.randint(0, len(self.idc[d]) - 1)] for d in class_set]
                self.class_queue += class_set[:]
            self.queue = self.queue[:self.total_size * 2]
            self.class_queue = self.class_queue[:self.total_size * 2]
            self.queue = [[self.queue[i],self.queue[i+1]] for i in range(0, self.total_size * 2, 2)]
            self.class_queue = [[self.class_queue[i],self.class_queue[i+1]] for i in range(0, self.total_size * 2, 2)]
            assert len(self.queue) == self.total_size, "generate data error!!" 
        else:
            if self.eval_mode:
                    if len(self.config.eval_list) == 0:
                        class_set = [*range(self.classes_num)]
                    else:
                        class_set = self.config.eval_list[:]
            else:
                class_set = [*range(self.classes_num)]
                class_set = list(set(class_set) - set(self.config.eval_list))
            self.class_queue = random.choices(class_set, k = self.total_size * 2)
            self.queue = [self.idc[d][random.randint(0, len(self.idc[d]) - 1)] for d in self.class_queue]
            self.queue = [[self.queue[i],self.queue[i+1]] for i in range(0, self.total_size * 2, 2)]
            self.class_queue = [[self.class_queue[i],self.class_queue[i+1]] for i in range(0, self.total_size * 2, 2)]
            assert len(self.queue) == self.total_size, "generate data error!!" 
        logging.info("queue regenerated:%s" %(self.queue[-5:]))

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name_1": str,
            "waveform_1": (clip_samples,),
            "class_id_1": int,
            "audio_name_2": str,
            "waveform_2": (clip_samples,),
            "class_id_2": int,
            ...
            "check_num": int
        }
        """
        # put the right index here!!!
        data_dict = {}
        for k in range(2):
            s_index = self.queue[index][k]
            target = self.class_queue[index][k]
            audio_name = self.fp["audio_name"][s_index].decode()
            hdf5_path = self.fp["hdf5_path"][s_index].decode().replace("/home/tiger/DB/knut/data/audioset", self.config.dataset_path)
            r_idx = self.fp["index_in_hdf5"][s_index]
            with h5py.File(hdf5_path, "r") as f:
                waveform = int16_to_float32(f["waveform"][r_idx])
            data_dict["audio_name_" + str(k+1)] = audio_name
            data_dict["waveform_" + str(k+1)] = waveform
            data_dict["class_id_" + str(k+1)] = target
        data_dict["check_num"] = str(self.queue[-5:])
        return data_dict

    def __len__(self):
        return self.total_size

# only for test
class TestDataset(Dataset):
    def __init__(self, dataset_size):
        print("init")
        self.dataset_size = dataset_size
        self.base_num = 100
        self.dicts = [(self.base_num + 2 * i, self.base_num + 2 * i + 1) for i in range(self.dataset_size)]
    
    def get_new_list(self):
        self.base_num = random.randint(0,10)
        print("base num changed:", self.base_num)
        self.dicts = [(self.base_num + 2 * i, self.base_num + 2 * i + 1) for i in range(self.dataset_size)]

    def __getitem__(self, index):
        return self.dicts[index]
    
    def __len__(self):
        return self.dataset_size

