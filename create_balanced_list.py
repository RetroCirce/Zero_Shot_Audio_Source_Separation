# Ke Chen
# knutchen@ucsd.edu
import os
import sys
import config
import logging
import numpy as np

from utils import get_balanced_class_list

def main():
    train_indexes_hdf5_path = os.path.join(config.dataset_path, "hdf5s", "indexes", 
        "{}.h5".format(config.data_type))

    eval_indexes_hdf5_path = os.path.join(config.dataset_path, "hdf5s", "indexes", "eval.h5")
    logging.info("Process training data")
    indexes_per_class = get_balanced_class_list(train_indexes_hdf5_path, random_seed = config.random_seed)
    np.save("idc_train.npy", indexes_per_class)
    logging.info("Process testing data")
    indexes_per_class = get_balanced_class_list(eval_indexes_hdf5_path, random_seed = config.random_seed)
    np.save("idc_eval.npy", indexes_per_class)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()