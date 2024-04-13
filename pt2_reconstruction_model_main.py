import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse
from box import Box
import yaml
from ruamel.yaml import RoundTripDumper
import random
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime

from pt2_reconstruction_model_utils import T2_Dataset
from data_simulation import main as data_simulation_main
from pt2_reconstruction_model_training import PT2Net_Trainer

def split_data(data, args):
    val_frac = args.val_frac
    test_frac = args.test_frac

    length = len(data)
    indices = np.arange(length)
    np.random.shuffle(indices)

    val_length = int(length * val_frac)
    test_length = int(length * test_frac)

    val_indices = indices[:val_length]
    test_indices = indices[val_length: val_length + test_length]
    train_indices = indices[val_length + test_length:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, val_data, test_data

def get_train_val_test_data(args, data_folder):

    train_args = args.Trainer
    
    train_test_split_data = os.path.join(data_folder, 'train_val_test_sets')
    train_data_path = os.path.join(train_test_split_data, "T2_train_data.pkl")
    val_data_path = os.path.join(train_test_split_data, "T2_val_data.pkl")
    test_data_path = os.path.join(train_test_split_data, "T2_test_data.pkl")

    if os.path.exists(train_data_path) and os.path.exists(val_data_path) and os.path.exists(test_data_path):
        train_data = pickle.load(open(train_data_path, "rb"))
        val_data = pickle.load(open(val_data_path, "rb"))

    else:
        if not os.path.exists(train_test_split_data):
            os.mkdir(train_test_split_data)
        water_pools = args.water_pool.keys()
        train_data = []
        val_data = []
        test_data = []
        for wp in water_pools:
            folder = os.path.join(data_folder, wp)
            file = open(os.path.join(folder, f'{wp}_data.pkl'), 'rb')
            data = pickle.load(file)
            train_wp, val_wp, test_wp = split_data(data, train_args)
            train_data += train_wp
            val_data += val_wp
            test_data += test_wp

        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        # save the subsets
        train_file = open(os.path.join(train_test_split_data, "T2_train_data.pkl"), "wb")
        pickle.dump(train_data, train_file)
        train_file.close()
        
        val_file = open(os.path.join(train_test_split_data, "T2_val_data.pkl"), "wb")
        pickle.dump(val_data, val_file)
        val_file.close()

        test_file = open(os.path.join(train_test_split_data, "T2_test_data.pkl"), "wb")
        pickle.dump(test_data, test_file)
        test_file.close()
    
    train_ds = T2_Dataset(train_data)
    val_ds = T2_Dataset(val_data)

    train_dl = DataLoader(
        train_ds,
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers
        )
    val_dl = DataLoader(
        val_ds,
        batch_size=train_args.val_batch,
        shuffle=False,
        num_workers=train_args.num_workers
        )
    
    return train_dl, val_dl



def main(args, model_type, min_te, max_te):
    with open(args.config) as f:
        data_args = Box(yaml.load(f, Loader=yaml.FullLoader))

    data_args.model_type = model_type
    data_args.min_te = min_te
    data_args.max_te = max_te

    # define the output folder:
    epgs_info= os.path.basename(os.path.dirname(args.data_folder))
    snr_info = os.path.basename(args.data_folder)
    data_args.training_path = os.path.join(args.runs_outputs_path, f'{data_args.model_type}_{epgs_info}_{snr_info}_{args.dt_string}')
    
    #TODO uncommet this line for continue a pre-trained model- 
    # data_args.training_path = os.path.join(args.runs_outputs_path, '')
    
    if not os.path.exists(data_args.training_path):
        os.mkdir(data_args.training_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # save the training args:
    # add the args from args to data_args
    data_args.data_folder = args.data_folder
    data_args.runs_outputs_path = args.runs_outputs_path
    data_args.config_path = args.config
    data_args.device = device
    data_args.model_type = model_type
    
    data_args.to_yaml(os.path.join(data_args.training_path, os.path.basename(args.config)), Dumper=RoundTripDumper)

    # simulate & upload the data
    if not os.path.exists(args.data_folder):
        print('Could not find the data folder, start simulating data')
        os.mkdir(args.data_folder)
        # simulate data
        data_simulation_main(
            config_path=args.config, 
            min_te=min_te, 
            max_te=max_te, 
            n_echoes=data_args.n_echoes, 
            out_folder=args.data_folder, 
            model_type='P2T2' if 'P2T2' in model_type else 'MIML',
            num_signals=data_args.num_epgs_signals
            )
        print('Data simulation is done')
    train_dl, val_dl = get_train_val_test_data(data_args, args.data_folder)

    # start training:
    trainer = PT2Net_Trainer(data_args, device)
    trainer.train_model(train_dl, val_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct T2 distribution from mri signal for brain data')
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()

    args.data_folder = os.path.normpath('brain_data_7.9ms_miml')
    # args.data_folder = os.path.normpath('/tcmldrive/hds/T2_simulation/EPGs/EPGs_12_nTE_20_nEPGS_10000/signals_SNR_10_80')
    args.runs_outputs_path = os.path.normpath('runs/brain_data_7.9ms')
    os.makedirs(args.runs_outputs_path, exist_ok=True) 

    now = datetime.now()
    args.dt_string = now.strftime("%y%m%d_%H_%M_%S")
    model_type = 'MIML'
    min_te=7.9 
    max_te=None

    main(args, model_type, min_te, max_te)