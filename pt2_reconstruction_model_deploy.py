import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from monai.transforms import LoadImaged
import torch
from box import Box
import yaml
import json
import random
import math

from torch.utils.data import Dataset, DataLoader

from pt2_reconstruction_model_deploy_utils import  EPG_Generator, get_pixelwise_models
from pt2_reconstruction_model_utils import T2_Dataset
import tqdm
import glob

def normalize_mri(mri):
    first_echo = mri[0].clone()
    first_echo[first_echo == 0] = 1
    mri = mri / first_echo
    return mri



def load_mri_data(data_list, num_used_echoes):
    """
    :param data_list: dict, path to mri data and metadata

    """
    print('Uploading MRI data...')

    loader =  LoadImaged(keys=["mri"], ensure_channel_first=False)
    data = loader(data_list)
    data['mri'] = torch.moveaxis(data['mri'], -1, 0)  #  nTE, W, H

    img_metadata = json.load(open(data['metadata']))
    data['TEs'] = torch.tensor(img_metadata['TE_array']).float() 
    
    
    if num_used_echoes != data['mri'].shape[0]:
        data['mri'] = data['mri'][:num_used_echoes, ...]
        data['TEs'] = data['TEs'][:num_used_echoes]
    
    data['mri'] = normalize_mri(data['mri'])
    data['mri'] = data['mri'].unsqueeze(0)
    

    print('MRI data uploaded')
    return data


def convert_data_for_pixelwise_models(data):
    image = data['mri']
    te_arr = data['TEs']
    B, N, W, H = image.shape
    data = []
    for x in range(W):
        for y in range(H):
            norm_signal = image[:,:, x, y].squeeze()  # normalized during data loading
            tmp_data = {
                'ind': [x, y],
                'TE': te_arr,
                's_norm': norm_signal,
            }
            data.append(tmp_data)
    data_ds = T2_Dataset(data)
    data_dl = DataLoader(
        data_ds,
        batch_size=5000,
        shuffle=False,
        num_workers=4
        )
    return data_dl

def load_model(model_path, device, model_type, model_args):
    """
    :param model_path: str, path to the model
    :param device: str, device to run the model
    :param model_type: str, type of the model, e.g. 'P2T2-FC', 'MIML'
    """
    print('Loading the model...')
    model_weights = torch.load(model_path, map_location=device)
    model = get_pixelwise_models(
        model_type, 
        n_echoes=model_args.n_echoes,
        out_ch=60, 
        device=device
        ) 

    
    model.load_state_dict(model_weights)
    model.to(device)
    print('Model loaded')
    return model


def get_pred_mri_from_pixels_model(epg_array, p2t2_pred, observed_singal):

    #shape of simulated signal: flip angles, number pixel areas, number TEs
    sim_signal = torch.zeros((epg_array.shape[0], p2t2_pred.shape[0], epg_array.shape[1]))
    for fa in range(epg_array.shape[0]):
        for sig_dist in range(p2t2_pred.shape[0]): # B, 60
            #maybe change KM parameter
            sim_signal[fa][sig_dist] = torch.sum(epg_array[fa] * p2t2_pred[sig_dist], dim=1)
    #normalize simulated signal with first TE signal
    sim_first_te = torch.clip(sim_signal[:,:,0], min=1e-6)
    sim_signal = sim_signal / sim_first_te.unsqueeze(-1)
    # 5. calculate MSE
    mse = torch.zeros(epg_array.shape[0], p2t2_pred.shape[0])
    for fa in range(sim_signal.shape[0]):
        mse[fa] = ((sim_signal[fa] - observed_singal)**2).mean(1)

    best_fa = torch.argmin(mse, dim=0)
    pred_mri = sim_signal[best_fa, torch.arange(sim_signal.shape[1])]
    best_fa += 90
    return best_fa, pred_mri

def deploy_model(model, data, device, model_type, model_args):
    """
    :param model: torch.nn.Module, model to be deployed
    :param data: dict, input data
    :param device: str, device to run the model
    """
    print('Deploying the model...')
    model.eval()
    with torch.no_grad():
        # convert the data to the format required by the pixelwise models
        data_dl = convert_data_for_pixelwise_models(data)
        B, N, W, H = data['mri'].shape
        pt2_pred = torch.zeros((W, H, 60), dtype=torch.float32)
        predicted_fa = torch.zeros((W, H), dtype=torch.int)
        pred_mri = torch.zeros((W, H, N), dtype=torch.float32)
        # compute epgs dictionary
        print('Computing EPGs...')
        epgs_gen = EPG_Generator(data['TEs'].shape[0], TEmin=data['TEs'][0], fa_list=np.arange(90,180,1), T2grid_type='log', t2_range=[10., 2000.])
        epgs = torch.Tensor(epgs_gen.sim_multiple_epgs())
        print('EPGs computed')
        for batch_data in tqdm.tqdm(data_dl):
            img_indexs, mri_signals, echoes = batch_data["ind"], batch_data["s_norm"].to(device), batch_data["TE"].to(device)
            if model_type == 'MIML':
                pt2_pred_batch = model(mri_signals)
            else:
                # concatenate the used TEs to the input
                input = torch.cat([mri_signals, echoes], dim=-1).to(device)
                pt2_pred_batch = model(input)
            fa_pred_batch, mri_pred_batch = get_pred_mri_from_pixels_model(epgs, pt2_pred_batch.detach().cpu(), mri_signals.detach().cpu())
            pred_mri[img_indexs[0], img_indexs[1], :] = mri_pred_batch.detach().cpu()
            predicted_fa[img_indexs[0], img_indexs[1]] = fa_pred_batch.detach().cpu().squeeze().int()
            pt2_pred[img_indexs[0], img_indexs[1], :] = pt2_pred_batch.detach().cpu()
       

    return pt2_pred, pred_mri, predicted_fa




if __name__=='__main__':
    models_to_test = {
        'MIML': '/path/to/trained/MIML/model',
        'P2T2-FC': '/path/to/trained/P2T2-FC/model'
    }
    # load the model
    model_type = 'MIML'
    model_path = os.path.join(models_to_test[model_type], 'best_model.pt')
    device = 'cuda'
    model_args_path = glob.glob(os.path.join(models_to_test[model_type], '*.yaml'))[0]
    model_args = Box(yaml.safe_load(open(model_args_path)))
    if hasattr(model_args, 'max_echoes'):
        model_args.num_echoes = model_args.max_echoes
    
    num_used_echoes = model_args.n_echoes 

    # load the data
    data_root = 'mri_example/'
    data_name = 'Brats18_TCIA13_645_1_t1_230318_221644_slice_100'
    data_list = {
        'mri': os.path.join(data_root, f'{data_name}.nii.gz'),
        'metadata': os.path.join(data_root, f'{data_name}_metadata.json') 
    }
    data = load_mri_data(data_list, num_used_echoes)
    model_args.TEs = data['TEs']
    
    # load model
    model = load_model(model_path, device, model_type, model_args)

    # deploy the model
    pt2_pred, pred_mri, predicted_fa = deploy_model(model, data, device, model_type, model_args)
    
    output_dir = 'DeployModels/'
    output_folder = os.path.join(output_dir, data_name, model_type)
    os.makedirs(output_folder, exist_ok=True)

    # save the results
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(pt2_pred.cpu().numpy(), (2, 1, 0))), os.path.join(output_folder, 'pt2_pred.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(pred_mri.cpu().numpy(), (2, 1, 0))), os.path.join(output_folder, 'pred_mri.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(predicted_fa.cpu().numpy(), (1, 0))), os.path.join(output_folder, 'predicted_fa.nii.gz'))

