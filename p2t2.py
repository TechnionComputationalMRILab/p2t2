from typing import Optional

def simulate(
    config_file: str = "config.yaml",
    model_type: str = "P2T2",
    min_te: float = 5.0,
    max_te: float = 15.0,
    n_echoes: int = 20,
    num_signals: int = 10000,
    out_folder: str = 'data'
):
    """Generate simulated data using a configuration file.

    Args:
        config_file (str, optional): Configuration file in yaml format. Defaults to "config.yaml".
        model_type (str, optional): Model type. Defaults to "P2T2".
        min_te (float, optional): Minimum TE. Defaults to 5.0.
        max_te (float, optional): Maximum TE. Defaults to 15.0.
        n_echoes (int, optional): Number of echoes. Defaults to 20.
        num_signals (int, optional): Number of signals. Defaults to 10000.
        out_folder (str, optional): Destination folder for the simulated data. Defaults to 'data'.
    """
    from data_simulation import main
    
    main(
        config_path=config_file,
        model_type=model_type,
        min_te=min_te,
        max_te=max_te,
        n_echoes=n_echoes,
        num_signals=num_signals,
        out_folder=out_folder
    )


def train(
    config: str = 'config.yaml',
    data_folder: str = "data",
    output_path: str = "runs",
    model_type: str = 'P2T2-FC',
    min_te: float = 7.9,
    max_te: Optional[float] = None
):
    import argparse
    from pathlib import Path
    from datetime import datetime
    from pt2_reconstruction_model_main import main

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.config = config
    args.data_folder = data_folder

    Path(output_path).mkdir(parents=True, exist_ok=True)
    args.runs_outputs_path = output_path

    now = datetime.now()
    args.dt_string = now.strftime("%y%m%d_%H_%M_%S")

    main(
        args, 
        model_type, 
        min_te, 
        max_te
    )


def infer(
    model_type: str = "P2T2-FC",
    model_path: str = "model.pt",
    model_args_path: str = "model_args.yaml",
    data_dict: dict = {},
    output_dir: str = "output",
    n_echoes: Optional[int] = None,
):
    from pathlib import Path
    import SimpleITK as sitk
    import numpy as np
    from box import Box
    import torch
    import yaml
    import os

    from pt2_reconstruction_model_deploy import load_model, deploy_model, load_mri_data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model config
    model_args = Box(yaml.safe_load(open(model_args_path)))
    if hasattr(model_args, 'max_echoes'):
        model_args.num_echoes = model_args.max_echoes
    else:
        model_args.num_echoes = n_echoes    
    num_used_echoes = model_args.n_echoes 

    # load data
    data = load_mri_data(data_dict, num_used_echoes)
    model_args.TEs = data['TEs']

    # load model
    model = load_model(model_path, device, model_type, model_args)

    # deploy model
    pt2_pred, pred_mri, predicted_fa = deploy_model(model, data, device, model_type, model_args)

    # save output
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pt2_pred_array = sitk.GetImageFromArray(np.transpose(pt2_pred.cpu().numpy(), (2, 1, 0)))

    pred_mri_array = sitk.GetImageFromArray(np.transpose(pred_mri.cpu().numpy(), (2, 1, 0)))

    predicted_fa_array = sitk.GetImageFromArray(np.transpose(predicted_fa.cpu().numpy(), (1, 0)))

    sitk.WriteImage(
        pt2_pred_array, 
        os.path.join(output_dir, 'pt2_pred.nii.gz')
    )
    sitk.WriteImage(
        pred_mri_array, 
        os.path.join(output_dir, 'pred_mri.nii.gz')
    )
    sitk.WriteImage(
        predicted_fa_array, 
        os.path.join(output_dir, 'predicted_fa.nii.gz')
    )
