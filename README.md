# Robust Reconstruction of p(T2) from Multi-Echo T2 MRI Data

This repository contains the implementation of the methods described in the paper by Hadas Ben-Atya and Moti Freiman:

**"P2T2: A physically-primed deep-neural-network approach for robust T2 distribution estimation from quantitative T2-weighted MRI,"** Computerized Medical Imaging and Graphics, Volume 107, 2023, 102240, ISSN 0895-6111.

[Read the paper](https://www.sciencedirect.com/science/article/pii/S0895611123000587)

## Overview

This project focuses on the robust estimation of T2 distributions from quantitative T2-weighted MRI data using deep learning approaches described in the P2T2 and MIML papers. The repository includes scripts for simulating Echo Planar Graphs (EPGs) and training a model to reconstruct T2 distributions.

### Prerequisites

Before running the simulations and the model, ensure you have the following installed:
- Python 3.10
- NumPy
- PyTorch
- Any other dependencies listed in `requirements.txt`

Install the required packages using:
```bash
pip install -r requirements.txt
```



### Data Simulation

```
usage: p2t2_simulate [-h] [--config_file CONFIG_FILE] [--out_folder OUT_FOLDER] [--model_type {P2T2,MIML}] [--min_te MIN_TE] [--max_te MAX_TE] [--n_echoes N_ECHOES] [--num_signals NUM_SIGNALS]

Reconstruct T2 distribution from mri signal for brain data

options:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE, -c CONFIG_FILE
                        Path to config file
  --out_folder OUT_FOLDER, -o OUT_FOLDER
                        Path to output folder
  --model_type {P2T2,MIML}
                        Model type. 'MIML' for single TE sequence or 'P2T2' for varied TE sequences. Default is P2T2
  --min_te MIN_TE       Minimum echo time. Default is 5.0
  --max_te MAX_TE       Maximum echo time (only for P2T2 type). Default is 15.0
  --n_echoes N_ECHOES   Number of echoes (only for MIML type). Default is 20
  --num_signals NUM_SIGNALS
                        Number of signals (only for MIML type). Default is 10000
```

A sample config.yaml file is provided

### Model Training

To train the model, configure the settings in `config.yaml` and run `pt2_reconstruction_model_main.py`.

```
usage: p2t2_train [-h] --config_file CONFIG_FILE --data_folder DATA_FOLDER --output_path OUTPUT_PATH [--model_type {P2T2,MIML}] [--min_te MIN_TE] [--max_te MAX_TE]

Reconstruct T2 distribution from mri signal for brain data

options:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE, -c CONFIG_FILE
                        Path to config file
  --data_folder DATA_FOLDER, -d DATA_FOLDER
                        Path to data folder
  --output_path OUTPUT_PATH, -o OUTPUT_PATH
                        Path to output folder
  --model_type {P2T2,MIML}
                        Model type. 'MIML' for single TE sequence or 'P2T2' for varied TE sequences. Default is P2T2
  --min_te MIN_TE       Minimum echo time. Default is 7.9
  --max_te MAX_TE       Maximum echo time. Optional
```

### Model Inference
```
usage: p2t2_infer [-h] --model_path MODEL_PATH --model_args_path MODEL_ARGS_PATH --output_dir OUTPUT_DIR --mri MRI --metadata METADATA [--model_type {P2T2,MIML}] [--n_echoes N_ECHOES]

Reconstruct T2 distribution from mri signal for brain data

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH, -m MODEL_PATH
                        Path to model
  --model_args_path MODEL_ARGS_PATH, -a MODEL_ARGS_PATH
                        Path to model args
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory
  --mri MRI
  --metadata METADATA
  --model_type {P2T2,MIML}
                        Model type. 'MIML' for single TE sequence or 'P2T2' for varied TE sequences. Default is P2T2
  --n_echoes N_ECHOES   Number of echoes
```



#### Configuration

Edit `config.yaml` to set various parameters like batch size, learning rate, epochs, etc., according to your computational resources and requirements.

#### Training

Run the model using:
```bash
python pt2_reconstruction_model_main.py --config config.yaml
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite the following paper:
```
Hadas Ben-Atya, Moti Freiman, "P2T2: A physically-primed deep-neural-network approach for robust T2 distribution estimation from quantitative T2-weighted MRI," Computerized Medical Imaging and Graphics, Volume 107, 2023, 102240, ISSN 0895-6111.
```

## Acknowledgements

The study was supported in part by research grants from the United States Israel Bi-national Science Foundation (BSF), the Israel Innovation Authority, the Israel Ministry of Science and Technology, and the Microsoft Israel and Israel Inter-University Computation Center program . We thank Thomas Yu, Erick Jorge Canales Rodriguez, Marco Pizzolato, Gian Franco Piredda, Tom Hilbert, Elda Fischi-Gomez, Matthias Weigel, Muhamed Barakovic, Meritxell Bach-Cuadra, Cristina Granziera, Tobias Kober, and Jean-Philippe Thiran, from [Yu et al. (2021)](https://doi.org/10.1016/j.media.2020.101940) for sharing their synthetic data generator with us. We also thank Prof. Noam Ben-Eliezer and the Lab for Advanced MRI at Tel-Aviv University for sharing the real MRI data with us.
