# Robust Reconstruction of p(T2) from Multi-Echo T2 MRI Data

This repository contains the implementation of the methods described in the paper by Hadas Ben-Atya and Moti Freiman:

**"P2T2: A physically-primed deep-neural-network approach for robust T2 distribution estimation from quantitative T2-weighted MRI,"** Computerized Medical Imaging and Graphics, Volume 107, 2023, 102240, ISSN 0895-6111.

[Read the paper](https://www.sciencedirect.com/science/article/pii/S0895611123000587)

## Overview

This project focuses on the robust estimation of T2 distributions from quantitative T2-weighted MRI data using deep learning approaches described in the P2T2 and MIML papers. The repository includes scripts for simulating Echo Planar Graphs (EPGs) and training a model to reconstruct T2 distributions.

### Prerequisites

Before running the simulations and the model, ensure you have the following installed:
- Python 3.8 or later
- NumPy
- PyTorch
- Any other dependencies listed in `requirements.txt`

Install the required packages using:
```bash
pip install -r requirements.txt
```

### Data Simulation

Use the script `data_simulation.py` to simulate MRI data. You can specify the type of model (`MIML` or `P2T2`) to determine the simulation parameters.

#### Usage
```bash
python data_simulation.py --model_type <MODEL_TYPE> --min_TE <MIN_TE> --max_TE <MAX_TE> --n_echoes <N_ECHOES>
```

- `MODEL_TYPE`: Type of the model ('MIML' for single TE sequence or 'P2T2' for varied TE sequences).
- `MIN_TE`: Minimum echo time in milliseconds.
- `MAX_TE`: Maximum echo time in milliseconds (only for 'P2T2' type).
- `N_ECHOES`: Number of echo times (applies to 'MIML').

### Model Training

To train the model, configure the settings in `config.yaml` and run `pt2_reconstruction_model_main.py`.

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