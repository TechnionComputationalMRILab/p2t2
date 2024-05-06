def main():
    from p2t2 import simulate
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Reconstruct T2 distribution from mri signal for brain data')
    parser.add_argument('--config_file', '-c', type=str, help='Path to config file', required=True)
    parser.add_argument('--out_folder', '-o', type=str, help='Path to output folder', required=True)

    parser.add_argument('--model_type', type=str, default='P2T2', help="Model type. 'MIML' for single TE sequence or 'P2T2' for varied TE sequences. Default is P2T2", choices=['P2T2', 'MIML'])
    parser.add_argument('--min_te', type=float, default=5.0, help='Minimum echo time. Default is 5.0')
    parser.add_argument('--max_te', type=float, default=15.0, help='Maximum echo time (only for P2T2 type). Default is 15.0')
    parser.add_argument('--n_echoes', type=int, default=20, help='Number of echoes (only for MIML type). Default is 20')
    parser.add_argument('--num_signals', type=int, default=10000, help='Number of signals (only for MIML type). Default is 10000')

    args = parser.parse_args()

    simulate(
        args.config_file,
        args.model_type,
        args.min_te,
        args.max_te,
        args.n_echoes,
        args.num_signals,
        args.out_folder
    )
