def main():
    from p2t2 import train
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Reconstruct T2 distribution from mri signal for brain data')

    parser.add_argument('--config_file', "-c", type=str, help='Path to config file', required=True)
    parser.add_argument('--data_folder', "-d", type=str, help='Path to data folder', required=True)
    parser.add_argument('--output_path', "-o", type=str, help='Path to output folder', required=True)
    parser.add_argument('--model_type', type=str, default='P2T2', help="Model type. 'MIML' for single TE sequence or 'P2T2' for varied TE sequences. Default is P2T2", choices=['P2T2', 'MIML'])
    parser.add_argument('--min_te', type=float, default=7.9, help='Minimum echo time. Default is 7.9')
    parser.add_argument('--max_te', type=float, help='Maximum echo time. Optional')

    args = parser.parse_args()

    train(
        config=args.config_file,
        data_folder=args.data_folder,
        output_path=args.output_path,
        model_type=args.model_type,
        min_te=args.min_te,
        max_te=args.max_te
    )
