def main():
    from p2t2 import infer
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Reconstruct T2 distribution from mri signal for brain data')
    parser.add_argument('--model_path', '-m', type=str, help='Path to model', required=True)
    parser.add_argument('--model_args_path', '-a', type=str, help='Path to model args', required=True)
    parser.add_argument('--output_dir', '-o', type=str, help='Output directory', required=True)
    parser.add_argument('--mri', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True)

    parser.add_argument('--model_type', type=str, default='P2T2', help="Model type. 'MIML' for single TE sequence or 'P2T2' for varied TE sequences. Default is P2T2", choices=['P2T2', 'MIML'])
    parser.add_argument('--n_echoes', type=int, help='Number of echoes')

    args = parser.parse_args()

    infer(
        model_type=args.model_type,
        model_path=args.model_path,
        model_args_path=args.model_args_path,
        data_dict={
            'mri_path': args.mri_path,
            'metadata_path': args.metadata_path
        },
        output_dir=args.output_dir,
        n_echoes=args.n_echoes
    )
