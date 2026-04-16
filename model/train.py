import argparse
import yaml
import multiprocessing as mp
from model_manager import ARCEMEPipeline


def main():
    parser = argparse.ArgumentParser(description="Train ARCEME ConvLSTM Model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )

    # Resume Arguments
    parser.add_argument(
        "--resume_run",
        type=str,
        default=None,
        help="Path to an existing run directory (e.g. wand_db_logs/run_2026-04-10_...) to resume training.",
    )
    parser.add_argument(
        "--resume_fold",
        type=int,
        default=0,
        help="Which fold to start/resume from. Folds before this will be skipped.",
    )
    parser.add_argument(
        "--resume_type",
        type=str,
        default="last",
        choices=["last", "best"],
        help="Whether to resume from 'last.ckpt' or the 'best' checkpoint.",
    )

    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Initialize Pipeline
    pipeline = ARCEMEPipeline(config=cfg, mode="train", run_dir=args.resume_run)

    # Start Training
    pipeline.run_cv(start_fold=args.resume_fold, resume_from_type=args.resume_type)


if __name__ == "__main__":
    # Required for PyTorch multiprocessing with DataLoader workers
    mp.set_start_method("spawn", force=True)
    main()


# In terminal:
# check what GPU is free: ndvidia-smi  (zero indexed)
# Start model from zero: (not passing --resume_run or --resume_fold)
#       - CUDA_VISIBLE_DEVICES=X  python train.py
# Start model after it crashed in fold X: either use last or best checkpoint (depending on what you want)
#       - CUDA_VISIBLE_DEVICES=X  python train.py --resume_run path/to/run_dir --resume_fold X --resume-type last
