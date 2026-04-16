import argparse
import os
import yaml
import multiprocessing as mp
from model_manager import ARCEMEPipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARCEME ConvLSTM Model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )

    # Intelligente Checkpoint-Auswahl
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Directory of the run (e.g., wand_db_logs/run_2026...)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Evaluate a specific fold. If None, finds the overall best model across all folds.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Load 'best' or 'last' checkpoint of the fold.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Explicit path to a .ckpt file (overrides run_dir logic)",
    )

    # Optional argument
    # do this to pass specific files for evaluation instead of the hold out test set
    parser.add_argument(
        "--test_list",
        type=str,
        default=None,
        help="Path to a txt or csv file containing absolute paths to .zarr files (one per line).",
    )

    args = parser.parse_args()

    # Load the config save in run_dir to ensure evaluation uses the same parameters as training
    run_config_path = os.path.join(args.run_dir, "config_used.yaml")
    config_to_load = run_config_path if os.path.exists(run_config_path) else args.config

    # Load Config
    with open(config_to_load, "r") as f:
        cfg = yaml.safe_load(f)

    # Initialize Pipeline in eval mode
    pipeline = ARCEMEPipeline(config=cfg, mode="eval", run_dir=args.run_dir)

    # Dynamically get checkoint
    ckpt_to_load = args.ckpt
    # if no specific checkpoint provided, find in run_dir or fold
    if not ckpt_to_load:
        # if fold specified, look for best or last checkpoint in that fold, otherwise look for overall best checkpoint across all folds
        if args.fold is not None:
            print(f"🔍 Looking for {args.type} checkpoint in Fold {args.fold}...")
            ckpt_to_load = pipeline.get_checkpoint_path(args.fold, type=args.type)
        else:
            print(
                f"🔍 Looking for the OVERALL BEST checkpoint across all folds in {args.run_dir}..."
            )
            ckpt_to_load = pipeline.get_best_overall_checkpoint()

    if not ckpt_to_load or not os.path.exists(ckpt_to_load):
        raise FileNotFoundError(
            f"❌ Could not find a valid checkpoint at: {ckpt_to_load}"
        )

    # Load custom test files if provided
    custom_test_files = None
    if args.test_list:
        if not os.path.exists(args.test_list):
            raise FileNotFoundError(f"❌ Test list file not found: {args.test_list}")

        print(f"📄 Reading custom evaluation paths from: {args.test_list}")
        # Reads .txt file
        with open(args.test_list, "r") as f:
            # Filter lines that end with .zarr and strip whitespace
            custom_test_files = [
                line.strip() for line in f.readlines() if line.strip().endswith(".zarr")
            ]

        if not custom_test_files:
            raise ValueError("❌ No valid .zarr paths found in the provided test_list!")
        print(f"✅ Loaded {len(custom_test_files)} paths directly from file.")

    # Start Evaluation
    print(f"Starting Evaluation using checkpoint: {ckpt_to_load}")
    # If custom_test_files is None, looks automatically in cfg["data"]["test_data_dir"]
    results = pipeline.evaluate(ckpt_path=ckpt_to_load, test_files=custom_test_files)

    print("\n✅ Evaluation Finished!")
    print(results)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

# In terminal:
# Evaluate specific fold and checkpoint type (best or last):
#       - python evaluate.py --run_dir path/to/run_dir --fold X --type best
#       - python evaluate.py --run_dir path/to/run_dir --fold X --type last
# Evaluate overall best checkpoint across all folds:
#       - python evaluate.py --run_dir path/to/run_dir
# Evaluate specific checkpoint directly:
#       - python evaluate.py --ckpt path/to/specific_checkpoint.ckpt
# Evaluate on a different test set (create a txt file with absolut paths - get from cv splits for example):
#       - python evaluate.py --run_dir wand_db_logs/run_Dein_Run --fold 2 --type best --test_list val_fold_2.txt
