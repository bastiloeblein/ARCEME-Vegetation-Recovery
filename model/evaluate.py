"""Run inference for a trained ARCEME model checkpoint."""

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

    # Checkpoint selection
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

    # Optional list of cubes, e.g. one cross-validation validation fold.
    parser.add_argument(
        "--test_list",
        type=str,
        default=None,
        help="Path to a txt or csv file containing absolute paths to .zarr files (one per line).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional separate directory for tensors, metrics and plots.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip full-cube qualitative plots (recommended for OOF export).",
    )

    args = parser.parse_args()

    # Prefer the configuration stored with the training run.
    run_config_path = os.path.join(args.run_dir, "config_used.yaml")
    config_to_load = run_config_path if os.path.exists(run_config_path) else args.config

    with open(config_to_load, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline = ARCEMEPipeline(config=cfg, mode="eval", run_dir=args.run_dir)

    # Resolve the requested checkpoint.
    ckpt_to_load = args.ckpt
    if not ckpt_to_load:
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

    custom_test_files = None
    if args.test_list:
        if not os.path.exists(args.test_list):
            raise FileNotFoundError(f"❌ Test list file not found: {args.test_list}")

        print(f"📄 Reading custom evaluation paths from: {args.test_list}")
        with open(args.test_list, "r") as f:
            custom_test_files = [
                line.strip() for line in f.readlines() if line.strip().endswith(".zarr")
            ]

        if not custom_test_files:
            raise ValueError("❌ No valid .zarr paths found in the provided test_list!")
        print(f"✅ Loaded {len(custom_test_files)} paths directly from file.")

    print(f"Starting Evaluation using checkpoint: {ckpt_to_load}")
    results = pipeline.evaluate(
        ckpt_path=ckpt_to_load,
        test_files=custom_test_files,
        output_dir=args.output_dir,
        plot_samples=not args.no_plots,
    )

    print("\n✅ Evaluation Finished!")
    print(results)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()


# python evaluate.py --run_dir  wand_db_logs/Ablation_SGConvLSTM_big_model --fold 0 --type best --test_list  wand_db_logs/Ablation_SGConvLSTM_big_model/custom_test_list.txt
