import torch
import numpy as np
import pandas as pd
import xarray as xr
from evaluation.helpers import plot_flexible_metrics, plot_comparison


class CubeEvaluator:
    def __init__(self, model, device, target_length, context_length):
        self.model = model.to(device).eval()
        self.device = device
        self.target_length = target_length
        self.context_length = context_length

    def run(self, loader, test_files, patches_per_cube, patch_plot=False):
        iter_loader = iter(loader)
        all_cube_results = []
        CUBE_RES = 1000

        for cube_url in test_files:
            cube_name = cube_url.split("/")[-1]
            print(f"\n🌍 EVALUATE CUBE: {cube_name}")

            # 1. Zarr Data (for Validation and visualization)
            ds_raw = self._load_zarr_raw(cube_url)

            # --- STITCHING SETUP ---
            # Shapes: [Target_Length, 1000, 1000]
            full_ctx = np.zeros((self.context_length, CUBE_RES, CUBE_RES))
            full_pred = np.zeros((self.target_length, CUBE_RES, CUBE_RES))
            full_true = np.zeros((self.target_length, CUBE_RES, CUBE_RES))
            full_mask = np.zeros((self.target_length, CUBE_RES, CUBE_RES))
            full_base = np.zeros((CUBE_RES, CUBE_RES))

            # Count Maps for mean calculation of overlapping pixels in patches
            counts_3d = np.zeros((self.target_length, CUBE_RES, CUBE_RES))
            counts_2d = np.zeros((CUBE_RES, CUBE_RES))

            patch_results = []
            for p in range(patches_per_cube):
                try:
                    batch = next(iter_loader)
                except StopIteration:
                    break

                # 2. Prediction & Data Extraction
                data = self._process_batch(batch)
                p_size = data["pred"].shape[-1]
                y_m, x_m = data["y_m"], data["x_m"]

                # 3. Consistency check
                self._verify_gt_consistency(data, ds_raw)

                # 4. Calculate metrics per patch
                metrics = self._calculate_metrics(data)

                # 5. Patch-Reporting
                self._print_patch_report(p, patches_per_cube, data, metrics)

                # 6. Plotting
                if patch_plot:
                    self._plot_patch(data, metrics, ds_raw)

                patch_results.append(metrics)

                # --- 7. Add to complete cube ---
                full_ctx[:, y_m : y_m + p_size, x_m : x_m + p_size] += data["ctx"]
                full_pred[:, y_m : y_m + p_size, x_m : x_m + p_size] += data["pred"]
                full_true[:, y_m : y_m + p_size, x_m : x_m + p_size] += data["true"]
                full_mask[:, y_m : y_m + p_size, x_m : x_m + p_size] += data["mask"]
                full_base[y_m : y_m + p_size, x_m : x_m + p_size] += data["base"]

                # Increase  counter
                counts_3d[:, y_m : y_m + p_size, x_m : x_m + p_size] += 1
                counts_2d[y_m : y_m + p_size, x_m : x_m + p_size] += 1

            # --- 8. Calculate Mean (Overlap-Handling) ---
            full_pred = np.divide(
                full_pred, counts_3d, out=np.zeros_like(full_pred), where=counts_3d > 0
            )
            full_true = np.divide(
                full_true, counts_3d, out=np.zeros_like(full_true), where=counts_3d > 0
            )
            full_mask = (full_mask > 0).astype(np.float32)
            full_base = np.divide(
                full_base, counts_2d, out=np.zeros_like(full_base), where=counts_2d > 0
            )

            # --- 9. cube-level metrics ---
            cube_data = {
                "ctx": full_ctx,
                "true": full_true,
                "pred": full_pred,
                "mask": full_mask,
                "base": full_base,
                "meta": {"path": [cube_url]},
            }
            cube_metrics = self._calculate_metrics(cube_data)

            # --- 6. Final report & plots ---
            self._print_cube_summary(cube_name, cube_metrics)
            self._plot_full_cube(cube_data, cube_metrics)

            all_cube_results.append(cube_metrics)

        return all_cube_results

    # --- Subfunctions ---

    def _load_zarr_raw(self, url):
        ds = xr.open_zarr(url)
        cutoff = pd.to_datetime(ds.attrs["precip_end_date"])
        return {
            "full": ds,
            "ctx": ds.sel(time_sentinel_2_l2a=slice(None, cutoff)).tail(
                time_sentinel_2_l2a=20
            ),
            "tgt": ds.sel(
                time_sentinel_2_l2a=slice(cutoff + pd.Timedelta(days=1), None)
            ).head(time_sentinel_2_l2a=self.target_length),
        }

    def _process_batch(self, batch):
        x_ctx, x_fut, y_true, mask, meta, baseline = batch
        with torch.no_grad():
            y_pred, _, _ = self.model(
                x_ctx.to(self.device),
                self.target_length,
                x_fut.to(self.device),
                baseline.to(self.device),
            )

        return {
            "ctx": x_ctx[0, :, 0].cpu().numpy(),  # kNDVI Context
            "true": y_true[0, :, 0].cpu().numpy(),  # GT
            "pred": y_pred[0, :, 0].cpu().numpy(),  # Prediction
            "mask": mask[0, :, 0].cpu().numpy(),  # Maske
            "base": baseline[0, 0].cpu().numpy(),  # Baseline
            "meta": meta,
            "y_m": int(meta["top"][0]),
            "x_m": int(meta["left"][0]),
        }

    def _verify_gt_consistency(self, data, ds_raw):
        """Verifies that dataloader GT aligns with zarr GT."""
        z_tgt = (
            ds_raw["tgt"]
            .kNDVI.isel(
                y=slice(data["y_m"], data["y_m"] + 256),
                x=slice(data["x_m"], data["x_m"] + 256),
            )
            .values
        )
        diff = np.abs(data["true"] - z_tgt)
        # Only check for valid pixels
        masked_diff = diff[data["mask"] == 1]
        if len(masked_diff) > 0 and np.max(masked_diff) > 1e-4:
            print(
                f"⚠️ WARNING: GT inconsistency found! Max Diff: {np.max(masked_diff):.6f}"
            )

    def _calculate_metrics(self, data):
        step_metrics = []
        for t in range(self.target_length):
            m = data["mask"][t] == 1
            num_valid = m.sum()

            if num_valid > 0:
                # MSE
                mse_pred = np.mean((data["true"][t][m] - data["pred"][t][m]) ** 2)
                mse_base = np.mean((data["true"][t][m] - data["base"][m]) ** 2)
                # MAE
                mae_pred = np.mean(np.abs(data["true"][t][m] - data["pred"][t][m]))
                mae_base = np.mean(np.abs(data["true"][t][m] - data["base"][m]))

                # Bias (Pred - True)
                # Positive: Model overestimates vegetation | Negative: Model underestimates
                bias_abs = np.mean(data["pred"][t][m] - data["true"][t][m])

                # Deltas
                d_true = data["true"][t][m] - data["base"][m]
                d_pred = data["pred"][t][m] - data["base"][m]
                mse_delta = np.mean((d_true - d_pred) ** 2)
                bias_delta = np.mean(d_pred - d_true)
            else:
                mse_pred = mse_base = mae_pred = mae_base = bias_abs = 0.0
                mse_delta = bias_delta = 0.0
                d_true = d_pred = np.array([0])

            step_metrics.append(
                {
                    "step": t + 1,
                    "num_valid": num_valid,
                    "mse_pred": mse_pred,
                    "mse_base": mse_base,
                    "mae_pred": mae_pred,
                    "mae_base": mae_base,
                    "bias_abs": bias_abs,
                    "mse_delta": mse_delta,
                    "bias_delta": bias_delta,
                    "d_true_mean": d_true.mean(),
                    "d_true_std": d_true.std(),
                    "d_pred_mean": d_pred.mean(),
                    "d_pred_std": d_pred.std(),
                }
            )
        return step_metrics

    def _plot_patch(self, data, metrics, ds_raw):
        data["true_ctx"] = data["ctx"]
        plot_comparison(data, metrics)
        plot_flexible_metrics(metrics, ["mse_pred", "mse_base"])
        plot_flexible_metrics(metrics, ["bias_abs", "bias_delta"])
        plot_flexible_metrics(metrics, ["d_true_mean", "d_pred_mean"])

    def _plot_full_cube(self, data, metrics):
        data["true_ctx"] = data["ctx"]
        plot_comparison(data, metrics)
        plot_flexible_metrics(metrics, ["mse_pred", "mse_base"])
        plot_flexible_metrics(metrics, ["bias_abs", "bias_delta"])
        plot_flexible_metrics(metrics, ["d_true_mean", "d_pred_mean"])

    def _print_stats_report(self, name, metrics, p_size, is_cube=False):
        """
        Function for patch and cube reporting.
        """
        GREEN, RED, BOLD, RESET = "\033[92m", "\033[91m", "\033[1m", "\033[0m"

        label = "🌍 CUBE SUMMARY" if is_cube else f"🧩 Patch {name}"
        print(f"\n{BOLD}{'='*30}{RESET}")
        print(f"{BOLD}{label} | Size={p_size}{RESET}")
        print(f"{BOLD}{'='*30}{RESET}")

        header = f"{'Step':<7} | {'MSE Pred':<10} | {'MSE Base':<10} | {'MAE Pred':<10} | {'Δ True':<9} | {'Δ Pred':<9} | {'Δ Bias':<9} | {'Valid'}"
        print(header)
        print("-" * len(header))

        total_valid_pixels = 0
        valid_mses_p = []
        valid_mses_b = []
        valid_biases = []

        for m in metrics:
            t, num_v = m["step"], int(m["num_valid"])
            total_valid_pixels += num_v
            pct = (num_v / (p_size**2)) * 100

            if num_v > 0:
                is_better = m["mse_pred"] < m["mse_base"]
                mark = f"{GREEN}✔{RESET}" if is_better else f"{RED}✘{RESET}"
                color = GREEN if is_better else RED
                valid_mses_p.append(m["mse_pred"])
                valid_mses_b.append(m["mse_base"])
                valid_biases.append(m["bias_delta"])
            else:
                mark, color = " ", RESET

            print(
                f"T+{t:<4} | {color}{m['mse_pred']:<10.6f}{RESET} | {m['mse_base']:<10.6f} | "
                f"{m['mae_pred']:<10.5f} | {m['d_true_mean']:>9.4f} | {m['d_pred_mean']:>9.4f} | "
                f"{m['bias_delta']:>9.4f} | {num_v:>7} ({pct:>3.1f}%) {mark}"
            )

        # Aggregation
        avg_p = np.mean(valid_mses_p) if valid_mses_p else 0.0
        avg_b = np.mean(valid_mses_b) if valid_mses_b else 0.0
        avg_bias = np.mean(valid_biases) if valid_biases else 0.0

        total_pct = (total_valid_pixels / ((p_size**2) * len(metrics))) * 100

        if total_valid_pixels > 0:
            total_better = avg_p < avg_b
            res_msg = (
                f"{GREEN}✅ MODEL WINS{RESET}"
                if total_better
                else f"{RED}❌ BASELINE WINS{RESET}"
            )
            sum_color = GREEN if total_better else RED
        else:
            res_msg, sum_color = "⚪ NO DATA", RESET

        print("-" * len(header))
        print(
            f"{BOLD}OVERALL | {sum_color}{avg_p:<10.6f}{RESET} | {avg_b:<10.6f} | {BOLD}Avg Bias: {avg_bias:.5f}{RESET}"
        )
        print(
            f"{BOLD}COVERAGE| {total_valid_pixels:,} valid pixels total ({total_pct:.2f}% of area){RESET}"
        )
        print(f"{BOLD}RESULT  | {res_msg}{RESET}")
        print("-" * len(header) + "\n")

    def _print_patch_report(self, p, total, data, metrics):
        p_size = data["ctx"].shape[-1]
        self._print_stats_report(f"{p+1}/{total}", metrics, p_size, is_cube=False)

    def _print_cube_summary(self, cube_name, metrics):
        full_res = 1000
        self._print_stats_report(cube_name, metrics, full_res, is_cube=True)
