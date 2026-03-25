import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(data, metrics):
    """
    Visualizes comparison between GT and preds and some more metrics
    Column 1-3: Last 3 Context Images
    Column 4-8: Target Timesteps 1-5
    """
    # Data extraction
    ctx, true, pred = data["true_ctx"], data["true"], data["pred"]
    mask, baseline = data["mask"], data["base"]

    is_full_cube = data["true_ctx"].shape[-1] > 500
    type_label = "FULL CUBE" if is_full_cube else "PATCH"
    cube_id = data["meta"]["path"][0].split("/")[-1].replace(".zarr", "")

    if is_full_cube:  # It's a full cube (1000x1000)
        base_size_w = 28
        base_size_h = 24
        title_font = 22
    else:  # it's a patch
        base_size_w = 20
        base_size_h = 16
        title_font = 16

    # Grid Setup (8 columns and x rows)
    rows = [
        "GT & Context",
        "Prediction",
        "True Delta",
        "Pred Delta",
        "MSE Error",
        "MAE Error",
        "Mask",
    ]
    fig, axes = plt.subplots(len(rows), 8, figsize=(base_size_w, base_size_h), dpi=100)
    fig.patch.set_facecolor("white")

    for i in range(8):
        # --- COLUMN LOGIC ---
        if i < 3:  # Last 3 context steps
            t_idx = -3 + i
            # Line 0: Last context steps
            axes[0, i].imshow(ctx[t_idx], vmin=0, vmax=0.8, cmap="Greens")
            axes[0, i].set_title(f"Ctx {t_idx}")

            # Baseline to row 2, column 3
            if i == 2:
                axes[1, i].imshow(baseline, vmin=0, vmax=0.8, cmap="Greens")
                axes[1, i].set_title("Baseline (T=40)")

            start_row = 2 if i == 2 else 1
            for r in range(start_row, len(rows)):
                axes[r, i].axis("off")

        else:  # Target steps (Column 4 to 8)
            t = i - 3  # Index 0 to 4
            m = metrics[t]
            m_step = mask[t] == 1
            curr_mask = mask[t] == 0

            # Data for this timestep
            y_t = np.ma.masked_where(curr_mask, true[t])
            d_t = np.ma.masked_where(curr_mask, true[t] - baseline)
            # y_p = np.ma.masked_where(curr_mask, pred[t])
            # d_p = np.ma.masked_where(curr_mask, pred[t] - baseline)
            err = np.ma.masked_where(curr_mask, true[t] - pred[t])

            # Prediction should not be masked
            y_p = pred[t]
            d_p = pred[t] - baseline

            # Row 0: Ground Truth
            axes[0, i].imshow(y_t, vmin=0, vmax=0.8, cmap="Greens")
            axes[0, i].set_title(f"GT T+{t+1}")

            # Row 1: Prediction (mean value)
            axes[1, i].imshow(y_p, vmin=0, vmax=0.8, cmap="Greens")
            mean_p = y_p.mean() if m_step.any() else 0
            axes[1, i].set_title(f"Pred (μ={mean_p:.3f})")

            # Row 2: True Delta
            axes[2, i].imshow(d_t, vmin=-0.1, vmax=0.1, cmap="seismic")
            mean_dt = d_t.mean() if m_step.any() else 0
            axes[2, i].set_title(f"True Δ (μ={mean_dt:.4f})")

            # Row 3: Predicted Delta
            axes[3, i].imshow(d_p, vmin=-0.1, vmax=0.1, cmap="seismic")
            mean_dp = d_p.mean() if m_step.any() else 0
            axes[3, i].set_title(f"Pred Δ (μ={mean_dp:.4f})")

            # Row 4: MSE Map
            axes[4, i].imshow(err**2, vmin=0, vmax=0.05, cmap="Reds")
            axes[4, i].set_title(f"MSE: {m['mse_pred']:.4f}")

            # Row 5: MAE Map (Absolute Error)
            axes[5, i].imshow(np.abs(err), vmin=0, vmax=0.2, cmap="YlOrRd")
            axes[5, i].set_title(f"MAE: {m['mae_pred']:.4f}")

            # Row 6: Mask
            axes[6, i].imshow(mask[t], cmap="gray")
            axes[6, i].set_title(f"Valid: {m['num_valid']}")

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if is_full_cube:
        plt.suptitle(f"{type_label} Evaluation | {cube_id}", fontsize=title_font)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        import os

        save_dir = "evaluation_plots"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{save_dir}/{type_label}_{cube_id}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"💾 Plot saved under: {filename}")
    else:
        plt.suptitle(
            f"{type_label} Evaluation | {cube_id} | Pos: y={data['y_m']}, x={data['x_m']}",
            fontsize=title_font,
        )
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    plt.show()
    plt.close(fig)


def plot_flexible_metrics(metrics, *var_groups, title="Temporal Analysis"):
    if not metrics:
        print("No metrics found")
        return

    steps = [m["step"] for m in metrics]
    n_plots = len(var_groups)

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)
    axes = axes.flatten()

    for i, group in enumerate(var_groups):
        vars_to_plot = [group] if isinstance(group, str) else group

        for var in vars_to_plot:
            if var in metrics[0]:
                # Logik: Wenn num_valid == 0, setze den Wert auf np.nan
                data_points = [
                    m[var] if m.get("num_valid", 0) > 0 else np.nan for m in metrics
                ]
                axes[i].plot(
                    steps, data_points, marker="o", markersize=6, linewidth=2, label=var
                )
            else:
                print(f"Warning: Variable '{var}' not found.")

        axes[i].set_title(f"Verlauf: {', '.join(vars_to_plot)}")
        axes[i].set_xlabel("Timestep")
        axes[i].set_xticks(steps)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        if any(kw in str(vars_to_plot).lower() for kw in ["bias", "delta"]):
            axes[i].axhline(0, color="black", linestyle="-", alpha=0.4)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
