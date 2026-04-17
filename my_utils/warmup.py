import pytorch_lightning as pl


class ConfigWarmupCallback(pl.Callback):
    """
    Executes linear warmup of the learning rate based on the config.
    Directly manipulates the PyTorch optimizer to avoid scheduler conflicts.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Extract values from config
        opt_cfg = self.cfg["training"]["optimizer"]
        warmup_cfg = opt_cfg.get("warmup", {})

        self.warmup_epochs = warmup_cfg.get("epochs", 5)

        # Get target_lr (e.g., 0.0003)
        self.target_lr = opt_cfg["start_learn_rate"]

        # Start at 10% of target lr
        self.start_lr = self.target_lr / 10.0

    def on_train_epoch_start(self, trainer, pl_module):
        # If less than 1 warmup epoch, skip warmup logic
        if self.warmup_epochs <= 0:
            return

        epoch = trainer.current_epoch

        # Check if still in warump phase
        if epoch < self.warmup_epochs:
            # Linear Increase of LR: start_lr -> target_lr over warmup_epochs
            lr = self.start_lr + (self.target_lr - self.start_lr) * (
                epoch / self.warmup_epochs
            )

            # Directly set the LR in the optimizer (bypassing any scheduler)
            for param_group in trainer.optimizers[0].param_groups:
                param_group["lr"] = lr

            print(
                f"\n🔥 Warmup (Epoch {epoch}/{self.warmup_epochs}): LR set to {lr:.6f}."
            )

        # At first normal epoch, print message about warmup end
        elif epoch == self.warmup_epochs:
            print(
                f"\n🚀 Warmup ended! Starting normal Training with target LR {self.target_lr:.6f}."
            )
            print("From here on, the Plateau Scheduler takes over.")
