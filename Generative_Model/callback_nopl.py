def build_callbacks(cfg: DictConfig):
    """
    Build a list of custom callbacks without PyTorch Lightning.

    :param cfg: Configuration dictionary
    :return: A dictionary of callbacks
    """
    callbacks = {}

    # Learning Rate Monitor
    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateLogger>")
        
        class LearningRateLogger:
            def __init__(self, logging_interval="epoch"):
                self.logging_interval = logging_interval

            def log(self, optimizer, epoch=None):
                lr = optimizer.param_groups[0]["lr"]
                log_msg = f"Learning Rate: {lr:.6f}"
                if epoch is not None:
                    log_msg = f"Epoch {epoch}: {log_msg}"
                print(log_msg)

        callbacks["lr_logger"] = LearningRateLogger(logging_interval=cfg.logging.lr_monitor.logging_interval)

    # Early Stopping
    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        
        class EarlyStopping:
            def __init__(self, monitor="val_loss", mode="min", patience=10, verbose=True):
                self.monitor = monitor
                self.mode = mode
                self.patience = patience
                self.verbose = verbose
                self.best_score = None
                self.wait = 0
                self.stopped_epoch = 0

            def check(self, current_score, epoch):
                if self.best_score is None or (
                    (self.mode == "min" and current_score < self.best_score)
                    or (self.mode == "max" and current_score > self.best_score)
                ):
                    self.best_score = current_score
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        return True
                return False

        callbacks["early_stopping"] = EarlyStopping(
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            patience=cfg.train.early_stopping.patience,
            verbose=cfg.train.early_stopping.verbose,
        )

    # Model Checkpointing
    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        
        class ModelCheckpoint:
            def __init__(self, dirpath, monitor="val_loss", mode="min", save_top_k=1, verbose=True):
                self.dirpath = Path(dirpath)
                self.monitor = monitor
                self.mode = mode
                self.save_top_k = save_top_k
                self.verbose = verbose
                self.best_scores = []

            def save_checkpoint(self, model, score, epoch):
                file_path = self.dirpath / f"epoch={epoch}-score={score:.4f}.pth"
                torch.save(model.state_dict(), file_path)
                if self.verbose:
                    print(f"Saved checkpoint: {file_path}")
                
                # Manage top K checkpoints
                self.best_scores.append((score, file_path))
                self.best_scores.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
                if len(self.best_scores) > self.save_top_k:
                    _, removed_path = self.best_scores.pop()
                    removed_path.unlink(missing_ok=True)
                    if self.verbose:
                        print(f"Removed old checkpoint: {removed_path}")

        callbacks["model_checkpoint"] = ModelCheckpoint(
            dirpath=HydraConfig.get().run.dir,
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            save_top_k=cfg.train.model_checkpoints.save_top_k,
            verbose=cfg.train.model_checkpoints.verbose,
        )

    return callbacks



# Example:
callbacks = build_callbacks(cfg)

for epoch in range(cfg.train.epochs):
    train_loss = train_one_epoch(train_loader)
    val_loss = validate(val_loader)

    # Log Learning Rate
    if "lr_logger" in callbacks:
        callbacks["lr_logger"].log(optimizer, epoch)

    # Early Stopping
    if "early_stopping" in callbacks:
        if callbacks["early_stopping"].check(val_loss, epoch):
            break

    # Save Model Checkpoints
    if "model_checkpoint" in callbacks:
        callbacks["model_checkpoint"].save_checkpoint(model, val_loss, epoch)