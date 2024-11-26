def run(cfg: DictConfig) -> None:
    """
    Generic train loop without PyTorch Lightning and WandB.

    :param cfg: run configuration, defined by Hydra in /conf
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    # Set deterministic behavior
    if cfg.train.deterministic:
        torch.manual_seed(cfg.train.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger-friendly configuration!"
        )
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate dataset and dataloaders
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    train_loader = DataLoader(datamodule.train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(datamodule.val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(datamodule.test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.to(cfg.train.device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr)
    criterion = nn.MSELoss()  # Example: Replace with your loss function

    # Function to train for one epoch
    def train_one_epoch(loader):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            inputs, targets = batch  # Assuming datasets return (inputs, targets)
            inputs, targets = inputs.to(cfg.train.device), targets.to(cfg.train.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    # Function to validate
    def validate(loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                inputs, targets = batch
                inputs, targets = inputs.to(cfg.train.device), targets.to(cfg.train.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(loader)

    # Training loop
    hydra.utils.log.info("Starting training!")
    best_val_loss = float('inf')
    for epoch in range(cfg.train.epochs):
        train_loss = train_one_epoch(train_loader)
        val_loss = validate(val_loader)

        hydra.utils.log.info(f"Epoch {epoch + 1}/{cfg.train.epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), hydra_dir / "best_model.pth")

    # Testing loop
    hydra.utils.log.info("Starting testing!")
    test_loss = validate(test_loader)
    hydra.utils.log.info(f"Test Loss = {test_loss:.4f}")

    # Save scalers for reproducibility
    hydra.utils.log.info(f"Saving scalers to {hydra_dir}")
    torch.save(datamodule.lattice_scaler, hydra_dir / "lattice_scaler.pt")
    torch.save(datamodule.scaler, hydra_dir / "prop_scaler.pt")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)
