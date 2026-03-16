#!/usr/bin/env python3
"""
Modular Training Script for Neural Networks
Supports multiple datasets and model architectures
"""

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Disable albumentations update check

import torch
import torch.optim as optim
from torchinfo import summary

import yaml
from src.datasets import DatasetRegistry
from src.models import ModelRegistry
from src.training import Trainer, CheckpointManager, ExperimentManager
from src.config import create_config_parser, load_config
from src.utils import get_device, get_optimal_workers, setup_logger


def get_layer_id_mapping(model):
    """Generate a mapping from torchinfo-style layer IDs to module names.

    Returns a dict mapping "depth-idx" strings to module names.
    """
    id_to_name = {}
    depth_counters = {}

    def traverse_modules(module, prefix="", depth=1):
        if depth not in depth_counters:
            depth_counters[depth] = 0

        for name, child in module.named_children():
            depth_counters[depth] += 1
            idx = depth_counters[depth]
            layer_id = f"{depth}-{idx}"
            full_name = f"{prefix}.{name}" if prefix else name
            id_to_name[layer_id] = full_name

            # Recurse into children
            traverse_modules(child, full_name, depth + 1)

    traverse_modules(model)
    return id_to_name


def parse_freeze_spec(spec):
    """Parse a freeze specification.

    Supports:
    - Single layer ID: "2-1"
    - Range: "2-1:2-5" (freeze layers 2-1 through 2-5)
    - Name pattern: "features" (starts with)
    """
    if ":" in spec:
        # Range specification
        start, end = spec.split(":")
        return ("range", start.strip(), end.strip())
    elif "-" in spec and spec.split("-")[0].isdigit():
        # Single layer ID
        return ("id", spec.strip())
    else:
        # Name pattern
        return ("name", spec.strip())


def freeze_layers(model, freeze_specs, id_to_name=None, logger=None):
    """Freeze model layers based on specifications.

    Args:
        model: The neural network model
        freeze_specs: List of freeze specifications (layer IDs, ranges, or name patterns)
        id_to_name: Optional mapping from layer IDs to module names
        logger: Optional logger for warnings

    Returns:
        Tuple of (frozen_count, modules_to_freeze)
    """

    def warn(msg):
        if logger:
            logger.warning(msg)
        else:
            print(f"Warning: {msg}")

    if not freeze_specs:
        return 0, set()

    # Build ID mapping if not provided
    if id_to_name is None:
        id_to_name = get_layer_id_mapping(model)

    # Collect all module names to freeze
    modules_to_freeze = set()

    for spec in freeze_specs:
        spec_type, *rest = parse_freeze_spec(spec)

        if spec_type == "id":
            # Single layer ID
            layer_id = rest[0]
            if layer_id in id_to_name:
                modules_to_freeze.add(id_to_name[layer_id])
            else:
                warn(f"Layer ID '{layer_id}' not found")

        elif spec_type == "range":
            # Range of layer IDs (same depth)
            start_id, end_id = rest
            start_depth, start_idx = map(int, start_id.split("-"))
            end_depth, end_idx = map(int, end_id.split("-"))

            if start_depth != end_depth:
                warn(
                    f"Range {start_id}:{end_id} has different depths, using start depth"
                )

            depth = start_depth
            for idx in range(start_idx, end_idx + 1):
                layer_id = f"{depth}-{idx}"
                if layer_id in id_to_name:
                    modules_to_freeze.add(id_to_name[layer_id])

        elif spec_type == "name":
            # Name pattern
            pattern = rest[0]
            for layer_id, module_name in id_to_name.items():
                if module_name.startswith(pattern):
                    modules_to_freeze.add(module_name)

    # Freeze parameters in the selected modules
    frozen_count = 0
    frozen_modules = set()

    for name, param in model.named_parameters():
        for module_name in modules_to_freeze:
            if name.startswith(module_name):
                param.requires_grad = False
                frozen_count += 1
                frozen_modules.add(module_name)
                break

    return frozen_count, modules_to_freeze


def create_optimizer(model, config):
    """Create optimizer based on configuration."""
    import muon

    optimizer_name = config.optimization.get("optimizer", "adamw").lower()
    learning_rate = config.optimization.get("learning_rate", 1e-3)
    weight_decay = config.optimization.get("weight_decay", 0.01)

    # Only pass trainable parameters to optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "adamw":
        return optim.AdamW(
            trainable_params, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        return optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = config.optimization.get("momentum", 0.9)
        return optim.SGD(
            trainable_params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "muon":
        muon_lr = config.optimization.get("learning_rate", 1e-3)
        muon_momentum = config.optimization.get("muon_momentum", 0.95)
        muon_params = [p for p in trainable_params if p.ndim >= 2]
        return muon.SingleDeviceMuon(
            muon_params,
            lr=muon_lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
        )
    elif optimizer_name == "muon_with_aux_adam":
        muon_lr = config.optimization.get("learning_rate", 1e-3)
        muon_momentum = config.optimization.get("muon_momentum", 0.95)
        adam_lr = config.optimization.get("adam_lr", 3e-4)
        adam_betas = tuple(config.optimization.get("adam_betas", [0.9, 0.95]))

        hidden_weights = [p for p in trainable_params if p.ndim >= 2]
        nonhidden_params = [p for p in trainable_params if p.ndim < 2]

        param_groups = [
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=weight_decay,
            ),
            dict(
                params=nonhidden_params,
                use_muon=False,
                lr=adam_lr,
                betas=adam_betas,
                weight_decay=weight_decay,
            ),
        ]
        return muon.SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on configuration."""
    scheduler_name = config.optimization.get("scheduler", "none").lower()

    if scheduler_name == "none":
        return None
    elif scheduler_name == "step":
        step_size = config.optimization.get("scheduler_step_size", 10)
        gamma = config.optimization.get("scheduler_gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "cosine":
        t_max = config.optimization.get("scheduler_t_max", 100)
        eta_min = config.optimization.get("scheduler_eta_min", 1e-6)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    elif scheduler_name == "plateau":
        patience = config.optimization.get("scheduler_patience", 5)
        factor = config.optimization.get("scheduler_factor", 0.1)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=patience, factor=factor
        )
    elif scheduler_name == "exponential":
        gamma = config.optimization.get("scheduler_gamma", 0.95)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def main():
    """Main training function."""
    # Parse arguments
    parser = create_config_parser()
    args = parser.parse_args()

    # Load configuration
    config = load_config(config_path=args.config, args=args)

    # Set up logging
    logger = setup_logger()
    logger.info("Starting training")

    # Get device
    device, using_cpu = get_device()
    logger.info(f"Using device: {device}")

    # Get optimal workers
    train_workers, val_workers = get_optimal_workers(using_cpu)
    logger.info(
        f"Using {train_workers} workers for training, {val_workers} for validation"
    )

    # Validate arguments
    if args.resume and args.fork:
        raise ValueError("Cannot use both --resume and --fork at the same time")

    # Create experiment manager (YOLO-style runs/expX)
    experiment_manager = ExperimentManager(
        base_dir="runs", resume_exp=args.resume, fork_exp=args.fork
    )
    logger.info(experiment_manager.get_experiment_info())

    # Check if resuming or forking
    resume_checkpoint = None
    start_epoch = 0
    source_checkpoints_dir = experiment_manager.checkpoints_dir

    if args.fork:
        # Fork: load from source experiment but save to new one
        assert experiment_manager.fork_source_dir is not None, "fork_source_dir should be set when forking"

        source_checkpoints_dir = os.path.join(
            experiment_manager.fork_source_dir, "checkpoints"
        )
        checkpoint_path = os.path.join(source_checkpoints_dir, "latest_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            logger.info(f"Forking from checkpoint: {checkpoint_path}")
            resume_checkpoint = checkpoint_path
            # Copy config.yaml from source to new experiment
            source_config = os.path.join(
                experiment_manager.fork_source_dir, "config.yaml"
            )
            if os.path.exists(source_config):
                import shutil

                shutil.copy(source_config, experiment_manager.config_file)
                logger.info(f"Copied config from {source_config}")
        else:
            logger.warning(
                f"No checkpoint found at {checkpoint_path}, starting from scratch"
            )
    elif args.resume:
        checkpoint_path = os.path.join(
            experiment_manager.checkpoints_dir, "latest_checkpoint.pt"
        )
        if os.path.exists(checkpoint_path):
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            resume_checkpoint = checkpoint_path
        else:
            logger.warning(
                f"No checkpoint found at {checkpoint_path}, starting from scratch"
            )

    # Save config to experiment directory (if not resuming)
    # Note: Config is saved after dataset creation to get actual values

    # Get input_size from config (model's requirement)
    input_size_config = config.model.get("input_size", [64, 64])
    if isinstance(input_size_config, list):
        input_size = tuple(input_size_config)
    else:
        input_size = input_size_config

    # Create dataset - adapts to model's input_size
    logger.info(f"Creating dataset: {config.dataset['name']}")
    reapply_transforms = getattr(args, "reapply_transforms", False)

    if reapply_transforms:
        logger.info("Reapplying transforms after each epoch")

    dataset = DatasetRegistry.create(
        config.dataset["name"],
        root=config.dataset["root"],
        download=config.dataset["download"],
        reapply_transforms=reapply_transforms,
        image_size=input_size[0],  # Model's target size
    )

    num_classes = dataset.num_classes

    # Create model - input_size from config, input_channels from dataset
    logger.info(f"Creating model: {config.model['name']}")
    model_kwargs = {
        "num_classes": num_classes,
        "input_size": input_size,
        "input_channels": dataset.input_channels,  # From dataset
    }

    if "embedding_dim" in config.model:
        model_kwargs["embedding_dim"] = config.model["embedding_dim"]

    model = ModelRegistry.create(config.model["name"], **model_kwargs).to(device)

    # Export class mappings to experiment directory
    try:
        dataset.export_index_label_json(
            output_path=os.path.join(
                experiment_manager.experiment_dir, "index_label_mapping.json"
            )
        )
    except (NotImplementedError, AttributeError) as e:
        logger.warning(f"Could not export class mappings: {e}")

    # Get data loaders
    train_loader, val_loader = dataset.get_dataloaders(
        batch_size=config.training["batch_size"],
        num_workers=train_workers,
        shuffle_train=config.training.get("shuffle_train", True),
    )

    # Save config with model as source of truth
    config_dict = config.to_dict()
    config_dict["dataset"]["name"] = config.dataset["name"]
    config_dict["model"]["num_classes"] = model.num_classes
    config_dict["model"]["input_channels"] = model.input_channels
    config_dict["model"]["input_size"] = model.input_size
    with open(experiment_manager.config_file, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Create checkpoint manager first (needed for resume)
    save_frequency = config.checkpointing.get("save_frequency", 10)
    checkpoint_manager = CheckpointManager(
        checkpoints_dir=experiment_manager.checkpoints_dir,
        save_frequency=save_frequency,
    )

    # Load checkpoint config if resuming
    model_config = None
    if resume_checkpoint:
        checkpoint_data = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=True
        )
        model_config = checkpoint_data.get("model_config", {})
        logger.info(f"Loaded model config from checkpoint: {model_config}")

    # Get layer ID mapping for freeze functionality
    id_to_name = get_layer_id_mapping(model)

    # Freeze layers if specified
    if args.freeze:
        frozen_count, frozen_modules = freeze_layers(
            model, args.freeze, id_to_name, logger
        )
        logger.info(
            f"Frozen {frozen_count} parameter tensors in modules: {sorted(frozen_modules)}"
        )

    # Print model info
    model_info = model.get_model_info()
    info_str = ""
    padding_num = max(map(lambda x: len(x), [k for k in model_info.keys()]))
    for k, v in model_info.items():
        info_str += f"\n  {k.rjust(padding_num)}: {v}"
    logger.info(f"Model info: {info_str}")
    summary(model, input_size=(1, model.input_channels, config_dict["model"]["input_size"], config_dict["model"]["input_size"]), col_names=["input_size", "output_size", "num_params", "mult_adds"])

    # Create criterion using model's get_criterion method
    model_class = ModelRegistry.get(config.model["name"])
    criterion = model_class.get_criterion(**config.optimization)
    logger.info(f"Using criterion: {criterion.__class__.__name__}")

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    if scheduler:
        logger.info(f"Using scheduler: {config.optimization.get('scheduler', 'none')}")

    if args.mixed_precision and device.type == "cuda":
        logger.info("Mixed precision training enabled (FP16)")
    elif args.mixed_precision and device.type != "cuda":
        logger.warning("Mixed precision requested but not available on CPU, using FP32")

    # Create trainer first
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment_manager=experiment_manager,
        checkpoint_manager=checkpoint_manager,
        dataset=dataset,
        patience=args.patience,
        scheduler=scheduler,
        use_amp=args.mixed_precision,
    )

    # Load checkpoint weights if resuming or forking
    if resume_checkpoint:
        # Load history only when resuming (not forking)
        if args.resume:
            start_epoch = trainer.load_history_from_log()
            logger.info(f"Loaded history from {start_epoch} epochs")

        # Load checkpoint weights (non-strict to allow partial loading like YOLO)
        try:
            checkpoint_info = checkpoint_manager.load_checkpoint(
                resume_checkpoint, model, optimizer, strict=False
            )

            # Check if checkpoint was fully restored
            if not checkpoint_info.get("fully_restored", True):
                logger.warning(
                    f"Checkpoint not fully restored: "
                    f"{checkpoint_info.get('loaded_layers', 0)}/{checkpoint_info.get('total_layers', 0)} "
                    f"layers matched. Treating as new architecture."
                )
                checkpoint_manager.best_accuracy = 0.0
                # Clear best_model.pt since it's from a different architecture
                best_model_path = os.path.join(
                    experiment_manager.checkpoints_dir, "best_model.pt"
                )
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                    logger.info(
                        "Removed old best_model.pt from different architecture"
                    )
            else:
                # Use epoch from log file as it's more reliable
                checkpoint_manager.best_accuracy = checkpoint_info["accuracy"]

                # Also check best_model.pt to get the true best accuracy
                # When forking, check source directory; otherwise check current directory
                best_model_path = os.path.join(source_checkpoints_dir, "best_model.pt")
                if (
                    os.path.exists(best_model_path)
                    and best_model_path != resume_checkpoint
                ):
                    best_checkpoint = torch.load(
                        best_model_path, map_location="cpu", weights_only=True
                    )
                    best_acc = best_checkpoint.get("accuracy", 0.0)
                    if best_acc > checkpoint_manager.best_accuracy:
                        checkpoint_manager.best_accuracy = best_acc
                        logger.info(
                            f"Loaded best accuracy from best_model.pt: {best_acc:.2f}%"
                        )

            logger.info(
                f"Resumed from epoch {start_epoch}, best accuracy: {checkpoint_manager.best_accuracy:.2f}%"
            )

            # Override learning rate from config/CLI (not from checkpoint)
            new_lr = config.optimization.get("learning_rate", 1e-3)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            logger.info(
                f"Learning rate set to {new_lr:.2e} (from config/CLI, not checkpoint)"
            )

            # Sync trainer's early stopping state with checkpoint
            trainer.best_accuracy = checkpoint_manager.best_accuracy
            trainer.best_epoch = checkpoint_info["epoch"]
            # Note: epochs_without_improvement resets on resume by design
            # This allows training to continue and potentially find better results
            trainer.epochs_without_improvement = 0
        except RuntimeError as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.error(
                "The checkpoint may be incompatible with the current model architecture."
            )
            logger.error(
                "Please check if the model code has changed since the checkpoint was created."
            )
            raise

    # Start training
    if args.patience > 0:
        logger.info(f"Early stopping enabled with patience={args.patience}")
    logger.info(f"Starting training for {config.training['epochs']} epochs")

    try:
        results = trainer.train(
            epochs=config.training["epochs"], start_epoch=start_epoch
        )

        # Print summary
        if results.get("stopped_early", False):
            logger.info(
                f"\nTraining stopped early (no improvement for {args.patience} epochs)"
            )
            logger.info(f"Best epoch: {results['best_epoch']}")
        else:
            logger.info("\nTraining completed!")
        logger.info(f"Experiment directory: {results['experiment_dir']}")
        logger.info(f"Epochs trained: {results['epochs_trained']}")
        logger.info(f"Best validation accuracy: {results['best_accuracy']:.2f}%")
        logger.info(f"Training time: {results['training_time']:.2f} seconds")

        # Save final model
        final_model_path = experiment_manager.checkpoints_dir + "/final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved as {final_model_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

        # Save interrupted model
        interrupted_path = experiment_manager.checkpoints_dir + "/interrupted_model.pth"
        torch.save(model.state_dict(), interrupted_path)
        logger.info(f"Interrupted model saved as {interrupted_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
