import argparse
from typing import Dict, Any, Optional, List

import yaml

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def get_available_datasets() -> List[str]:
    """Get list of available datasets from registry."""
    try:
        from ..datasets import DatasetRegistry

        return DatasetRegistry.list_available()
    except ImportError:
        # Fallback to hardcoded list if import fails
        return [
            "mnist",
            "cifar10",
            "subset_631",
            "subset_1000",
            "triplet_mnist",
            "balanced_triplet_mnist",
        ]


def get_available_models() -> List[str]:
    """Get list of available models from registry."""
    try:
        from ..models import ModelRegistry

        return ModelRegistry.list_available()
    except ImportError:
        # Fallback to hardcoded list if import fails
        return ["lenet", "mynet", "bottleneck_vit", "fpn_vit", "siamese"]


class Config:
    """Configuration management."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self.dataset = config_dict.get("dataset", {})
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})
        self.optimization = config_dict.get("optimization", {})
        self.checkpointing = config_dict.get("checkpointing", {})

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML configuration files. Install with: pip install pyyaml"
            )

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create configuration from command line arguments."""
        config_dict = {
            "dataset": {"name": args.dataset, "root": args.data_root, "download": True},
            "model": {
                "name": args.model,
                "num_classes": args.num_classes,
                "input_size": [args.input_size, args.input_size],
                "embedding_dim": getattr(args, "embedding_dim", 256),
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            },
            "optimization": {
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "scheduler": args.scheduler,
                "scheduler_step_size": args.scheduler_step_size,
                "scheduler_gamma": args.scheduler_gamma,
                "scheduler_t_max": args.scheduler_t_max,
                "scheduler_eta_min": args.scheduler_eta_min,
                "scheduler_patience": args.scheduler_patience,
                "scheduler_factor": args.scheduler_factor,
                "triplet_margin": getattr(args, "triplet_margin", 1.0),
                "embedding_dim": getattr(args, "embedding_dim", 256),
                "muon_momentum": args.muon_momentum,
                "muon_ns_steps": args.muon_ns_steps,
                "adam_lr": args.adam_lr,
                "adam_betas": args.adam_betas,
            },
            "checkpointing": {
                "checkpoint_dir": args.checkpoint_dir,
                "save_frequency": args.save_frequency,
            },
        }
        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "dataset": {"name": "mnist", "root": "./data", "download": True},
        "model": {
            "name": "mynet",
            "num_classes": 10,
            "input_channels": 1,
            "input_size": [28, 28],
        },
        "training": {
            "epochs": 20,
            "batch_size": 64,
            "num_workers": 4,
            "shuffle_train": True,
        },
        "optimization": {
            "learning_rate": 1e-3,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "scheduler": "none",
            "scheduler_step_size": 10,
            "scheduler_gamma": 0.1,
            "scheduler_t_max": 100,
            "scheduler_eta_min": 1e-6,
            "scheduler_patience": 5,
            "scheduler_factor": 0.1,
            "muon_momentum": 0.95,
            "muon_ns_steps": 5,
            "adam_lr": 3e-4,
            "adam_betas": [0.9, 0.95],
        },
        "checkpointing": {"checkpoint_dir": "checkpoints", "save_frequency": 10},
    }


def create_config_parser() -> argparse.ArgumentParser:
    """Create argument parser for configuration."""
    parser = argparse.ArgumentParser(description="Train a neural network")

    # Get available choices dynamically
    available_datasets = get_available_datasets()
    available_models = get_available_models()

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=available_datasets,
        help=f"Dataset to use. Available: {', '.join(available_datasets)}",
    )
    parser.add_argument(
        "--data-root", type=str, default="./data", help="Root directory for datasets"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="mynet",
        choices=available_models,
        help=f"Model architecture. Available: {', '.join(available_models)}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use for training",
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument(
        "--input-size",
        type=int,
        default=64,
        help="Input image size (height and width, will be squared)",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    # Optimization arguments
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd", "muon", "muon_with_aux_adam"],
        help="Optimizer",
    )
    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=0.95,
        help="Muon momentum parameter (only used for muon optimizers)",
    )
    parser.add_argument(
        "--muon-ns-steps",
        type=int,
        default=5,
        help="Newton-Schulz iteration steps for Muon (only used for muon optimizers)",
    )
    parser.add_argument(
        "--adam-lr",
        type=float,
        default=3e-4,
        help="Learning rate for Adam parts in MuonWithAuxAdam",
    )
    parser.add_argument(
        "--adam-betas",
        type=float,
        nargs=2,
        default=[0.9, 0.95],
        help="Adam betas for MuonWithAuxAdam (two values)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "step", "cosine", "plateau", "exponential"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=10,
        help="Step size for StepLR scheduler",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="Gamma for learning rate decay",
    )
    parser.add_argument(
        "--scheduler-t-max",
        type=int,
        default=100,
        help="T_max for CosineAnnealingLR",
    )
    parser.add_argument(
        "--scheduler-eta-min",
        type=float,
        default=1e-6,
        help="Eta_min for CosineAnnealingLR",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=5,
        help="Patience for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.1,
        help="Factor for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--triplet-margin",
        type=float,
        default=1.0,
        help="Margin for triplet loss (used by siamese models)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension for siamese/metric learning models",
    )

    # Checkpointing arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="Save frequency in epochs"
    )

    # Configuration file
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        metavar="EXP",
        help="Resume training from a specific experiment (e.g., exp1, exp2)",
    )

    # Fork training
    parser.add_argument(
        "--fork",
        type=str,
        metavar="EXP",
        help="Fork training from a specific experiment - loads checkpoint but saves to a new experiment",
    )

    # Layer freezing
    parser.add_argument(
        "--freeze",
        type=str,
        nargs="+",
        help=(
            "Freeze specific layers. Supports: "
            "(1) Layer IDs: --freeze 2-1 2-2; "
            "(2) ID ranges: --freeze 2-1:2-5; "
            "(3) Name patterns: --freeze features classifier"
        ),
    )

    # Data augmentation
    parser.add_argument(
        "--reapply-transforms",
        action="store_true",
        help="Reapply random transforms after each epoch for better generalization",
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (0 to disable). Stops training if no improvement after N epochs.",
    )

    # Mixed precision training
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training (FP16) for faster training on compatible GPUs",
    )

    return parser


def load_config(
    config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None
) -> Config:
    """Load configuration from file or arguments."""
    if config_path:
        return Config.from_yaml(config_path)
    elif args:
        return Config.from_args(args)
    else:
        return Config(get_default_config())
