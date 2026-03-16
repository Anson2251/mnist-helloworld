import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ..utils import setup_logger

logger = setup_logger("checkpoint")


class CheckpointManager:
    """Manage model checkpoints within an experiment."""

    def __init__(self, checkpoints_dir: str, save_frequency: int = 10):
        self.checkpoints_dir = checkpoints_dir
        self.save_frequency = save_frequency
        os.makedirs(checkpoints_dir, exist_ok=True)
        self.best_accuracy = 0.0
        self.best_loss = float("inf")

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        accuracy: float,
        filepath: str,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
            "model_config": getattr(model, "get_model_info", lambda: {})(),
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, filepath)

    def load_checkpoint(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            strict: If False, loads compatible layers and skips incompatible ones (YOLO-style)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        model_device = next(model.parameters()).device
        checkpoint = torch.load(filepath, map_location=model_device, weights_only=True)
        state_dict = checkpoint["model_state_dict"]
        model_state = model.state_dict()

        # Initialize for return values
        compatible_layers = {}
        incompatible_keys = []

        if strict:
            model.load_state_dict(state_dict, strict=True)
        else:
            # YOLO-style partial loading: only load compatible layers
            for k, v in state_dict.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        compatible_layers[k] = v
                    else:
                        incompatible_keys.append((k, v.shape, model_state[k].shape))
                else:
                    incompatible_keys.append((k, v.shape, None))

            # Load compatible layers
            model.load_state_dict(compatible_layers, strict=False)

            # Report results
            total_keys = len(state_dict)
            loaded_keys = len(compatible_layers)
            logger.info(f"Checkpoint loaded: {loaded_keys}/{total_keys} layers matched")
            if incompatible_keys:
                logger.info(f"Skipped {len(incompatible_keys)} incompatible layers:")
                for key, ckpt_shape, model_shape in incompatible_keys[:10]:
                    if model_shape:
                        logger.info(
                            f"  {key}: checkpoint {ckpt_shape} vs model {model_shape}"
                        )
                    else:
                        logger.info(
                            f"  {key}: not in model (checkpoint shape: {ckpt_shape})"
                        )
                if len(incompatible_keys) > 10:
                    logger.info(f"  ... and {len(incompatible_keys) - 10} more")

        # Determine if checkpoint was fully restored
        fully_restored = strict or (len(incompatible_keys) == 0 if not strict else True)

        # Skip optimizer state if model architecture differs
        load_optimizer = (
            optimizer and "optimizer_state_dict" in checkpoint and fully_restored
        )

        if load_optimizer:
            try:
                assert optimizer is not None, "Optimizer is None, cannot load state"
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError as e:
                logger.warning(f"Could not load optimizer state: {e}")
                logger.warning("Optimizer will be initialized from scratch")
        elif optimizer and "optimizer_state_dict" in checkpoint and not fully_restored:
            logger.info(
                "Model architecture changed, optimizer state will be initialized from scratch"
            )

        # Determine if checkpoint was fully restored
        fully_restored = strict or (len(incompatible_keys) == 0 if not strict else True)

        return {
            "epoch": checkpoint.get("epoch", 0),
            "loss": checkpoint.get("loss", 0.0),
            "accuracy": checkpoint.get("accuracy", 0.0),
            "fully_restored": fully_restored,
            "loaded_layers": len(compatible_layers) if not strict else len(state_dict),
            "total_layers": len(state_dict),
        }

    def save_best_model(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        accuracy: float,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """Save best model if current accuracy is better."""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_path = os.path.join(self.checkpoints_dir, "best_model.pt")
            self.save_checkpoint(
                model, optimizer, epoch, loss, accuracy, best_path, additional_info
            )
            return True
        return False

    def save_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        accuracy: float,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """Save latest checkpoint."""
        latest_path = os.path.join(self.checkpoints_dir, "latest_checkpoint.pt")
        self.save_checkpoint(
            model, optimizer, epoch, loss, accuracy, latest_path, additional_info
        )

    def save_epoch_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        accuracy: float,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """Save checkpoint for a specific epoch."""
        epoch_path = os.path.join(self.checkpoints_dir, f"epoch_{epoch}.pt")
        self.save_checkpoint(
            model, optimizer, epoch, loss, accuracy, epoch_path, additional_info
        )
