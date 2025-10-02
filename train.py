#!/usr/bin/env python3
"""
Modular Training Script for Neural Networks
Supports multiple datasets and model architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets import DatasetRegistry
from src.models import ModelRegistry
from src.training import Trainer, CheckpointManager
from src.config import create_config_parser, load_config
from src.utils import get_device, get_optimal_workers, setup_logger

def create_optimizer(model, config):
    """Create optimizer based on configuration."""
    optimizer_name = config.optimization.get('optimizer', 'adamw').lower()
    learning_rate = config.optimization.get('learning_rate', 1e-3)
    weight_decay = config.optimization.get('weight_decay', 0.01)

    if optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.optimization.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def main():
    """Main training function."""
    # Parse arguments
    parser = create_config_parser()
    args = parser.parse_args()

    # Load configuration
    config = load_config(config_path=args.config, args=args)

    # Set up logging
    logger = setup_logger()
    logger.info("Starting modular training script")

    # Get device
    device, using_cpu = get_device()
    logger.info(f"Using device: {device}")

    # Get optimal workers
    train_workers, val_workers = get_optimal_workers(using_cpu)
    logger.info(f"Using {train_workers} workers for training, {val_workers} for validation")

    # Create dataset
    logger.info(f"Creating dataset: {config.dataset['name']}")
    dataset = DatasetRegistry.create(
        config.dataset['name'],
        root=config.dataset['root'],
        download=config.dataset['download']
    )

    # Get data loaders
    train_loader, val_loader = dataset.get_dataloaders(
        batch_size=config.training['batch_size'],
        num_workers=train_workers,
        shuffle_train=config.training.get('shuffle_train', True)
    )

    # Create model
    logger.info(f"Creating model: {config.model['name']}")
    model = ModelRegistry.create(
        config.model['name'],
        num_classes=dataset.num_classes,
        input_channels=dataset.input_channels,
        input_size=dataset.input_size
    ).to(device)

    # Print model info
    model_info = model.get_model_info()
    info_str = ""
    padding_num = max(map(lambda x: len(x), [k for k in model_info.keys()]))
    for (k, v) in model_info.items():
        info_str += f"\n  {k.rjust(padding_num)}: {v}"
    logger.info(f"Model info: {info_str}")

    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.checkpointing['checkpoint_dir'] + "/" + config.model['name']
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_manager=checkpoint_manager
    )

    # Start training
    logger.info(f"Starting training for {config.training['epochs']} epochs")

    try:
        results = trainer.train(epochs=config.training['epochs'])

        # Print summary
        logger.info(f"\nTraining completed!")
        logger.info(f"Epochs trained: {results['epochs_trained']}")
        logger.info(f"Best validation accuracy: {results['best_accuracy']:.2f}%")
        logger.info(f"Training time: {results['training_time']:.2f} seconds")

        # Save final model
        final_model_path = f"{config.checkpointing['checkpoint_dir']}/{config.model['name']}/final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved as {final_model_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

        # Save interrupted model
        interrupted_path = f"{config.checkpointing['checkpoint_dir']}/{config.model['name']}/interrupted_model.pth"
        torch.save(model.state_dict(), interrupted_path)
        logger.info(f"Interrupted model saved as {interrupted_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == '__main__':
    main()
