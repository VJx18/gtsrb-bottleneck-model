import argparse
import torch
import sys
import os
from src.training.stage2 import train_label_predictor
from src.training.stage1 import train_concept_predictor

def add_to_sys_path(path):
    absPath = os.path.abspath(path)

    if not os.path.isdir(absPath):
        raise ValueError(f"path {absPath} does not exist!")
    if absPath not in sys.path:
        sys.path.append(absPath)
    try:
        import config
        return config
    except ImportError:
        print(f"No import of {absPath}")
        return None
    
    

    

def main():
    parser = argparse.ArgumentParser(description='Train CBM Model')
    parser.add_argument('--config', type=str, default='./src/config', help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--patience', type=int, help='Early stopping patience (overrides config)')
    args = parser.parse_args()
    
    config = add_to_sys_path(args.config)
    config = config.Config()

    config.seed = torch.manual_seed(args.seed)
    
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.lr = args.lr
    if args.patience is not None:
        config.training.patience = args.patience

    print(f'Training with learning rate: {config.training.lr}, epochs: {config.training.epochs}, patience: {config.training.patience}, seed: {args.seed}, ')
    # train both stages
    train_concept_predictor(config)
    train_label_predictor(config)
    print('Training completed.')

if __name__ == '__main__':
    main()