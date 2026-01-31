import argparse
import torch
from src.config.config import Config
from src.training.stage2 import train_label_predictor
from src.training.stage1 import train_concept_predictor

def main():
    parser = argparse.ArgumentParser(description='Train CBM Model')
    parser.add_argument('--config', type=str, default='src/config/config.py', help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    args = parser.parse_args()

    # update config
    config = Config()
    config.seed = torch.manual_seed(args.seed)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.lr = args.lr

    # train both stages
    train_concept_predictor(config)
    train_label_predictor(config)
    print('Training completed.')

if __name__ == '__main__':
    main()