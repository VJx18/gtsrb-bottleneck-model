import argparse
from src.config.config import Config
from src.evaluation.evaluation import evaluate_cbm_model
from src.data.dataset import get_dataloaders

def main():
    parser = argparse.ArgumentParser(description='Evaluate CBM Model')
    parser.add_argument('--checkpoint', type=str, default='./experiments/checkpoints/best_cbm_model.pth', help='Path to model checkpoint')
    parser.add_argument('--data_path_training', type=str, help='Path to training data directory')
    parser.add_argument('--data_path_testing', type=str,  help='Path to test data directory')
    args = parser.parse_args()

    # update config if needed
    config = Config()
    config.training.checkpoint_dir = args.checkpoint
    if args.data_path_training is not None:
        config.dataset.train_images = args.data_path_training
    if args.data_path_testing is not None:
        config.dataset.test_images = args.data_path_testing

    # evaluate model
    _, _, test_loader = get_dataloaders(config)
    evaluate_cbm_model(config=config, test_loader=test_loader)

if __name__ == '__main__':
    main()