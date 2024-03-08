from experiment import Experiment
import os
import torch
import argparse

torch.backends.cudnn.benchmark = True


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    """
    Arguments parser
    """
    parser = argparse.ArgumentParser(description="Hyperparameters and config parser")

    parser.add_argument("--BATCH_SIZE", dest="BATCH_SIZE",
                      help="{choose a batch size, prefferably a power of 2}",
                      default = 2,
                      type=int)

    parser.add_argument("--DATASET", dest="DATASET",
                      help="{Path to the train and evaluate the model}",
                      default = f"WikiMatrix",
                      type=str)

    parser.add_argument("--DEVICE", dest="DEVICE",
                      choices=["cuda:0"],
                      help="{cuda device to use}",
                      default = "cuda:0",
                      type=str)

    parser.add_argument("--EPOCH_NUM", dest="EPOCH_NUM",
                      help="{choose a number of epochs, -1 for infinite}",
                      default = 100,
                      type=int)
    
    parser.add_argument("--LANGUAGE_INPUT", dest="LANGUAGE_INPUT",
                      help="{language of the input data}",
                      default = f"fr",
                      type=str)
    
    parser.add_argument("--LANGUAGE_TARGET", dest="LANGUAGE_TARGET",
                      help="{language of the target data}",
                      default = f"it",
                      type=str)
    
    parser.add_argument("--LEARNING_RATE", dest="LEARNING_RATE",
                      help="{choose a learning rate}",
                      default = 1e-5,
                      type=float)
    
    parser.add_argument("--LOAD_CHECKPOINT", dest="LOAD_CHECKPOINT",
                      help="{.pth checkpoint file name ex : checkpoint.pth}",
                      default = None)
    
    parser.add_argument("--MIN_LEARNING_RATE", dest="MIN_LEARNING_RATE",
                      help="{choose a minimal learning rate}",
                      default = 1e-6,
                      type=float)
    
    parser.add_argument("--NUM_WORKERS", dest="NUM_WORKERS",
                      help="{Number of workers on CPU for data loading}",
                      default = 1,
                      type=int)
    
    parser.add_argument("--PARAM_BITS", dest="PARAM_BITS",
                      help="{Number of bits for the model parameters}",
                      choices=[16, 32],
                      default = 16,
                      type=int)

    parser.add_argument("--RANDOM_SEED", dest="RANDOM_SEED",
                      help="{Random seed for experiments reproducibility}",
                      default = 42,
                      type=int)
    
    parser.add_argument("--RUN_MODE", dest="RUN_MODE",
                      choices=["train", "test"],
                      help="{train, test}",
                      default = "train",
                      type=str)
    
    parser.add_argument("--SAVE_CHECKPOINTS", dest="SAVE_CHECKPOINTS",
                      choices=["yes", "no"],
                      help="{yes, no}",
                      default = "no",
                      type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Configuration
    cfg = parse_args()

    # Use CPU if no GPU is available
    cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Run
    if cfg.RUN_MODE == "train":
        execute = Experiment(cfg)
        execute.train()

    if cfg.RUN_MODE == "test":
        execute = Experiment(cfg)
        execute.test()