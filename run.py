from experiment import Experiment
import os
import torch
import argparse


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    """
    Arguments parser
    """
    parser = argparse.ArgumentParser(description="Hyperparameters and config parser")

    parser.add_argument("--BATCH_SIZE", dest="BATCH_SIZE",
                      help="Size of the data batches to process",
                      default = 1,
                      type=int)

    parser.add_argument("--DATASET_NAME", dest="DATASET_NAME",
                      help="Path to the train and evaluate the model",
                      default = f"WikiMatrix",
                      type=str)

    parser.add_argument("--DEVICE", dest="DEVICE",
                      choices=["cuda:0"],
                      help="Cuda device to use (defaults to cpu when no cuda device is available)",
                      default = "cuda:0",
                      type=str)

    parser.add_argument("--EPOCH_NUM", dest="EPOCH_NUM",
                      help="Number of training epochs, -1 for infinite",
                      default = 100,
                      type=int)
    
    parser.add_argument("--HALF_BITS", dest="HALF_BITS",
                      help="If enables, uses 16 bits model weights instead of 32 bits",
                      action="store_true")
    
    parser.add_argument("--LANGUAGE_INPUT", dest="LANGUAGE_INPUT",
                      help="Language of the input data",
                      default = f"fr",
                      type=str)
    
    parser.add_argument("--LANGUAGE_TARGET", dest="LANGUAGE_TARGET",
                      help="Language of the target data",
                      default = f"it",
                      type=str)
    
    parser.add_argument("--LEARNING_RATE", dest="LEARNING_RATE",
                      help="Learning rate for model parameters update",
                      default = 1e-5,
                      type=float)
    
    parser.add_argument("--LOAD_CHECKPOINT", dest="LOAD_CHECKPOINT",
                      help="If set, loads .pt checkpoint file. ex : checkpoint",
                      nargs="?",
                      default=False,)
    
    parser.add_argument("--MIN_LEARNING_RATE", dest="MIN_LEARNING_RATE",
                      help="choose a minimal learning rate",
                      default = 1e-6,
                      type=float)
    
    parser.add_argument("--MODEL_DIMS", dest="MODEL_DIMS",
                      help="Number of the dimensions for signal repesentation",
                      default=512,
                      type=int)
    
    parser.add_argument("--MODEL_HEADS", dest="MODEL_HEADS",
                      help="Number of attention heads per model layer",
                      default=8,
                      type=int)
    
    parser.add_argument("--MODEL_LAYERS", dest="MODEL_LAYERS",
                      help="Number of layers in the model",
                      default=6,
                      type=int)
    
    parser.add_argument("--NUM_WORKERS", dest="NUM_WORKERS",
                      help="Number of workers on CPU for data loading",
                      default = 1,
                      type=int)

    parser.add_argument("--RANDOM_SEED", dest="RANDOM_SEED",
                      help="Random seed for experiments reproducibility",
                      default = 42,
                      type=int)
    
    parser.add_argument("--RUN_MODE", dest="RUN_MODE",
                      choices=["train", "test"],
                      help="{train, test}",
                      default = "train",
                      type=str)
    
    parser.add_argument("--SAVE_CHECKPOINTS", dest="SAVE_CHECKPOINTS",
                      help="If set, saves checkpoints in a .pt file with the provided name. ex : bestmodel",
                      nargs="?",
                      default=False,)
    
    parser.add_argument("--SEQUENCE_LENGTH", dest="SEQUENCE_LENGTH",
                      help="Maximal sequence length to be processed",
                      default = 128,
                      type=int)
    
    parser.add_argument("--TEST_EXAMPLES", dest="TEST_EXAMPLES",
                      help="Number of test examples",
                      default = 10000,
                      type=int)
    
    parser.add_argument("--VAL_EXAMPLES", dest="VAL_EXAMPLES",
                      help="Number of validation examples",
                      default = 10000,
                      type=int)
    
    parser.add_argument("--VOCAB_SIZE", dest="VOCAB_SIZE",
                      help="Number of tokens in the tokenizer vocabulary",
                      default = 30000,
                      type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Configuration
    cfg = parse_args()

    # Use CPU if no GPU is available
    cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Cudnn setup
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    print(cfg)

    # Run
    if cfg.RUN_MODE == "train":
        execute = Experiment(cfg)
        execute.train()

    if cfg.RUN_MODE == "test":
        execute = Experiment(cfg)
        execute.test()