import argparse

from preprocess import preprocess
from train import train



def main(args):
    if args.preprocess:
        preprocess()
    if args.train:
        print("Training")
        train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Transformer")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--train", action="store_true", help="Launch the training")
    args = parser.parse_args()
    main(args)





