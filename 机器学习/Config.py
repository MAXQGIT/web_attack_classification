import argparse


def Config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='../data/train', type=str, required=False)
    parser.add_argument('--test_radio', default=0.05, type=float, required=False)
    parser.add_argument('--num_classes', default=6, type=float, required=False)
    parser.add_argument('--FOLDS', default=5, type=int, required=False)
    args = parser.parse_args()
    return args
