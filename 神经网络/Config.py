import argparse


def Config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='../data/train', type=str, required=False)
    parser.add_argument('--label', default='label', type=str, required=False)
    parser.add_argument('--train_radio', default=0.90, type=float, required=False)
    parser.add_argument('--num_class', default=6, type=int, required=False)
    parser.add_argument('--batch_size', default=8, type=int, required=False)
    parser.add_argument('--device', default='cuda', type=str, required=False)
    parser.add_argument('--input_dim', default=3, type=int, required=False)
    parser.add_argument('--hidden', default=256, type=int, required=False)
    parser.add_argument('--output_dim', default=6, type=int, required=False)
    parser.add_argument('--lr', default=0.001, type=float, required=False)
    parser.add_argument('--epochs', default=100, type=int, required=False)
    args = parser.parse_args()
    return args
