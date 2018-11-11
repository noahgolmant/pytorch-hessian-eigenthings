"""
A simple example to calculate the top eigenvectors for the hessian of
ResNet18 network for CIFAR-10
"""

import track
import skeletor
from skeletor.datasets import build_dataset
from skeletor.models import build_model

import torch

from hessian_eigenthings import compute_hessian_eigenthings


def extra_args(parser):
    parser.add_argument('--num_eigenthings', default=5, type=int,
                        help='number of eigenvals/vecs to compute')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='train set batch size')
    parser.add_argument('--eval_batch_size', default=16, type=int,
                        help='test set batch size')
    parser.add_argument('--momentum', default=0.0, type=float,
                        help='power iteration momentum term')
    parser.add_argument('--num_steps', default=20, type=int,
                        help='number of power iter steps')


def main(args):
    trainloader, testloader = build_dataset('cifar10',
                                            dataroot=args.dataroot,
                                            batch_size=args.batch_size,
                                            eval_batch_size=args.eval_batch_size,
                                            num_workers=2)
    model = build_model('ResNet18', num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    eigenvals, eigenvecs = compute_hessian_eigenthings(model, testloader,
                                                       criterion,
                                                       args.num_eigenthings,
                                                       args.num_steps,
                                                       momentum=args.momentum)
    print("Eigenvecs:")
    print(eigenvecs)
    print("Eigenvals:")
    print(eigenvals)
    track.metric(iteration=0, eigenvals=eigenvals)


if __name__ == '__main__':
    skeletor.supply_args(extra_args)
    skeletor.execute(main)
