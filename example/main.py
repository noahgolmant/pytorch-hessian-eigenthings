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
    parser.add_argument(
        "--num_eigenthings",
        default=5,
        type=int,
        help="number of eigenvals/vecs to compute",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="train set batch size"
    )
    parser.add_argument(
        "--eval_batch_size", default=16, type=int, help="test set batch size"
    )
    parser.add_argument(
        "--momentum", default=0.0, type=float, help="power iteration momentum term"
    )
    parser.add_argument(
        "--num_steps", default=50, type=int, help="number of power iter steps"
    )
    parser.add_argument("--max_samples", default=2048, type=int)
    parser.add_argument("--cuda", action="store_true", help="if true, use CUDA/GPUs")
    parser.add_argument(
        "--full_dataset",
        action="store_true",
        help="if true,\
                        loop over all batches in set for each gradient step",
    )
    parser.add_argument("--fname", default="", type=str)
    parser.add_argument("--mode", type=str, choices=["power_iter", "lanczos"])


def main(args):
    trainloader, testloader = build_dataset(
        "cifar10",
        dataroot=args.dataroot,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=2,
    )
    if args.fname:
        print("Loading model from %s" % args.fname)
        model = torch.load(args.fname, map_location="cpu").cuda()
    else:
        model = build_model("ResNet18", num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model,
        testloader,
        criterion,
        args.num_eigenthings,
        mode=args.mode,
        # power_iter_steps=args.num_steps,
        max_possible_gpu_samples=args.max_samples,
        # momentum=args.momentum,
        full_dataset=args.full_dataset,
        use_gpu=args.cuda,
    )
    print("Eigenvecs:")
    print(eigenvecs)
    print("Eigenvals:")
    print(eigenvals)
    # track.metric(iteration=0, eigenvals=eigenvals)


if __name__ == "__main__":
    skeletor.supply_args(extra_args)
    skeletor.execute(main)
