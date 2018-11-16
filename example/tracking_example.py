"""
Main file to orchestrate model training! This will track the top
eigenvalues during training and log them to track.
"""

import os
import time

import torch
import track

import skeletor
from skeletor.datasets import build_dataset, num_classes
from skeletor.models import build_model
from skeletor.optimizers import build_optimizer
from skeletor.utils import AverageMeter, accuracy, progress_bar

from hessian_eigenthings import HessianTracker

def add_train_args(parser):
    # Main arguments go here
    parser.add_argument('--arch', default='ResNet18')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--lr', default=.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 190],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='SGD weight decay')
    parser.add_argument('--cuda', action='store_true',
                        help='if true, use GPU!')
    parser.add_argument('--num_eigenthings', default=1, type=int,
                        help='number of eigenvalues to track')


def adjust_learning_rate(epoch, optimizer, lr, schedule, decay):
    if epoch in schedule:
        new_lr = lr * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        new_lr = lr
    return new_lr


def train(trainloader, model, criterion, optimizer, epoch, cuda=False):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        progress_str = 'Loss: %.3f | Acc: %.3f%% (%d/%d)'\
            % (losses.avg, top1.avg, top1.sum, top1.count)
        progress_bar(batch_idx, len(trainloader), progress_str)

        iteration = epoch * len(trainloader) + batch_idx
        track.metric(iteration=iteration, epoch=epoch,
                     avg_train_loss=losses.avg,
                     avg_train_acc=top1.avg,
                     cur_train_loss=loss.item(),
                     cur_train_acc=prec1.item())
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, epoch, cuda=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs = torch.autograd.Variable(inputs, volatile=True)
            targets = torch.autograd.Variable(targets, volatile=True)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            progress_str = 'Loss: %.3f | Acc: %.3f%% (%d/%d)'\
                % (losses.avg, top1.avg, top1.sum, top1.count)
            progress_bar(batch_idx, len(testloader), progress_str)
    track.metric(iteration=0, epoch=epoch,
                 avg_test_loss=losses.avg,
                 avg_test_acc=top1.avg)
    return (losses.avg, top1.avg)


def do_training(args):
    trainloader, testloader = build_dataset(args.dataset,
                                            dataroot=args.dataroot,
                                            batch_size=args.batch_size,
                                            eval_batch_size=args.eval_batch_size,
                                            num_workers=2)
    model = build_model(args.arch, num_classes=num_classes(args.dataset))
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    # Calculate total number of model parameters
    num_params = sum(p.numel() for p in model.parameters())
    track.metric(iteration=0, num_params=num_params)

    optimizer = build_optimizer('SGD', params=model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()

    # Create the hessian tracker!
    hessian_tracker = HessianTracker(model, trainloader, criterion,
                                     num_eigenthings=args.num_eigenthings,
                                     momentum=0.9,
                                     use_gpu=args.cuda)

    best_acc = 0.0
    for epoch in range(args.epochs):
        track.debug("Starting epoch %d" % epoch)
        args.lr = adjust_learning_rate(epoch, optimizer, args.lr, args.schedule,
                                       args.gamma)
        train_loss, train_acc = train(trainloader, model, criterion,
                                      optimizer, epoch, args.cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch,
                                   args.cuda)
        track.debug('Finished epoch %d... | train loss %.3f | train acc %.3f '
                    '| test loss %.3f | test acc %.3f'
                    % (epoch, train_loss, train_acc, test_loss, test_acc))

        # Save model
        model_fname = os.path.join(track.trial_dir(),
                                   "model{}.ckpt".format(epoch))
        torch.save(model, model_fname)
        if test_acc > best_acc:
            best_acc = test_acc
            best_fname = os.path.join(track.trial_dir(), "best.ckpt")
            track.debug("New best score! Saving model")
            torch.save(model, best_fname)

        # Compute the hessian spectrum
        hessian_tracker.step()
        eigenvals, _ = hessian_tracker.get_eigenthings()
        track.debug('Top eigenvalue: %.3f' % float(max(eigenvals)))
        track.metric(iteration=0, epoch=epoch,
                     eigenvals=eigenvals)


def postprocess(proj):
    df = skeletor.proc.df_from_proj(proj)
    if 'avg_test_acc' in df.columns:
        best_trial = df.ix[df['avg_test_acc'].idxmax()]
        print("Trial with top accuracy:")
        print(best_trial)


if __name__ == '__main__':
    skeletor.supply_args(add_train_args)
    skeletor.supply_postprocess(postprocess, save_proj=True)
    skeletor.execute(do_training)
