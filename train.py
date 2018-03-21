import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision
import models
import utils
import tabulate


parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
ds = getattr(torchvision.datasets, args.dataset)
path = os.path.join(args.data_path, args.dataset.lower())
train_set = ds(path, train=True, download=True, transform=model_cfg.transform_train)
test_set = ds(path, train=False, download=True, transform=model_cfg.transform_test)
loaders = {
    'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'test': torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
}
num_classes = max(train_set.train_labels) + 1

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()


if args.swa:
    print('SWA training')
    swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    swa_model.cuda()
    swa_n = 0
else:
    print('SGD training')


def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


criterion = F.cross_entropy
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if args.swa:
        swa_state_dict = checkpoint['swa_state_dict']
        if swa_state_dict is not None:
            swa_model.load_state_dict(swa_state_dict)
        swa_n_ckpt = checkpoint['swa_n']
        if swa_n_ckpt is not None:
            swa_n = swa_n_ckpt

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
if args.swa:
    columns = columns[:-1] + ['swa_te_loss', 'swa_te_acc'] + columns[-1:]
    swa_res = {'loss': None, 'accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    swa_state_dict=swa_model.state_dict() if args.swa else None,
    swa_n=swa_n if args.swa else None,
    optimizer=optimizer.state_dict()
)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)
    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)
    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None}

    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
        swa_n += 1
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            utils.bn_update(loaders['train'], swa_model)
            swa_res = utils.eval(loaders['test'], swa_model, criterion)
        else:
            swa_res = {'loss': None, 'accuracy': None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict() if args.swa else None,
            swa_n=swa_n if args.swa else None,
            optimizer=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep]
    if args.swa:
        values = values[:-1] + [swa_res['loss'], swa_res['accuracy']] + values[-1:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict() if args.swa else None,
        swa_n=swa_n if args.swa else None,
        optimizer=optimizer.state_dict()
    )
