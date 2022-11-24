
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from data import imagenet
from models import *
from utils import progress_bar
from mask import *
import utils
from compute_comp_ratio import compute_ratio, compute_ratio_iterative, compute_ratio_nn, compute_ratio_greedy
from compute_comp_ratio_googlenet import compute_ratio as compute_ratio_googlenet

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10','imagenet'),
    help='dataset')
parser.add_argument(
    '--lr',
    default=0.01,
    type=float,
    help='initial learning rate')
parser.add_argument(
    '--lr_decay_step',
    default='5, 10',
    type=str,
    help='learning rate decay step')
parser.add_argument(
    '--resume',
    type=str,
    default='./checkpoints/',
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--resume_mask',
    type=str,
    default=None,
    help='mask loading')
parser.add_argument(
    '--gpu',
    type=str,
    default='4,6',
    help='Select gpu to use')
parser.add_argument(
    '--job_dir',
    type=str,
    default='./result/temp',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--epochs',
    type=int,
    default=15,
    help='The num of epochs to train.')
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation.')
parser.add_argument(
    '--start_cov',
    type=int,
    default=0,
    help='The num of conv to start prune')
parser.add_argument(
    '--compress_rate',
    type=str,
    default='[0.95]+[0.5]*6+[0.9]*4+[0.8]+[0.0]',
    help='compress rate of each conv')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'),
    help='The architecture to prune')
parser.add_argument(
    '--pr_step',
    type=float,
    default=0.05,
    help='compress rate of each conv')
parser.add_argument(
    '--total_pr',
    type=float,
    default=0.75,
    help='pruning ratio')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument(
    '--limit',
    type=int,
    default=10,
    help='The num of batch to get rank.')

args        = parser.parse_args()
args.resume = args.resume + args.arch + '.pt'

# 'vgg_16_bn'  : [0.7]*7+[0.1]*6,
# 'resnet_56'  : [0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4],
# 'resnet_110' : [0.1]+[0.40]*36+[0.40]*36+[0.4]*36
# 'densenet_40': [0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0],
# 'googlenet'  : [0.10]+[0.7]+[0.5]+[0.8]*4+[0.5]+[0.6]*2,

# 'resnet_50'  : [0.2]+[0.8]*10+[0.8]*13+[0.55]*19+[0.45]*10,

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device_ids = list(map(int, args.gpu.split(',')))

if len(device_ids)==1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed) ##for cpu
if args.gpu:
    torch.cuda.manual_seed(args.seed) ##for gpu

best_acc      = 0  # best test accuracy
start_epoch   = 0  # start from epoch 0 or last checkpoint epoch
lr_decay_step = list(map(int, args.lr_decay_step.split(',')))

ckpt         = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
utils.print_params(vars(args), print_logger.info)

# Data
print_logger.info('==> Preparing data..')

if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset    = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset     = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2)
elif args.dataset=='imagenet':
    data_tmp    = imagenet.Data(args)
    trainloader = data_tmp.loader_train
    testloader  = data_tmp.loader_test
else:
    assert 1 == 0

if args.compress_rate:
    import re
    cprate_str      = args.compress_rate
    cprate_str_list = cprate_str.split('+')
    pat_cprate      = re.compile(r'\d+\.\d*')
    pat_num         = re.compile(r'\*\d+')
    cprate = []
    for x in cprate_str_list:
        num      = 1
        find_num = re.findall(pat_num,x)
        if find_num:
            assert len(find_num) == 1
            num     = int(find_num[0].replace('*',''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])]*num

    compress_rate = cprate
    args.compress_rate = cprate

compress_rate = [0.4964518384054989, 0.2849058934492172, 0.08980860036024552, 0.37073162484256195, 0.2093322216310498, 0.12194669511621586, 0.010377582469950596, 0.357737695584358, 0.01995439420085444, 0.6296248337514334, 0.45419335484341333, 0.384814073689322, 0.5206852081311238]
# compress_rate=[0.21875, 0.0, 0.015625, 0.0, 0.4375, 0.4375, 0.41796875, 0.99609375, 0.974609375, 0.962890625, 0.9765625, 0.984375, 0.8]
# Model
print_logger.info('==> Building model..')
net = eval(args.arch)(compress_rate=compress_rate)
net = net.to(device)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

cudnn.benchmark = True
# print(net)

if len(args.gpu)>1:
    m = eval('mask_' + args.arch)(model = net, compress_rate = net.module.compress_rate, job_dir = args.job_dir, device = device,args=args)
else:
    m = eval('mask_' + args.arch)(model = net, compress_rate = net.compress_rate, job_dir = args.job_dir, device = device,args=args)

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch, cov_id, optimizer, scheduler, pruning=True):
    print_logger.info('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct    = 0
    total      = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        with torch.cuda.device(device):
            inputs  = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            # if pruning:
            #     m.grad_mask(cov_id)

            train_loss   += loss.item()
            _, predicted = outputs.max(1)
            total        += targets.size(0)
            correct      += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,len(trainloader),
                         'Cov: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (cov_id, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def test(epoch, cov_id, optimizer, scheduler):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    global best_acc
    net.eval()
    num_iterations = len(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs         = net(inputs)
            loss            = criterion(outputs, targets)

            prec1, prec5    = utils.accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

        print_logger.info(
            'Epoch[{0}]({1}/{2}): '
            'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                epoch, batch_idx, num_iterations, top1=top1, top5=top5))

    if top1.avg > best_acc:
        print_logger.info('Saving to '+args.arch+'_cov'+str(cov_id)+'.pt')
        state = {
            'state_dict': net.state_dict(),
            'best_prec1': top1.avg,
            'epoch': epoch,
            'scheduler':scheduler.state_dict(),
            'optimizer': optimizer.state_dict() 
        }
        if not os.path.isdir(args.job_dir+'/pruned_checkpoint'):
            os.mkdir(args.job_dir+'/pruned_checkpoint')
        best_acc = top1.avg
        torch.save(state, args.job_dir+'/pruned_checkpoint/'+args.arch+'_cov'+str(cov_id)+'.pt')

    print_logger.info("=>Best accuracy {:.3f}".format(best_acc))

def total_num_filters():
    filters = 0

    if len(args.gpu)>1:
        for name, module in net.module.features.named_modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "mask"):
                    filters += module.mask.sum()
                else:
                    filters += module.out_channels
    else:
        for name, module in net.features.named_modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "mask"):
                    filters += module.mask.sum()
                else:
                    filters += module.out_channels

    return filters


if len(args.gpu)>1:
    convcfg = net.module.covcfg
else:
    convcfg = net.covcfg

param_per_cov_dic={
    'vgg_16_bn': 4,
    'densenet_40': 3,
    'googlenet': 28,
    'resnet_50':3,
    'resnet_56':3,
    'resnet_110':3
}

if len(args.gpu)>1:
    print_logger.info('compress rate: ' + str(net.module.compress_rate))
else:
    print_logger.info('compress rate: ' + str(net.compress_rate))

number_of_filters = total_num_filters()
args.num_filters_to_prune_per_iteration = int(number_of_filters * args.pr_step)  # 0.05 (5%) -> 0.01 (1%) temporally
if args.num_filters_to_prune_per_iteration == 0:
    iterations = 10
else:
    iterations = int(
        float(number_of_filters) / args.num_filters_to_prune_per_iteration)

iterations = int(iterations * args.total_pr)  # up to 80%

#start iterative pruning
for cov_id in range(iterations):
    #2. compute pruning ratio layer wisely (for each iteration)
    compress_rate = compute_ratio_greedy(args, print_logger=print_logger) #pruning ratio 받아오는 for one iteration

    for cov_id in range(args.start_cov, len(convcfg)):  # pruning mask 준비
        # Load pruned_checkpoint
        print_logger.info("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id))
        m.layer_mask(cov_id + 1, resume=args.resume_mask, param_per_cov=param_per_cov_dic[args.arch], arch=args.arch)
        m.grad_mask(cov_id)

    if cov_id == 0: #load pretrained and fully-connected model (in the first iteration)
        pruned_checkpoint = torch.load(args.resume, map_location=device)  # load pretrained full-model

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        if args.arch == 'resnet_50':
            tmp_ckpt = pruned_checkpoint
        else:
            tmp_ckpt = pruned_checkpoint['state_dict']

        if len(args.gpu) > 1:
            for k, v in tmp_ckpt.items():
                new_state_dict['module.' + k.replace('module.', '')] = v
        else:
            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v

        net.load_state_dict(new_state_dict)  # '''
    else: #load pretrained and fully-connected model (in the first iteration)
        if args.arch == 'resnet_50':
            skip_list = [1, 5, 8, 11, 15, 18, 21, 24, 28, 31, 34, 37, 40, 43, 47, 50, 53]
            if cov_id + 1 not in skip_list:
                continue
            else:
                pruned_checkpoint = torch.load(
                    args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(
                        skip_list[skip_list.index(cov_id + 1) - 1]) + '.pt')
                net.load_state_dict(pruned_checkpoint['state_dict'])
        else:
            if len(args.gpu) == 1:
                pruned_checkpoint = torch.load(
                    args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) + '.pt',
                    map_location='cuda:0')
            else:
                pruned_checkpoint = torch.load(
                    args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) + '.pt')
            net.load_state_dict(pruned_checkpoint['state_dict'])

    print_logger.info("iteration: %d ====> Resuming from pruned_checkpoint..." % (cov_id))

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    #4. Pruning and Fine-tuning
    best_acc=0.
    for epoch in range(0, args.epochs):
        train(epoch, cov_id + 1, optimizer, scheduler)
        scheduler.step()
        test(epoch, cov_id + 1, optimizer, scheduler)