
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from data import imagenet
from models import *
from utils.utils import progress_bar
from utils.parser import get_args
import utils
from mask import *
from compute_comp_ratio import compute_ratio, compute_ratio_iterative, compute_ratio_nn, compute_ratio_energy
from compute_comp_ratio_googlenet import compute_ratio as compute_ratio_googlenet

if __name__== '__main__':
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device_ids = list(map(int, args.gpu.split(',')))

    if len(device_ids) == 1:
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
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

        testset     = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
        testloader  = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
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

    # if args.arch == 'googlenet':
    #     compress_rate = compute_ratio_googlenet(args, print_logger=print_logger)
    # elif args.arch == 'densenet_40':
    #     print(f'no global pruning')
    # else:
    #     # compress_rate = compute_ratio(args, print_logger=print_logger) #pr1
    #     compress_rate = compute_ratio_iterative(args, print_logger=print_logger) #pr2
    #     # compress_rate = compute_ratio_nn(args, print_logger=print_logger) #pr3
    #
    #     if args.arch == 'vgg_16_bn':
    #         compress_rate = compress_rate + [0.]


    # compress_rate = compute_ratio_energy(net, args, print_logger=print_logger, trainloader=trainloader, device=device)

    # #VGG-16
    # compress_rate = [0.446988, 0.289265, 0.195314, 0.052018, 0.019805, 0.150464, 0.332670, 0.467028, 0.497656, 0.604502, 0.487944, 0.666465, 0.138802]  # from eagleeye것
    # compress_rate = [0.051752, 0.080923, 0.147122, 0.126646, 0.234942, 0.196759, 0.185370, 0.422472, 0.382292, 0.459086, 0.521959, 0.573066, 0.155299] #from energy
    compress_rate = [0.390625, 0.0,      0.718750, 0.281250, 0.062500, 0.343750, 0.792968, 0.023437, 0.363281, 0.812500, 0.300781, 0.365234, 0.533203] #내 concept으로 계산한 것

    # ResNet-56
    # compress_rate = [0.1875, 0.0625, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0625, 0.25, 0.0,
    #  0.03125, 0.03125, 0.0, 0.0, 0.0, 0.03125, 0.03125, 0.0, 0.0, 0.03125, 0.0, 0.0, 0.03125, 0.0, 0.0, 0.03125, 0.0,
    #  0.0, 0.78125, 0.78125, 0.859375, 0.78125, 0.84375, 0.75, 0.8125, 0.796875, 0.859375, 0.75, 0.890625, 0.671875,
    #  0.84375, 0.71875, 0.78125, 0.703125, 0.765625, 0.609375] #원래 꺼 (from hrank)
    # compress_rate = [0.1875, 0.0625, 0.0625, 0.0, 0.0625, 0.0, 0.25, 0.0, 0.3125, 0.0, 0.4375, 0.0, 0.3125, 0.0, 0.4375, 0.0, 0.25,
    #  0.5625, 0.25, 0.25, 0.375, 0.4375, 0.78125, 0.21875, 0.875, 0.25, 0.84375, 0.25, 0.875, 0.09375, 0.90625, 0.3125,
    #  0.875, 0.28125, 0.84375, 0.1875, 0.75, 0.5625, 0.171875, 0.40625, 0.59375, 0.4375, 0.71875, 0.34375, 0.703125,
    #  0.328125, 0.578125, 0.453125, 0.4375, 0.46875, 0.234375, 0.46875, 0.1875, 0.078125, 0.015625] #내 concept으로 계산한 것

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

                if pruning:
                    m.grad_mask(cov_id)

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

    # print(convcfg)
    for cov_id in range(args.start_cov, len(convcfg)): #0에서부터 11까지 (즉 1에서 12번의 rank_conv 방문)
        # Load pruned_checkpoint
        print_logger.info("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id))
        m.layer_mask(cov_id + 1, resume=args.resume_mask, param_per_cov=param_per_cov_dic[args.arch], arch=args.arch)

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

        if cov_id == 0:
            pruned_checkpoint = torch.load(args.resume, map_location=device) #load pretrained full-model

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

            net.load_state_dict(new_state_dict)#'''
        else:
            if args.arch=='resnet_50':
                skip_list=[1,5,8,11,15,18,21,24,28,31,34,37,40,43,47,50,53]
                if cov_id+1 not in skip_list:
                    continue
                else:
                    pruned_checkpoint = torch.load(
                        args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(skip_list[skip_list.index(cov_id+1)-1]) + '.pt')
                    net.load_state_dict(pruned_checkpoint['state_dict'])
            else:
                if len(args.gpu) == 1:
                    pruned_checkpoint = torch.load(args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) + '.pt', map_location='cuda:0')
                else:
                    pruned_checkpoint = torch.load(args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) + '.pt')
                net.load_state_dict(pruned_checkpoint['state_dict'])

        best_acc=0.
        for epoch in range(0, args.epochs):
            train(epoch, cov_id + 1, optimizer, scheduler)
            scheduler.step()
            test(epoch, cov_id + 1, optimizer, scheduler)

