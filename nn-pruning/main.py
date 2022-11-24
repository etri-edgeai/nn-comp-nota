
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

    # Model
    print_logger.info('==> Building model..')
    net = eval(args.arch)(compress_rate=compress_rate)
    net = net.to(device)

    compress_rate = compute_ratio_energy(net, args, print_logger=print_logger, trainloader=trainloader, device=device)

    # compress_rate = [0.446988357583512, 0.28926514619837845, 0.1953148126044888, 0.052018662068368256,
    #                  0.01980565049656292, 0.15046404292195228, 0.33267048810557215, 0.46702857773833417,
    #                  0.49765683784607306, 0.6045027284494889, 0.48794417067817586, 0.6664655723084874,
    #                  0.13880225635771748]  # from eagleeye
    # compress_rate = [0.05175237, 0.0809230, 0.147122, 0.1266462, 0.2349422, 0.1967598, 0.18537, 0.4224726, 0.3822923, 0.4590861, 0.5219590, 0.573066, 0.1552993] #from energy
    # compress_rate=[0.21875, 0.0, 0.015625, 0.0, 0.4375, 0.4375, 0.41796875, 0.99609375, 0.974609375, 0.962890625, 0.9765625, 0.984375, 0.8]

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

