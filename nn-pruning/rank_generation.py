import torch
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import os
import argparse

import data.imagenet as imagenet
from models import *
from utils.utils import progress_bar
from utils.parser import get_args
import utils
import numpy as np

def rank_generation(net, args, trainloader, device, energy=False):
    global feature_result
    global total
    global batch_count
    criterion = nn.CrossEntropyLoss()
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)
    batch_count = 0

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def get_feature_hook
    def get_feature_hook_densenet
    def get_feature_hook_googlenet
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def get_feature_hook(self, input, output):
        global feature_result
        global total
        global batch_count
        a = output.shape[0] #batch
        b = output.shape[1] #filter

        # ########### Hrank (from CVPR2020)
        # c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])
        # c = c.view(a, -1).float()
        # c = c.sum(0)
        #
        # feature_result = feature_result * total + c
        # total = total + a
        # feature_result = feature_result / total

        # ########## Seul-Ki's approach (version1)
        # c = torch.tensor([torch.svd(output[:, i, :, :])[1].sum() for i in range(b)])
        #
        # feature_result = feature_result * total + c
        # total = total + a
        # feature_result = feature_result / total

        ########## Seul-Ki's approach (version2)
        c = torch.tensor([torch.svd(output.view(a, b, -1)[:, i, :])[1].sum() for i in range(b)])

        feature_result = feature_result * total + c
        total = total + a
        feature_result = feature_result / total

        # ########## Seul-Ki's approach (version3) / 모든 배치를 누적한 다음에 마지막에 한번만 svd 계산
        # if total == 0:
        #     feature_result = output.view(a, b, -1)
        # else:
        #     feature_result = torch.cat((feature_result, output.view(a, b, -1)), dim=0)
        #
        # total = total + a
        # batch_count += 1
        #
        # if batch_count == args.limit:
        #     feature_result = torch.tensor([torch.svd(feature_result[:, i, :])[1].sum() for i in range(b)])
        #     feature_result /= total
        #     batch_count = 0

    def get_feature_hook_densenet(self, input, output):
        global feature_result
        global total
        global batch_count
        a = output.shape[0] #batch
        b = output.shape[1] #filter

        # ########### Hrank (from CVPR2020)
        # c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b-12,b)])
        # c = c.view(a, -1).float()
        # c = c.sum(0)
        #
        # feature_result = feature_result * total + c
        # total = total + a
        # feature_result = feature_result / total

        ########### Seul-Ki's approach (version2)
        c = torch.tensor([torch.svd(output.view(a, b, -1)[:, i, :])[1].sum() for i in range(b-12, b)])

        feature_result = feature_result * total + c
        total = total + a
        feature_result = feature_result / total

    def get_feature_hook_googlenet(self, input, output):
        global feature_result
        global total
        global batch_count
        a = output.shape[0] #batch
        b = output.shape[1] #filter

        # ########### Hrank (from CVPR2020)
        # c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b-12,b)])
        # c = c.view(a, -1).float()
        # c = c.sum(0)

        ########### Seul-Ki's approach (version2)
        c = torch.tensor([torch.svd(output.view(a, b, -1)[:, i, :])[1].sum() for i in range(b-12, b)])

        feature_result = feature_result * total + c
        total = total + a
        feature_result = feature_result / total


    def test():
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        limit = args.limit

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                if batch_idx >= limit:  # use the first 6 batches to estimate the rank.
                   break
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''



    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model module
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if args.arch in ['vgg_16_bn', 'resnet_56', 'resnet_110', 'googlenet']:
        cnt = 0
        for name, cov_layer in net.named_modules():  # vgg_16_bn, resnet_56, resnet_110, googlenet
            if isinstance(cov_layer, nn.BatchNorm2d):
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()

                if not os.path.isdir('rank_conv/' + args.arch + '_limit%d' % (args.limit)):
                    os.mkdir('rank_conv/' + args.arch + '_limit%d' % (args.limit))
                np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + args.save_name + str(cnt + 1) + '.npy',
                        feature_result.numpy())

                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

        if args.arch in ['vgg_16_bn']:  # dense layer
            for name, dense_layer in net.named_modules():  # vgg_16_bn
                if isinstance(dense_layer, nn.BatchNorm1d):

                    handler = dense_layer.register_forward_hook(get_feature_hook)
                    test()
                    handler.remove()

                    if not os.path.isdir('rank_conv/' + args.arch + '_limit%d' % (args.limit)):
                        os.mkdir('rank_conv/' + args.arch + '_limit%d' % (args.limit))
                    np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + args.save_name + str(cnt + 1) + '.npy',
                            feature_result.numpy())

                    feature_result = torch.tensor(0.)
                    total = torch.tensor(0.)

    elif args.arch =='resnet_50':

        cov_layer = eval('net.maxpool')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
        np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + args.save_name + '%d' % (1) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # ResNet50 per bottleneck
        cnt=1
        for i in range(4):
            block = eval('net.layer%d' % (i + 1))
            for j in range(net.num_blocks[i]):
                cov_layer = block[j].bn1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + args.save_name + '%d'%(cnt+1)+'.npy', feature_result.numpy())
                cnt+=1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].bn2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + args.save_name + '%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].bn3
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                if j==0:
                    np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + args.save_name + '%d' % (cnt + 1) + '.npy',
                            feature_result.numpy())#shortcut conv
                    cnt += 1
                np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + args.save_name + '%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())#conv3
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

    elif args.arch=='densenet_40':

        if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))

        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # Densenet per block & transition
        for i in range(3):
            dense = eval('net.dense%d' % (i + 1))
            for j in range(12):
                cov_layer = dense[j].bn1
                if j==0:
                    handler = cov_layer.register_forward_hook(get_feature_hook)
                else:
                    handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
                test()
                handler.remove()

                np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + args.save_name + '%d'%(13*i+j+1)+'.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

            if i<2:
                trans=eval('net.trans%d' % (i + 1))
                cov_layer = trans.bn1

                handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
                test()
                handler.remove()

                np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + args.save_name + '%d' % (13 * (i+1)) + '.npy', feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)#'''

        cov_layer = net.bn
        handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
        test()
        handler.remove()
        np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + args.save_name + '%d' % (39) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

def rank_generation_energy(net, args, trainloader, device, energy=False):
    global feature_result, total, batch_count
    criterion = nn.CrossEntropyLoss()
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)
    batch_count = 0
    rank = []

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def get_feature_hook
    def get_feature_hook_densenet
    def get_feature_hook_googlenet
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def get_feature_hook(self, input, output):
        global feature_result, total, batch_count
        a = output.shape[0]  # batch
        b = output.shape[1]  # filter
        import numpy as np

        c = []
        for i in range(b):
            x = torch.svd(output.view(a, b, -1)[:, i, :])[1]
            f = (torch.pow(x, 2) / (torch.pow(x, 2).sum()))
            E = (-1 / torch.log(torch.tensor(len(f[f.nonzero(as_tuple=False)])).float())) * (
                    f[f.nonzero(as_tuple=False)] * torch.log(f[f.nonzero(as_tuple=False)])).sum()
            c.append(E)

        feature_result = feature_result * total + torch.tensor(c)
        total = total + a
        feature_result = feature_result / total

    def get_feature_hook_densenet(self, input, output):
        global feature_result, total, batch_count
        a = output.shape[0] #batch
        b = output.shape[1] #filter

        # ########### Hrank (from CVPR2020)
        # c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b-12,b)])
        # c = c.view(a, -1).float()
        # c = c.sum(0)
        #
        # feature_result = feature_result * total + c
        # total = total + a
        # feature_result = feature_result / total

        ########### Seul-Ki's approach (version2)
        c = torch.tensor([torch.svd(output.view(a, b, -1)[:, i, :])[1].sum() for i in range(b-12, b)])

        feature_result = feature_result * total + c
        total = total + a
        feature_result = feature_result / total

    def get_feature_hook_googlenet(self, input, output):
        global feature_result
        global total
        global batch_count
        a = output.shape[0] #batch
        b = output.shape[1] #filter

        # ########### Hrank (from CVPR2020)
        # c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b-12,b)])
        # c = c.view(a, -1).float()
        # c = c.sum(0)

        ########### Seul-Ki's approach (version2)
        c = torch.tensor([torch.svd(output.view(a, b, -1)[:, i, :])[1].sum() for i in range(b-12, b)])

        feature_result = feature_result * total + c
        total = total + a
        feature_result = feature_result / total


    def test():
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        limit = args.limit

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                if batch_idx >= limit:  # use the first 6 batches to estimate the rank.
                   break
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''



    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model module
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if args.arch in ['vgg_16_bn', 'resnet_56', 'resnet_110', 'googlenet']:
        for name, cov_layer in net.named_modules():  # vgg_16_bn, resnet_56, resnet_110, googlenet
            if isinstance(cov_layer, nn.BatchNorm2d):
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()

                rank.append(feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

        # if args.arch in ['vgg_16_bn']:  # dense layer
        #     for name, dense_layer in net.named_modules():  # vgg_16_bn
        #         if isinstance(dense_layer, nn.BatchNorm1d):
        #
        #             handler = dense_layer.register_forward_hook(get_feature_hook)
        #             test()
        #             handler.remove()
        #
        #             if not os.path.isdir('rank_conv/' + args.arch + '_limit%d' % (args.limit)):
        #                 os.mkdir('rank_conv/' + args.arch + '_limit%d' % (args.limit))
        #             np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + args.save_name + str(cnt + 1) + '.npy',
        #                     feature_result.numpy())
        #
        #             feature_result = torch.tensor(0.)
        #             total = torch.tensor(0.)

        return rank

    elif args.arch =='resnet_50':

        cov_layer = eval('net.maxpool')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        rank.append(feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # ResNet50 per bottleneck
        cnt=1
        for i in range(4):
            block = eval('net.layer%d' % (i + 1))
            for j in range(net.num_blocks[i]):
                cov_layer = block[j].bn1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + args.save_name + '%d'%(cnt+1)+'.npy', feature_result.numpy())
                cnt+=1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].bn2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + args.save_name + '%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].bn3
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                if j==0:
                    np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + args.save_name + '%d' % (cnt + 1) + '.npy',
                            feature_result.numpy())#shortcut conv
                    cnt += 1
                np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + args.save_name + '%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())#conv3
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

        return rank

    elif args.arch=='densenet_40':
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # Densenet per block & transition
        for i in range(3):
            dense = eval('net.dense%d' % (i + 1))
            for j in range(12):
                cov_layer = dense[j].bn1
                if j==0:
                    handler = cov_layer.register_forward_hook(get_feature_hook)
                else:
                    handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
                test()
                handler.remove()

                rank.append(feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

            if i<2:
                trans=eval('net.trans%d' % (i + 1))
                cov_layer = trans.bn1

                handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
                test()
                handler.remove()

                rank.append(feature_result.numpy())
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)#'''

        cov_layer = net.bn
        handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
        test()
        handler.remove()
        rank.append(feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        return rank

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Rank extraction')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/ssd7/skyeom/data',
        help='dataset path')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=('cifar10', 'imagenet'),
        help='dataset')
    parser.add_argument(
        '--job_dir',
        type=str,
        default='result/tmp',
        help='The directory where the summaries will be stored.')
    parser.add_argument(
        '--arch',
        type=str,
        default='googlenet',
        choices=('resnet_50', 'vgg_16_bn', 'resnet_56', 'resnet_110', 'densenet_40', 'googlenet'),
        help='The architecture to prune')
    parser.add_argument(
        '--resume',
        type=str,
        default='./checkpoints/',
        help='load the model from the specified checkpoint')
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='The num of batch to get rank.')
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
        '--start_idx',
        type=int,
        default=0,
        help='The index of conv to start extract rank.')
    parser.add_argument(
        '--gpu',
        type=str,
        default='6',
        help='Select gpu to use')
    parser.add_argument(
        '--adjust_ckpt',
        action='store_true',
        help='adjust ckpt from pruned checkpoint')
    parser.add_argument(
        '--compress_rate',
        type=str,
        default=None,
        help='compress rate of each conv')
    parser.add_argument(
        '--save_name',
        type=str,
        default='/rank_conv_w',
        help='npy file name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    args.resume = args.resume + args.arch + '.pt'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    if len(args.gpu) == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)  ##for cpu
    if args.gpu:
        torch.cuda.manual_seed(args.seed)  ##for gpu

    # Data
    print('==> Preparing data..')
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
        trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                                transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True,
                                                  num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    elif args.dataset == 'imagenet':
        data_tmp = imagenet.Data(args)
        trainloader = data_tmp.loader_train
        testloader = data_tmp.loader_test

    if args.compress_rate:
        import re

        cprate_str = args.compress_rate
        cprate_str_list = cprate_str.split('+')
        pat_cprate = re.compile(r'\d+\.\d*')
        pat_num = re.compile(r'\*\d+')
        cprate = []
        for x in cprate_str_list:
            num = 1
            find_num = re.findall(pat_num, x)
            if find_num:
                assert len(find_num) == 1
                num = int(find_num[0].replace('*', ''))
            find_cprate = re.findall(pat_cprate, x)
            assert len(find_cprate) == 1
            cprate += [float(find_cprate[0])] * num

        compress_rate = cprate
    else:
        default_cprate = {
            'vgg_16_bn': [0.7] * 7 + [0.1] * 6,
            'densenet_40': [0.0] + [0.1] * 6 + [0.7] * 6 + [0.0] + [0.1] * 6 + [0.7] * 6 + [0.0] + [0.1] * 6 + [
                0.7] * 5 + [0.0],
            'googlenet': [0.10] + [0.7] + [0.5] + [0.8] * 4 + [0.5] + [0.6] * 2,
            'resnet_50': [0.2] + [0.8] * 10 + [0.8] * 13 + [0.55] * 19 + [0.45] * 10,
            'resnet_56': [0.1] + [0.60] * 35 + [0.0] * 2 + [0.6] * 6 + [0.4] * 3 + [0.1] + [0.4] + [0.1] + [0.4] + [
                0.1] + [0.4] + [0.1] + [0.4],
            'resnet_110': [0.1] + [0.40] * 36 + [0.40] * 36 + [0.4] * 36
        }
        compress_rate = default_cprate[args.arch]

    # Model
    print('==> Building model..')
    print(compress_rate)
    net = eval(args.arch)(compress_rate=compress_rate)
    net = net.to(device)
    if args.arch == "resnet_50":
        summary(net, (3, 224, 224))
    else:
        summary(net, (3, 32, 32))

    'GPU Check'
    if len(args.gpu) > 1 and torch.cuda.is_available():
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        net = torch.nn.DataParallel(net, device_ids=device_id)

    'Load checkpoint'
    if args.resume:  # Load checkpoint (i.e. pretrained full model).
        print('==> Resuming from checkpoint..')

        if args.arch == "resnet_50":
            checkpoint = torch.load(args.resume + "h", map_location='cuda:0')
            net.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load(args.resume, map_location='cuda:0')

            from collections import OrderedDict

            new_state_dict = OrderedDict()
            if args.adjust_ckpt:
                for k, v in checkpoint.items():
                    new_state_dict[k.replace('module.', '')] = v
            else:
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k.replace('module.', '')] = v
            net.load_state_dict(new_state_dict)

    rank_generation(net, args=args, trainloader=trainloader, device=device, energy=True)
    print(f'finish')