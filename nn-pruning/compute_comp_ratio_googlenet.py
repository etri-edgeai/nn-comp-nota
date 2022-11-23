
from heapq import nsmallest
from operator import itemgetter

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


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/ssd7/skyeom/data',
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
    default='5,10',
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
    default='3, 4',
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
    default='[0.10]+[0.8]*5+[0.85]+[0.8]*3',
    help='compress rate of each conv')
parser.add_argument(
    '--arch',
    type=str,
    default='googlenet',
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'),
    help='The architecture to prune')

def compute_ratio(args, print_logger=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if len(args.gpu)==1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.compress_rate:
        import re
        cprate_str=args.compress_rate
        cprate_str_list=cprate_str.split('+')
        pat_cprate = re.compile(r'\d+\.\d*')
        pat_num = re.compile(r'\*\d+')
        cprate=[]
        for x in cprate_str_list:
            num=1
            find_num=re.findall(pat_num,x)
            if find_num:
                assert len(find_num) == 1
                num=int(find_num[0].replace('*',''))
            find_cprate = re.findall(pat_cprate, x)
            assert len(find_cprate)==1
            cprate+=[float(find_cprate[0])]*num

        compress_rate=cprate

    device_ids=list(map(int, args.gpu.split(',')))
    net = eval(args.arch)(compress_rate=compress_rate)
    net = net.to(device)

    if len(args.gpu)>1 and torch.cuda.is_available():
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        net = torch.nn.DataParallel(net, device_ids=device_id)

    cudnn.benchmark = True

    if len(args.gpu)>1:
        convcfg = net.module.covcfg
    else:
        convcfg = net.covcfg

    rank = {}
    all_filters = {}
    compressed_filters = {}
    pruned_filters = 0
    tot_filter_hrank = 0
    tot_filter = 0
    prefix = "rank_conv/original_hrank_save/" + args.arch + "/"

    arr = sorted(os.listdir(prefix))

    for cov_id in range(args.start_cov, len(convcfg)): #기존 hrank 에서 다루던 googlenet 의 layer 수 (=10)
        # Load pruned_checkpoint
        if cov_id == 0:
            hrank = np.load(prefix + arr[cov_id + 1])
            pruned_filters += int(compress_rate[cov_id] * hrank.__len__())
            tot_filter_hrank += hrank.__len__()
        else:
            ind = [n for n, l in enumerate(arr) if l.startswith('rank_conv' + str(cov_id + 1))]

            for i in range(len(ind)):
                hrank = np.load(prefix + arr[ind[i]])
                pruned_filters += int(compress_rate[cov_id] * hrank.__len__())
                tot_filter_hrank += hrank.__len__()

    print(pruned_filters)
    print(f'tot_filter_hrank: {tot_filter_hrank}')

    for cov_id in range(args.start_cov, 64): #googlenet 전체 conv layer 수 (새로 개선)
        # Load pruned_checkpoint
        print_logger.info("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id))

        prefix = "rank_conv/" + args.arch + "/"
        subfix = ".npy"

        rank[cov_id] = np.load(prefix + str(cov_id + 1) + subfix)
        rank[cov_id] /= np.linalg.norm(rank[cov_id], ord=1)

        tot_filter += rank[cov_id].__len__()
        all_filters[cov_id] = rank[cov_id].__len__()
        compressed_filters[cov_id] = rank[cov_id].__len__()

    def lowest_ranking_filters(filter_rank, num):
        data = []
        for i in range(len(filter_rank)):
            for j in range(len(filter_rank[i])):
                data.append((i, j, filter_rank[i][j]))
                # print(len(filter_rank[i]))

        # print(data)
        return nsmallest(num, data, itemgetter(2))

    filter_to_prune = lowest_ranking_filters(rank, pruned_filters)

    for (l, _, _) in filter_to_prune:
        compressed_filters[l] -= 1

    compression_ratio = np.zeros(len(rank))
    for i in range(len(compression_ratio)):
        compression_ratio[i] = 1.0 - (compressed_filters[i] / all_filters[i])

    new_compress_rate = compression_ratio.tolist()

    print(f'old compress rate: {compress_rate}')
    print(f'new_compress_rate: {new_compress_rate}')
    return new_compress_rate


if __name__ == "__main__":
    args = parser.parse_args()
    args.resume = args.resume + args.arch + '.pt'

    print_logger = utils.get_logger(os.path.join(args.job_dir, "cp_ratio_logger.log"))

    compress_rate = compute_ratio(args, print_logger=print_logger)
    print(compress_rate)