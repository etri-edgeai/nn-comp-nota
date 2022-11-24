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
from rank_generation import rank_generation_energy

def compute_ratio_energy(net, args, print_logger=None, trainloader=None, device='cuda:0'):

    #load pretrained model
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

    compress_rate = args.compress_rate

    device_ids =list(map(int, args.gpu.split(',')))

    rank = {}
    all_filters = {}
    compressed_filters = {}
    rank_sum = {}
    pruning_filter = 0
    remained_filters = 0
    tot_filter = 0
    print(len(convcfg))

    rank = rank_generation_energy(net, args = args, trainloader = trainloader, device = device, energy=True)

    for cov_id in range(args.start_cov, len(convcfg)):
        # Load pruned_checkpoint
        print_logger.info("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id))

        # rank[cov_id] /= np.linalg.norm(rank[cov_id], ord=1)

        tot_filter += rank[cov_id].__len__()
        all_filters[cov_id] = rank[cov_id].__len__()
        compressed_filters[cov_id] = rank[cov_id].__len__()
        rank_sum[cov_id] = rank[cov_id].sum()

    def lowest_ranking_filters(filter_rank, num):
        data = []
        for i in range(len(filter_rank)):
            for j in range(len(filter_rank[i])):
                data.append((i, j, filter_rank[i][j]))
                # print(len(filter_rank[i]))

        # print(data)
        return nsmallest(num, data, itemgetter(2))

    new_compress_rate = list(rank_sum.values())
    new_compress_rate = new_compress_rate / max(new_compress_rate) * args.total_pr

    # print(f'old compress rate: {compress_rate}')
    print(f'energy based compress_rate: {new_compress_rate}')
    return new_compress_rate



def compute_ratio_iterative(args, print_logger=None):
    if len(args.gpu)==1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if args.compress_rate:
    #     import re
    #     cprate_str=args.compress_rate
    #     cprate_str_list=cprate_str.split('+')
    #     pat_cprate = re.compile(r'\d+\.\d*')
    #     pat_num = re.compile(r'\*\d+')
    #     cprate=[]
    #     for x in cprate_str_list:
    #         num=1
    #         find_num=re.findall(pat_num,x)
    #         if find_num:
    #             assert len(find_num) == 1
    #             num=int(find_num[0].replace('*',''))
    #         find_cprate = re.findall(pat_cprate, x)
    #         assert len(find_cprate)==1
    #         cprate+=[float(find_cprate[0])]*num
    #
    #     compress_rate=cprate
    compress_rate = args.compress_rate

    device_ids =list(map(int, args.gpu.split(',')))
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
    pruning_filter = 0
    remained_filters = 0
    tot_filter = 0
    print(len(convcfg))

    for cov_id in range(args.start_cov, len(convcfg)):
        # Load pruned_checkpoint
        print_logger.info("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id))

        prefix = "rank_conv/" + args.arch + "_limit"+str(args.limit)+"/rank_conv_w"
        subfix = ".npy"

        rank[cov_id] = np.load(prefix + str(cov_id + 1) + subfix)
        rank[cov_id] /= np.linalg.norm(rank[cov_id], ord=1)

        # pruned_num = int(compress_rate[cov_id] * rank[cov_id].__len__())
        pruning_filter += int(compress_rate[cov_id] * rank[cov_id].__len__())
        # remained_filters += len(np.argsort(rank[cov_id])[pruned_num:])
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

    if args.pr:
        num_filters_to_prune_per_iteration = int(tot_filter * args.pruning_step)
        iterations = int(float(tot_filter) / num_filters_to_prune_per_iteration)
        iterations = int(iterations * args.pr)
    else:
        num_filters_to_prune_per_iteration = int(pruning_filter * args.pruning_step)
        iterations = int(float(pruning_filter) / num_filters_to_prune_per_iteration)

    def normalized_rank(rank, cp=None):
        for cov_id in range(args.start_cov, len(convcfg)):
            if rank[cov_id].__len__() != cp[cov_id]: #if there exist pruned filters in the rank[cov_id]
                ind = np.argsort(rank[cov_id])[-cp[cov_id]:]  # preserved filter id
                rank[cov_id][ind] /= np.linalg.norm(rank[cov_id][ind], ord=1)

        return rank

    import copy
    for kk in range(iterations):
        # compressed_filters = all_filters
        compressed_filters = copy.deepcopy(all_filters)
        filter_to_prune = lowest_ranking_filters(rank, num_filters_to_prune_per_iteration * (kk + 1))

        for (l, _, _) in filter_to_prune:
            compressed_filters[l] -= 1

        rank = normalized_rank(rank, cp = compressed_filters)


    compression_ratio = np.zeros(len(rank))
    for i in range(len(compression_ratio)):
        compression_ratio[i] = 1.0 - (compressed_filters[i] / all_filters[i])

    new_compress_rate = compression_ratio.tolist()

    # print(f'old compress rate: {compress_rate}')
    # print(f'new_compress_rate: {new_compress_rate}')
    return new_compress_rate

def compute_ratio_nn(args, print_logger=None):
    if len(args.gpu)==1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compress_rate = args.compress_rate

    device_ids =list(map(int, args.gpu.split(',')))
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
    pruning_filter = 0
    tot_filter = 0
    print(len(convcfg))
    for cov_id in range(args.start_cov, len(convcfg)):
        # Load pruned_checkpoint
        print_logger.info("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id))

        prefix = "rank_conv/" + args.arch + "_limit"+str(args.limit)+"/rank_conv_w"
        subfix = ".npy"

        rank[cov_id] = np.load(prefix + str(cov_id + 1) + subfix)
        rank[cov_id] /= np.linalg.norm(rank[cov_id], ord=1)
        rank[cov_id] *= rank[cov_id].__len__()

        # pruned_num = int(compress_rate[cov_id] * rank[cov_id].__len__())
        pruning_filter += int(compress_rate[cov_id] * rank[cov_id].__len__())
        # remained_filters += len(np.argsort(rank[cov_id])[pruned_num:])
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

    filter_to_prune = lowest_ranking_filters(rank, pruning_filter)

    for (l, _, _) in filter_to_prune:
        compressed_filters[l] -= 1

    compression_ratio = np.zeros(len(rank))
    for i in range(len(compression_ratio)):
        compression_ratio[i] = 1.0 - (compressed_filters[i] / all_filters[i])

    new_compress_rate = compression_ratio.tolist()

    # print(f'old compress rate: {compress_rate}')
    # print(f'new_compress_rate: {new_compress_rate}')
    return new_compress_rate

def compute_ratio(args, print_logger=None):
    if len(args.gpu)==1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compress_rate = args.compress_rate

    device_ids =list(map(int, args.gpu.split(',')))
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
    pruning_filter = 0
    tot_filter = 0
    print(len(convcfg))
    for cov_id in range(args.start_cov, len(convcfg)):
        # Load pruned_checkpoint
        print_logger.info("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id))

        prefix = "rank_conv/" + args.arch + "_limit"+str(args.limit)+"/rank_conv_w"
        subfix = ".npy"

        rank[cov_id] = np.load(prefix + str(cov_id + 1) + subfix)
        rank[cov_id] /= np.linalg.norm(rank[cov_id], ord=1)

        # pruned_num = int(compress_rate[cov_id] * rank[cov_id].__len__())
        pruning_filter += int(compress_rate[cov_id] * rank[cov_id].__len__())
        # remained_filters += len(np.argsort(rank[cov_id])[pruned_num:])
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

    filter_to_prune = lowest_ranking_filters(rank, pruning_filter)

    for (l, _, _) in filter_to_prune:
        compressed_filters[l] -= 1

    compression_ratio = np.zeros(len(rank))
    for i in range(len(compression_ratio)):
        compression_ratio[i] = 1.0 - (compressed_filters[i] / all_filters[i])

    new_compress_rate = compression_ratio.tolist()

    # print(f'old compress rate: {compress_rate}')
    # print(f'new_compress_rate: {new_compress_rate}')
    return new_compress_rate

if __name__ == "__main__":

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
        choices=('cifar10', 'imagenet'),
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
        default='5',
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
        default='[0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4] ',
        # for resnet_56
        # default= '[0.1]+[0.40]*36+[0.40]*36+[0.4]*36', #for resnet_110
        # default= '[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0]', #for densenet
        # default='[0.95]+[0.5]*6+[0.9]*4+[0.8]+[0.0] ', #for vgg_16
        help='compress rate of each conv')
    parser.add_argument(
        '--arch',
        type=str,
        default='vgg_16_bn',
        choices=('resnet_50', 'vgg_16_bn', 'resnet_56', 'resnet_110', 'densenet_40', 'googlenet'),
        help='The architecture to prune')
    parser.add_argument(
        '--pruning_step',
        type=float,
        default='0.01',
        help='compress rate of each conv')
    parser.add_argument(
        '--pr',
        type=float,
        default=None,
        help='pruning ratio')
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='The num of batch to get rank.')

    args = parser.parse_args()
    args.resume = args.resume + args.arch + '.pt'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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

        args.compress_rate=cprate

    print_logger = utils.get_logger(os.path.join(args.job_dir, "cp_ratio_logger.log"))

    # compress_rate = compute_ratio(args, print_logger=print_logger)
    # print(compress_rate)
    # compress_rate = compute_ratio_nn(args, print_logger=print_logger)
    # print(compress_rate)
    compress_rate = compute_ratio_iterative(args, print_logger=print_logger)
    print(compress_rate)