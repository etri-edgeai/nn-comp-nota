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
    default='[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0]',
    help='compress rate of each conv')
parser.add_argument(
    '--arch',
    type=str,
    default='densenet_40',
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

        prefix = "rank_conv/" + args.arch + "/rank_conv_w"
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

    print(f'old compress rate: {compress_rate}')
    print(f'new_compress_rate: {new_compress_rate}')
    return new_compress_rate


if __name__ == "__main__":
    args = parser.parse_args()
    args.resume = args.resume + args.arch + '.pt'

    print_logger = utils.get_logger(os.path.join(args.job_dir, "cp_ratio_logger.log"))

    compress_rate = compute_ratio(args, print_logger=print_logger)
    print(compress_rate)

# from heapq import nsmallest
# from operator import itemgetter

# import torch
# import torch.optim as optim
# import torch.backends.cudnn as cudnn

# import torchvision
# import torchvision.transforms as transforms

# import os
# import argparse

# from data import imagenet
# from models import *
# from utils import progress_bar
# from mask import *
# import utils


# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument(
#     '--data_dir',
#     type=str,
#     default='/ssd7/skyeom/data',
#     help='dataset path')
# parser.add_argument(
#     '--dataset',
#     type=str,
#     default='cifar10',
#     choices=('cifar10','imagenet'),
#     help='dataset')
# parser.add_argument(
#     '--lr',
#     default=0.01,
#     type=float,
#     help='initial learning rate')
# parser.add_argument(
#     '--lr_decay_step',
#     default='5,10',
#     type=str,
#     help='learning rate decay step')
# parser.add_argument(
#     '--resume',
#     type=str,
#     default='./checkpoints/',
#     help='load the model from the specified checkpoint')
# parser.add_argument(
#     '--resume_mask',
#     type=str,
#     default=None,
#     help='mask loading')
# parser.add_argument(
#     '--gpu',
#     type=str,
#     default='1, 3',
#     help='Select gpu to use')
# parser.add_argument(
#     '--job_dir',
#     type=str,
#     default='./result/temp',
#     help='The directory where the summaries will be stored.')
# parser.add_argument(
#     '--epochs',
#     type=int,
#     default=15,
#     help='The num of epochs to train.')
# parser.add_argument(
#     '--train_batch_size',
#     type=int,
#     default=128,
#     help='Batch size for training.')
# parser.add_argument(
#     '--eval_batch_size',
#     type=int,
#     default=100,
#     help='Batch size for validation.')
# parser.add_argument(
#     '--start_cov',
#     type=int,
#     default=0,
#     help='The num of conv to start prune')
# parser.add_argument(
#     '--compress_rate',
#     type=str,
#     default='[0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]',
#     help='compress rate of each conv')
# parser.add_argument(
#     '--arch',
#     type=str,
#     default='resnet_56',
#     choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'),
#     help='The architecture to prune')

# def compute_ratio(args, print_logger=None):
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#     if len(args.gpu)==1:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if args.compress_rate:
#         import re
#         cprate_str=args.compress_rate
#         cprate_str_list=cprate_str.split('+')
#         pat_cprate = re.compile(r'\d+\.\d*')
#         pat_num = re.compile(r'\*\d+')
#         cprate=[]
#         for x in cprate_str_list:
#             num=1
#             find_num=re.findall(pat_num,x)
#             if find_num:
#                 assert len(find_num) == 1
#                 num=int(find_num[0].replace('*',''))
#             find_cprate = re.findall(pat_cprate, x)
#             assert len(find_cprate)==1
#             cprate+=[float(find_cprate[0])]*num

#         compress_rate=cprate

#     device_ids=list(map(int, args.gpu.split(',')))
#     net = eval(args.arch)(compress_rate=compress_rate)
#     net = net.to(device)

#     if len(args.gpu)>1 and torch.cuda.is_available():
#         device_id = []
#         for i in range((len(args.gpu) + 1) // 2):
#             device_id.append(i)
#         net = torch.nn.DataParallel(net, device_ids=device_id)

#     cudnn.benchmark = True

#     if len(args.gpu)>1:
#         convcfg = net.module.covcfg
#     else:
#         convcfg = net.covcfg

#     rank = {}
#     all_filters = {}
#     compressed_filters = {}
#     remained_filters = 0
#     tot_filter = 0
#     for cov_id in range(args.start_cov, len(convcfg)):
#         # Load pruned_checkpoint
#         print_logger.info("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id))

#         prefix = "rank_conv/" + args.arch + "/rank_conv_skyeom"
#         subfix = ".npy"

#         rank[cov_id] = np.load(prefix + str(cov_id + 1) + subfix)
#         rank[cov_id] /= np.linalg.norm(rank[cov_id], ord=1)

#         pruned_num = int(compress_rate[cov_id] * rank[cov_id].__len__())
#         remained_filters += len(np.argsort(rank[cov_id])[pruned_num:])
#         tot_filter += len(rank[cov_id])
#         all_filters[cov_id] = rank[cov_id].__len__()
#         compressed_filters[cov_id] = rank[cov_id].__len__()

#     def lowest_ranking_filters(filter_rank, num):
#         data = []
#         for i in range(len(filter_rank)):
#             for j in range(len(filter_rank[i])):
#                 data.append((i, j, filter_rank[i][j]))
#                 # print(len(filter_rank[i]))

#         # print(data)
#         return nsmallest(num, data, itemgetter(2))

#     filter_to_prune = lowest_ranking_filters(rank, tot_filter - remained_filters)

#     for (l, _, _) in filter_to_prune:
#         compressed_filters[l] -= 1

#     compression_ratio = np.zeros(len(rank))
#     for i in range(len(compression_ratio)):
#         compression_ratio[i] = 1.0 - (compressed_filters[i] / all_filters[i])

#     new_compress_rate = compression_ratio.tolist()

#     print(f'old compress rate: {compress_rate}')
#     print(f'new_compress_rate: {new_compress_rate}')
#     return new_compress_rate


# if __name__ == "__main__":
#     args = parser.parse_args()
#     args.resume = args.resume + args.arch + '.pt'

#     print_logger = utils.get_logger(os.path.join(args.job_dir, "cp_ratio_logger.log"))

#     compress_rate = compute_ratio(args, print_logger=print_logger)
#     print(compress_rate)

# from heapq import nsmallest
# from operator import itemgetter

# import torch
# import torch.optim as optim
# import torch.backends.cudnn as cudnn

# import torchvision
# import torchvision.transforms as transforms

# import os
# import argparse

# from data import imagenet
# from models import *
# from utils import progress_bar
# from mask import *
# import utils

# # 'vgg_16_bn'  : [0.7]*7+[0.1]*6,
# # 'densenet_40': [0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0],
# # [0.10]+[0.7]+[0.5]+[0.8]*4+[0.5]+[0.6]*2
# # 'googlenet'  : [0.10] + [0.7]*7 + [0.5]*7 + [0.8]*7 + [0.8]*7 + [0.8]*7 + [0.8]*7 + [0.5]*7 + [0.6]*7 + [0.6]*7
# # 'resnet_50'  : [0.2]+[0.8]*10+[0.8]*13+[0.55]*19+[0.45]*10,
# # 'resnet_56'  : [0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4],
# # 'resnet_110' : [0.1]+[0.40]*36+[0.40]*36+[0.4]*36
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--data_dir', type=str, default='./data', help='dataset path')
# parser.add_argument('--dataset',  type=str, default='cifar10',   choices=('cifar10','imagenet'), help='dataset')
# parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
# parser.add_argument('--lr_decay_step', default='5,10', type=str, help='learning rate decay step')
# parser.add_argument('--resume', type=str, default='./checkpoints/', help='load the model from the specified checkpoint')
# parser.add_argument('--resume_mask', type=str, default=None, help='mask loading')
# parser.add_argument('--gpu', type=str, default='1, 3', help='Select gpu to use')
# parser.add_argument('--job_dir', type=str, default='./result/temp', help='The directory where the summaries will be stored.')
# parser.add_argument('--epochs', type=int, default=15, help='The num of epochs to train.')
# parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size for training.')
# parser.add_argument('--eval_batch_size', type=int, default=100, help='Batch size for validation.')
# parser.add_argument('--start_cov', type=int, default=0, help='The num of conv to start prune')
# parser.add_argument('--compress_rate', type=str, default="[0.10] + [0.7]*7 + [0.5]*7 + [0.8]*7 + [0.8]*7 + [0.8]*7 + [0.8]*7 + [0.5]*7 + [0.6]*7 + [0.6]*7")
# parser.add_argument('--arch', type=str, default='googlenet', choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'), help='The architecture to prune')
# args = parser.parse_args()

# def compute_ratio(args, print_logger=None):
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#     if len(args.gpu)==1:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if args.compress_rate:
#         import re
#         cprate_str      = args.compress_rate
#         cprate_str_list = cprate_str.split('+')
#         pat_cprate      = re.compile(r'\d+\.\d*')
#         pat_num         = re.compile(r'\*\d+')
#         cprate = []
#         for x in cprate_str_list:
#             num = 1
#             find_num = re.findall(pat_num,x)
#             if find_num:
#                 assert len(find_num) == 1
#                 num = int(find_num[0].replace('*',''))
#             find_cprate = re.findall(pat_cprate, x)
#             assert len(find_cprate) == 1
#             cprate += [float(find_cprate[0])]*num

#         compress_rate = cprate
#     compress_rate = args.compress_rate
#     device_ids =list(map(int, args.gpu.split(',')))
#     net        = eval(args.arch)(compress_rate = compress_rate)
#     net        = net.to(device)
#     # for _, modules in net.named_modules():
#     #     print(modules)


#     if len(args.gpu)>1 and torch.cuda.is_available():
#         device_id = []
#         for i in range((len(args.gpu) + 1) // 2):
#             device_id.append(i)
#         net = torch.nn.DataParallel(net, device_ids=device_id)

#     cudnn.benchmark = True

#     if len(args.gpu)>1:
#         convcfg = net.module.covcfg
#     else:
#         convcfg = net.covcfg

#     rank               = {}
#     all_filters        = {}
#     compressed_filters = {}
#     remained_filters   = 0
#     tot_filter         = 0
#     # from torchsummary import summary
#     # summary(net, (3, 32, 32))
#     print(convcfg, args.start_cov)
#     for cov_id in range(args.start_cov, len(convcfg)):
#     # for cov_id in range (args.start_cov, 63):
#         # Load pruned_checkpoint
#         print_logger.info("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id))
#         prefix = "rank_conv/" + args.arch + "_limit1"
#         subfix = ".npy"

#         # 각 컨볼루션의 피처맵을 호출해서 rank의 키 값에 할당
#         rank[cov_id] = np.load(prefix + "/rank_conv_w" + str(cov_id + 1) + subfix)
#         # print(np.linalg.norm(rank[cov_id], ord = 1))
#         # 각 컨볼루션의 피처맵 랭크를 norm을 계산하여 정규화
#         rank[cov_id] = rank[cov_id] / np.linalg.norm(rank[cov_id], ord = 1)
#         # print(len(rank[cov_id]))

#         # 짤라내는 필터의 수 예를 들어서 64개의 필터인데 compress_rate가 0.8이면 0.8 * 64 = 8개가 남는 것
#         # 피처맵 nulcear norm을 argsort로 작은 순서부터 나열
#         # 목표는 큰 것부터 몇 개를 짜르는 것이므로 [prune_num : ]으로 뒤에 있는 것을 떼어내는 구조
#         pruned_num                 = int(compress_rate[cov_id] * rank[cov_id].__len__())
#         # print(compress_rate[cov_id])
#         # print(len(rank[cov_id]))
#         # print(pruned_num)
#         # print(rank)
#         # 0.7
#         # 12
#         # 8

#         # {0: array([0.01737489, 0.04998815, 0.03475241, 0.02194369, 0.03339726,
#         # 0.02043924, 0.02867828, 0.02856985, 0.07306752, 0.06947649,
#         # 0.04071735, 0.08303612, 0.06182968, 0.04806388, 0.04323768,
#         # 0.02952793, 0.07626414, 0.03245737, 0.04278338, 0.01763092,
#         # 0.0653312 , 0.03166654, 0.02994968, 0.01981639], dtype=float32)}

#         # print(rank[cov_id])
#         # [0.01737489 0.04998815 0.03475241 0.02194369 0.03339726 0.02043924
#         # 0.02867828 0.02856985 0.07306752 0.06947649 0.04071735 0.08303612
#         # 0.06182968 0.04806388 0.04323768 0.02952793 0.07626414 0.03245737
#         # 0.04278338 0.01763092 0.0653312  0.03166654 0.02994968 0.01981639]

#         # print(rank[cov_id][np.argsort(rank[cov_id])])
#         # [0.01737489 0.01763092 0.01981639 0.02043924 0.02194369 0.02856985
#         # 0.02867828 0.02952793 0.02994968 0.03166654 0.03245737 0.03339726
#         # 0.03475241 0.04071735 0.04278338 0.04323768 0.04806388 0.04998815
#         # 0.06182968 0.0653312  0.06947649 0.07306752 0.07626414 0.08303612]

#         # 짤라낸 피처맵 norm의 수들의 합을 계산
#         remained_filters           += len(np.argsort(rank[cov_id])[pruned_num:])

#         # 가장 큰 피처맵 랭크의 인덱스. 예를 들어 [1, 2, 3] 이면 3의 인덱스는 len([1, 2, 3]) = 3, 이것들을 모두 더함
#         tot_filter                 += len(rank[cov_id])

#         all_filters[cov_id]        = rank[cov_id].__len__()
#         compressed_filters[cov_id] = rank[cov_id][pruned_num:].__len__()
#         # print(all_filters[cov_id])
#         # print(compressed_filters[cov_id])

#     def lowest_ranking_filters(filter_rank, num):
#         data = []
#         for i in range(len(filter_rank)):
#             # print(filter_rank[i])
#             for j in range(len(filter_rank[i])):
#                 data.append((i, j, filter_rank[i][j]))
#                 # print(len(filter_rank[i]))

#         # print(data)
#         return nsmallest(num, data, itemgetter(2))

#     # 전체의 피처맵 갯수에서 살아있는 피처맵의 총 갯수를 빼면 총 프루닝하는 피처맵의 갯수
#     # print(tot_filter - remained_filters)
#     filter_to_prune = lowest_ranking_filters(rank, tot_filter - remained_filters)
#     # print(filter_to_prune)

#     for (l, _, _) in filter_to_prune:
#         compressed_filters[l] -= 1

#     compression_ratio = np.zeros(len(rank))
#     for i in range(len(compression_ratio)):
#         compression_ratio[i] = 1.0 - (compressed_filters[i] / all_filters[i])

#     new_compress_rate = compression_ratio.tolist()

#     # print(f'old compress rate: {compress_rate}')
#     # print(f'new_compress_rate: {new_compress_rate}')s
#     return new_compress_rate


# if __name__ == "__main__":
#     args.resume = args.resume + args.arch + '.pt'
#     print_logger = utils.get_logger(os.path.join(args.job_dir, "cp_ratio_logger.log"))
#     compress_rate = compute_ratio(args, print_logger=print_logger)
#     print(compress_rate)  
