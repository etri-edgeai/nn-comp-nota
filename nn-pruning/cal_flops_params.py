
import torch
import argparse
import utils
import os
import get_flops
from models import *
from compute_comp_ratio import compute_ratio
from compute_comp_ratio_googlenet import compute_ratio as compute_ratio_googlenet
import modules.flops_counter_mask as fcm
import modules.flop as flop

parser = argparse.ArgumentParser(description='Calculating flops and params')

parser.add_argument(
    '--input_image_size',
    type=int,
    default=32,
    help='The input_image_size')
parser.add_argument(
    '--gpu',
    type=str,
    default='5',
    help='Select gpu to use')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet','resnet_50'),
    help='The architecture to prune')
parser.add_argument(
    '--compress_rate',
    type=str,
    default='[0.95]+[0.5]*6+[0.9]*4+[0.8]*2',
    help='compress rate of each conv')
parser.add_argument(
    '--start_cov',
    type=int,
    default=0,
    help='The num of conv to start prune')
parser.add_argument(
    '--job_dir',
    type=str,
    default='./result/temp_cal_flops_params',
    help='The directory where the summaries will be stored.')
args = parser.parse_args()

# device = torch.device("cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ckpt         = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
utils.print_params(vars(args), print_logger.info)

# Data
print_logger.info('==> Preparing data..')


if args.compress_rate:
    import re
    cprate_str=args.compress_rate
    cprate_str_list=cprate_str.split('+')
    pat_cprate=re.compile(r'\d+\.\d*')
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
        print(float(find_cprate[0]),num)
        cprate+=[float(find_cprate[0])]*num
    compress_rate=cprate
    print(compress_rate)

args.compress_rate = cprate
#
# if args.arch == 'googlenet':
#     compress_rate = compute_ratio_googlenet(args, print_logger=print_logger)
# else:
#     compress_rate = compute_ratio(args, print_logger=print_logger)
#     if args.arch == 'vgg_16_bn':
#         compress_rate = compress_rate + [0.]
# # compress_rate=[0.21875, 0.0, 0.015625, 0.0, 0.4375, 0.4375, 0.41796875, 0.99609375, 0.974609375, 0.962890625, 0.9765625, 0.984375, 0.8]
# # Model
#
# if args.arch=='vgg_16_bn':
#     compress_rate[12]=0.


# compress_rate = [0.4964518384054989, 0.2849058934492172, 0.08980860036024552, 0.37073162484256195, 0.2093322216310498, 0.12194669511621586, 0.010377582469950596, 0.357737695584358, 0.01995439420085444, 0.6296248337514334, 0.45419335484341333, 0.384814073689322, 0.5206852081311238]
# compress_rate = [0.390625, 0.0, 0.71875, 0.28125, 0.0625, 0.34375, 0.792968, 0.023437, 0.363281, 0.8125, 0.300781, 0.365234, 0.533203]
# compress_rate = [0.051752, 0.080923, 0.147122, 0.126646, 0.234942, 0.196759, 0.185370, 0.422472, 0.382292, 0.459086, 0.521959, 0.573066, 0.155299]
print('==> Building model..')
net = eval(args.arch)(compress_rate=compress_rate)
print(net.compress_rate)
net.eval()

if args.arch=='googlenet' or args.arch=='resnet_50':
    flops, params = get_flops.measure_model(net, device, 3, args.input_image_size, args.input_image_size, True)
else:
    flops, params= get_flops.measure_model(net,device,3,args.input_image_size,args.input_image_size)

print('Params: %.2f'%(params))
print('Flops: %.2f'%(flops))

