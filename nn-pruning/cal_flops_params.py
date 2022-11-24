
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

if args.arch == 'googlenet':
    compress_rate = compute_ratio_googlenet(args, print_logger=print_logger)
else:
    compress_rate = compute_ratio(args, print_logger=print_logger)
    if args.arch == 'vgg_16_bn':
        compress_rate = compress_rate + [0.]
# compress_rate=[0.21875, 0.0, 0.015625, 0.0, 0.4375, 0.4375, 0.41796875, 0.99609375, 0.974609375, 0.962890625, 0.9765625, 0.984375, 0.8]
# Model

if args.arch=='vgg_16_bn':
    compress_rate[12]=0.

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

