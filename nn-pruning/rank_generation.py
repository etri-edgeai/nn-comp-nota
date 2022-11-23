import torch
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import data.imagenet as imagenet
from models import *
from utils import progress_bar
import numpy as np
parser = argparse.ArgumentParser(description='Rank extraction')

parser.add_argument('--data_dir',   type=str, default='./data/', help='dataset path')
parser.add_argument('--dataset',    type=str, default='cifar10', choices=('cifar10','imagenet'), help='dataset')
parser.add_argument('--job_dir',    type=str, default='result/tmp', help='The directory where the summaries will be stored.')
parser.add_argument('--arch',       type=str, default='googlenet', choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'), help='The architecture to prune')
parser.add_argument('--resume',     type=str, default='./checkpoints/', help='load the model from the specified checkpoint')
parser.add_argument('--limit',      type=int, default=1, help='The num of batch to get rank.')
parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size for training.')
parser.add_argument('--eval_batch_size',  type=int, default=100, help='Batch size for validation.')
parser.add_argument('--start_idx',        type=int, default=0, help='The index of conv to start extract rank.')
parser.add_argument('--gpu',              type=str, default='5', help='Select gpu to use')
parser.add_argument('--adjust_ckpt',      action='store_true', help='adjust ckpt from pruned checkpoint')
parser.add_argument('--compress_rate',    type=str, default=None, help='compress rate of each conv')
# 'vgg_16_bn'  : [0.7]*7+[0.1]*6,
# 'resnet_50'  : [0.2]+[0.8]*10+[0.8]*13+[0.55]*19+[0.45]*10,
# 'resnet_56'  : [0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4],
# 'resnet_110' : [0.1]+[0.40]*36+[0.40]*36+[0.4]*36
# 'densenet_40': [0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0],
# 'googlenet'  : [0.10]+[0.7]+[0.5]+[0.8]*4+[0.5]+[0.6]*2,

args        = parser.parse_args()
args.resume = args.resume + args.arch + '.pt'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cudnn.benchmark = True

if len(args.gpu)==1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
print('==> Preparing data..')
if args.dataset=='cifar10':
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
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif args.dataset=='imagenet':
    data_tmp = imagenet.Data(args)
    trainloader = data_tmp.loader_train
    testloader = data_tmp.loader_test

if args.compress_rate:
    import re
    cprate_str      = args.compress_rate
    cprate_str_list = cprate_str.split('+')
    pat_cprate      = re.compile(r'\d+\.\d*')
    pat_num         = re.compile(r'\*\d+')
    cprate          = []
    for x in cprate_str_list:
        num      = 1
        find_num = re.findall(pat_num,x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace('*',''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate)==1
        cprate += [float(find_cprate[0])]*num
    compress_rate = cprate
else:
    default_cprate={
        'vgg_16_bn'  : [0.7]*7+[0.1]*6,
        'densenet_40': [0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0],
        'googlenet'  : [0.10]+[0.7]+[0.5]+[0.8]*4+[0.5]+[0.6]*2,
        'resnet_50'  : [0.2]+[0.8]*10+[0.8]*13+[0.55]*19+[0.45]*10,
        'resnet_56'  : [0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4],
        'resnet_110' : [0.1]+[0.40]*36+[0.40]*36+[0.4]*36
    }
    compress_rate=default_cprate[args.arch]



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
MODEL............................................................................................................
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('==> Building model..')
print(compress_rate)
net = eval(args.arch)(compress_rate = compress_rate)
net = net.to(device)
from torchsummary import summary
summary(net, (3, 32, 32))

'GPU Check'
if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

'Load checkpoint'
if args.resume:
    # Load checkpoint (i.e. pretrained full model).
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume, map_location='cpu')

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    if args.adjust_ckpt:
        for k, v in checkpoint.items():
            new_state_dict[k.replace('module.', '')] = v
    else:
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('module.', '')] = v
    # net.load_state_dict(torch.load(args.resume, map_location='cpu'))


criterion      = nn.CrossEntropyLoss()
feature_result = torch.tensor(0.)
total          = torch.tensor(0.)



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def get_feature_hook
def get_feature_hook_densenet
def get_feature_hook_googlenet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def get_feature_hook(self, input, output):
    global feature_result
    global entropy
    global total
    a = output.shape[0] #batch
    b = output.shape[1] #filter

    # ########### Hrank (from CVPR2020)
    # c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])
    # c = c.view(a, -1).float()
    # c = c.sum(0)

    ########### Seul-Ki's approach
    c = torch.tensor([torch.svd(output[:, i, :, :])[1].sum() for i in range(b)])

    feature_result = feature_result * total + c
    total          = total + a
    feature_result = feature_result / total


def get_feature_hook_densenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0] #batch
    b = output.shape[1] #filter

    # ########### Hrank (from CVPR2020)
    # c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b-12,b)])
    # c = c.view(a, -1).float()
    # c = c.sum(0)

    ########### Seul-Ki's approach
    c = torch.tensor([torch.svd(output[:, i, :, :])[1].sum() for i in range(b-12, b)])
    print(c.shape)

    feature_result = feature_result * total + c
    total          = total + a
    feature_result = feature_result / total


def get_feature_hook_googlenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0] #batch
    b = output.shape[1] #filter

    # ########### Hrank (from CVPR2020)
    # c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b-12,b)])
    # c = c.view(a, -1).float()
    # c = c.sum(0)

    ########### Seul-Ki's approach
    c = torch.tensor([torch.svd(output[:, i, :, :])[1].sum() for i in range(b-12, b)])
    print(c.shape)

    feature_result = feature_result * total + c
    total          = total + a
    feature_result = feature_result / total



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def test module, interence accuracy
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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
            outputs         = net(inputs)
            loss            = criterion(outputs, targets)

            test_loss   += loss.item()
            _, predicted = outputs.max(1)
            total       += targets.size(0)
            correct     += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model module
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if args.arch   =='vgg_16_bn':
    if len(args.gpu) > 1:
        relucfg = net.module.relucfg
        covcfg  = net.module.covcfg
    else:
        relucfg = net.relucfg
        covcfg  = net.covcfg

    # print(covcfg)
    # print(relucfg), covcfg는 cov의 레이어 번호, relucfg는 relu의 레이어 번호
    for i, cov_id in enumerate(covcfg):
    # for i, cov_id in enumerate(relucfg):
        cov_layer = net.features[cov_id]
        handler   = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        if not os.path.isdir('rank_conv/'+ args.arch + '_limit%d'%(args.limit)):
            os.mkdir('rank_conv/'+args.arch+'_limit%d'%(args.limit))
        print(feature_result.shape)
        np.save('rank_conv/'+args.arch+'_limit%d'%(args.limit)+'/rank_conv_w' + str(i + 1) + '.npy', feature_result.numpy())

        feature_result = torch.tensor(0.)
        total          = torch.tensor(0.)

elif args.arch =='resnet_50':

    cov_layer = eval('net.maxpool')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    test()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv_w%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # ResNet50 per bottleneck
    cnt=1
    for i in range(4):
        block = eval('net.layer%d' % (i + 1))
        for j in range(net.num_blocks[i]):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv_w%d'%(cnt+1)+'.npy', feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv_w%d' % (cnt + 1) + '.npy',
                    feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu3
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            if j==0:
                np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv_w%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())#shortcut conv
                cnt += 1
            np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv_w%d' % (cnt + 1) + '.npy',
                    feature_result.numpy())#conv3
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch =='resnet_56':
    cov_layer = eval('net.conv1')
    print(cov_layer)
    handler = cov_layer.register_forward_hook(get_feature_hook)
    test()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit)+ '/rank_conv_w%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # ResNet56 per block
    cnt=1
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].conv1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit)+ '/rank_conv_w%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].conv2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit)+ '/rank_conv_w%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)
            
elif args.arch =='resnet_110':

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    test()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv_w%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    cnt = 1
    # ResNet110 per block
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(18):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch  + '_limit%d' % (args.limit) + '/rank_conv_w%d' % (
            cnt + 1) + '.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch  + '_limit%d' % (args.limit) + '/rank_conv_w%d' % (
                cnt + 1) + '.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch =='densenet_40':
    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))

    # number = 0
    for index, module1 in net.named_modules():
        if isinstance(module1, nn.Conv2d):
            # number  = number + 1
            handler = module1.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            print(feature_result.shape)
            np.save('rank_conv/' + args.arch + '_limit%d'%(args.limit) + "/" + str(number) + '.npy', feature_result.numpy())
            feature_result = torch.tensor(0.)
            total          = torch.tensor(0.)

    # # Densenet per block & transition
    # for i in range(3):
    #     dense = eval('net.dense%d' % (i + 1))
    #     for j in range(12):
    #         cov_layer = dense[j].conv1
    #         # print(cov_layer)
    #         if j==0:
    #             handler = cov_layer.register_forward_hook(get_feature_hook)
    #         else:
    #             handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
    #         test()
    #         handler.remove()

    #         np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + '/rank_conv_w%d'%(13*i+j+1)+'.npy', feature_result.numpy())
    #         feature_result = torch.tensor(0.)
    #         total = torch.tensor(0.)

    #     if i<2:
    #         trans=eval('net.trans%d' % (i + 1))
    #         cov_layer = trans.conv1
    #         # print(cov_layer)
    
    #         handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
    #         test()
    #         handler.remove()

    #         np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + '/rank_conv_w%d' % (13 * (i+1)) + '.npy', feature_result.numpy())
    #         feature_result = torch.tensor(0.)
    #         total = torch.tensor(0.)#'''

    feature_result = torch.tensor(0.)
    total          = torch.tensor(0.)
    cov_layer = net.relu #이것도 pruning 해야되나?
    handler   = cov_layer.register_forward_hook(get_feature_hook_densenet)
    test()
    handler.remove()
    np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + '/39' + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total          = torch.tensor(0.)



# elif args.arch=='googlenet':

#     if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
#         os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
#     feature_result = torch.tensor(0.)
#     total = torch.tensor(0.)

#     cov_list=['pre_layers',
#               'inception_a3',
#               'maxpool1',
#               'inception_a4',
#               'inception_b4',
#               'inception_c4',
#               'inception_d4',
#               'inception_e4',
#               'maxpool2',
#               'inception_a5',
#               'inception_b5',
#               ]

#     # branch type
#     tp_list = ['branch1x1', 'branch3x3', 'branch3x3', 'branch5x5', 'branch5x5', 'branch5x5','pool_planes']
#     for index1, module1 in net.named_modules():
#         if isinstance(module1, nn.Conv2d):
#             # print(f"{module1}")
#             handler = module1.register_forward_hook(get_feature_hook)
#             test()
#             handler.remove()

#             np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv_w%d_' % (idx + 1) + tp_list[
#                 count] + '_' + str(j + 1) + '.npy',
#                     feature_result.numpy())
            
#             feature_result = torch.tensor(0.)
#             total = torch.tensor(0.)

#     for idx, cov in enumerate(cov_list):
#         cov_idx = [0, 3, 6]

#         if idx<args.start_idx:
#             continue
#         cov_layer=eval('net.'+cov)

#         if idx > 0:
#             i = 0
#             count = 0
#             while i < 3:
#                 i += 1
#                 for tp_num, j in enumerate(range(i)):
#                     # print(f'i and tp_num: {i-1}{tp_num}')

#                     cov = eval('cov_layer.'+tp_list[count] + '[' + str(cov_idx[j]) + ']')
#                     handler = cov.register_forward_hook(get_feature_hook)
#                     test()
#                     handler.remove()

#                     np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv_w%d_' % (idx + 1) + tp_list[count] + '_' + str(j+1) + '.npy',
#                             feature_result.numpy())

#                     feature_result = torch.tensor(0.)
#                     total = torch.tensor(0.)
#                     count += 1

#             # pool_planes
#             cov = cov_layer.branch_pool[1]
#             handler = cov.register_forward_hook(get_feature_hook)
#             test()
#             handler.remove()

#             np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv_w%d_' % (idx + 1) + tp_list[-1] + '.npy',
#                     feature_result.numpy())

#         else:
#             handler = cov_layer.register_forward_hook(get_feature_hook)
#             test()
#             handler.remove()
#             np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv_w%d' % (idx + 1) + '.npy',
#                     feature_result.numpy())

#         feature_result = torch.tensor(0.)
#         total = torch.tensor(0.)


elif args.arch  == 'googlenet':
    cov_list=['pre_layers',
              'inception_a3',
              'maxpool1',
              'inception_a4',
              'inception_b4',
              'inception_c4',
              'inception_d4',
              'inception_e4',
              'maxpool2',
              'inception_a5',
              'inception_b5',
              ]
    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))

    number = 0
    for index, module1 in net.named_modules():
        if isinstance(module1, nn.Conv2d):
            number  = number + 1
            handler = module1.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            print(feature_result.shape)
            np.save('rank_conv/' + args.arch + '_limit%d'%(args.limit) + "/" + str(number) + '.npy', feature_result.numpy())
            feature_result = torch.tensor(0.)
            total          = torch.tensor(0.)

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # # branch type
    # tp_list=['n1x1','n3x3','n5x5','pool_planes']
    # for idx, cov in enumerate(cov_list):

    #     if idx<args.start_idx:
    #         continue
    #     cov_layer=eval('net.'+cov)
    #     print(cov_layer)

    #     handler = cov_layer.register_forward_hook(get_feature_hook)
    #     test()
    #     handler.remove()

    #     print(feature_result.shape)

    #     if idx>0:
    #         for idx1,tp in enumerate(tp_list):
    #             if idx1==3:
    #                 np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv_w%d_'%(idx+1)+tp+'.npy',
    #                         feature_result[sum(net.filters[idx-1][:-1]) : sum(net.filters[idx-1][:])].numpy())
    #             #elif idx1==0:
    #             #    np.save('rank_conv1/' + args.arch + '/rank_conv%d_'%(idx+1)+tp+'.npy',
    #             #            feature_result[0 : sum(net.filters[idx-1][:1])].numpy())
    #             else:
    #                 np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv_w%d_' % (idx + 1) + tp + '.npy',
    #                         feature_result[sum(net.filters[idx-1][:idx1]) : sum(net.filters[idx-1][:idx1+1])].numpy())
    #     #idx == 0
    #     else:
    #         print(idx)
    #         np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv_w%d' % (idx + 1) + '.npy',feature_result.numpy())
    #     feature_result = torch.tensor(0.)
    #     total = torch.tensor(0.)
