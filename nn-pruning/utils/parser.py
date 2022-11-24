import os
import argparse
def get_args():
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
        choices=('cifar10', 'imagenet'),
        help='dataset')
    parser.add_argument(
        '--lr',
        default=0.01,
        type=float,
        help='initial learning rate')
    parser.add_argument(
        '--lr_decay_step',
        default='5, 10, 15, 20, 25',
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
        default='6,7',
        help='Select gpu to use')
    parser.add_argument(
        '--job_dir',
        type=str,
        default='./result/temp',
        help='The directory where the summaries will be stored.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
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
        # default='[0.95]+[0.5]*6+[0.9]*4+[0.8]*2',
        # default='[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0]', #densenet_40
        # default='[0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]', #resnet56
        # default='[0.1]+[0.40]*36+[0.40]*36+[0.4]*36',  # resnet110
        default = '[0.10]+[0.8]*5+[0.85]+[0.8]*3', #googlenet
        help='compress rate of each conv')
    parser.add_argument(
        '--arch',
        type=str,
        default='googlenet',
        choices=('resnet_50', 'vgg_16_bn', 'resnet_56', 'resnet_110', 'densenet_40', 'googlenet'),
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
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='num of workers for dataloader'
    )
    parser.add_argument(
        '--energy',
        type=bool,
        default=False,
        help='use energy as a criterion'
    )
    args        = parser.parse_args()
    args.resume = args.resume + args.arch + '.pt'
    # args.resume = './checkpoints_epoch/' + args.arch + '_epoch_95.pt'
    return args