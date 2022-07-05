
import torch
import pickle
import numpy as np


class mask_vgg_16_bn:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model         = model
        self.compress_rate = compress_rate
        self.mask          = {}
        self.job_dir       = job_dir
        self.device        = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=4,  arch="vgg_16_bn"):
        params = self.model.parameters()
        # 파라미터 추출
        # prefix = "rank_conv/"+arch+"/rank_conv"
        # prefix = "rank_conv/" + arch + "/rank_conv_hrank" #Hrank (manually by seulki)

        # rank_cov에서 피처맵의 nulcear norm을 구한 것 로드
        prefix = "rank_conv/" + arch + "_limit9/rank_conv_w"  #seulki's idea (manually by seulki)
        # prefix = "rank_conv/" + arch + "/rank_conv_w"  # seulki's idea (manually by seulki)
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir+'mask'

        self.param_per_cov = param_per_cov

        #주의점: rank는 relu이후에 계산되지면, pruning이 적용되는건 conv weight에 적용이 된다
        for index, item in enumerate(params): 

            # index == 0 이라면
            if index == cov_id * param_per_cov:
                break

            # 
            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                print(f'cov_num: {self.compress_rate[cov_id - 1]}') #by seulki
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id (상위 indice)

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight에 binary masking을 만들어줌 (output 쪽에 씌워주는게 아니라)
                item.data = item.data * self.mask[index]

            if index > (cov_id - 1) * param_per_cov and index <= (cov_id - 1) * param_per_cov + param_per_cov-1: #conv 씌워준 다음에 다음 conv 전까지의 bn과 relu에서 동작됨
                self.mask[index] = torch.squeeze(zeros) #conv weight에 씌워줬던 동일한 mask (zeros)를 BN과 ReLU에도 씌워줌
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            item.data = item.data * self.mask[index]#prune certain weight


class mask_resnet_56:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  arch="resnet_56"):
        params = self.model.parameters()
        # prefix = "rank_conv/"+arch+"/rank_conv"
        # prefix = "rank_conv/" + arch + "/rank_conv_hrank" #Hrank (manually by seulki)
        prefix = "rank_conv/" + arch + "_limit9/rank_conv_w"  #seulki's idea (manually by seulki)
        # prefix = "rank_conv/" + arch + "/rank_conv_w"  # seulki's idea (manually by seulki)
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id*param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            elif index > (cov_id-1)*param_per_cov and index < cov_id*param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index].to(self.device)

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id*self.param_per_cov:
                break
            item.data = item.data * self.mask[index].to(self.device)#prune certain weight


class mask_densenet_40:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.job_dir=job_dir
        self.device=device
        self.mask = {}

    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  arch="densenet_40"):
        params = self.model.parameters()
        # prefix = "rank_conv/"+arch+"/rank_conv"
        # prefix = "rank_conv/" + arch + "/rank_conv_hrank" #Hrank (manually by seulki)
        prefix = "rank_conv/" + arch + "/rank_conv_w"  #seulki's idea (manually by seulki)
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id * param_per_cov:
                break
            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            # prune BN's parameter
            if index > (cov_id - 1) * param_per_cov and index <= (cov_id - 1) * param_per_cov + param_per_cov-1:
                # if this BN not belong to 1st conv or transition conv --> add pre-BN mask to this mask
                if cov_id>=2 and cov_id!=14 and cov_id!=27:
                    self.mask[index] = torch.cat([self.mask[index-param_per_cov], torch.squeeze(zeros)], 0).to(self.device)
                else:
                    self.mask[index] = torch.squeeze(zeros).to(self.device)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            item.data = item.data * self.mask[index].to(self.device)


class mask_googlenet:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model         = model
        self.compress_rate = compress_rate
        self.mask          = {}
        self.job_dir       = job_dir
        self.device        = device

        self.cov_list = ['pre_layers',
                    'inception_a3',
                    'inception_b3',
                    # 'maxpool1',
                    'inception_a4',
                    'inception_b4',
                    'inception_c4',
                    'inception_d4',
                    'inception_e4',
                    # 'maxpool2',
                    'inception_a5',
                    'inception_b5',
                    ]

        # branch type
        self.tp_list = ['branch1x1_1', 'branch3x3_1', 'branch3x3_2', 'branch5x5_1', 'branch5x5_2', 'branch5x5_3', 'pool_planes']

    def layer_mask(self, cov_id, resume=None, param_per_cov=28,  arch="googlenet"): #cov_id 1부터 시작
        # for name, param in self.model.named_parameters(): ##여기까지 진행 함
        #     print(f'{name} and {param.shape}')

        #여기 맞는지 체크 해야됨
        module_block = eval('self.model.module.' + self.cov_list[cov_id-1])

        # prefix = "rank_conv/"+arch+"/rank_conv"
        # prefix = "rank_conv/" + arch + "/rank_conv_hrank" #Hrank (manually by seulki)
        prefix = "rank_conv/" + arch + "/"  #seulki's idea (manually by seulki)
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov = param_per_cov
        count = 0
        for index, [name, item] in enumerate(module_block.named_parameters()): #index 0 to 257 , item: weight & bias
            if cov_id == 1: #pre-layer
                if index == 0: #conv
                    rank = np.load(prefix + str(cov_id) + subfix)

                    f, c, w, h = item.size()
                    pruned_num = int(self.compress_rate[cov_id - 1] * f)
                    ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                    zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in range(len(ind)):
                        zeros[ind[i], 0, 0, 0] = 1.
                    self.mask[index] = zeros  # covolutional weight
                    item.data = item.data * self.mask[index]
                else: #others
                    self.mask[index] = torch.squeeze(zeros)
                    item.data = item.data * self.mask[index]

            else: #inception_block
                if index in [0, 4, 8, 12, 16, 20, 24]: #conv
                    # rank = np.load(prefix + str(cov_id) + '_' +  self.tp_list[[0, 4, 8, 12, 16, 20, 24].index(index)] + subfix)
                    rank = np.load(prefix + str(7 * (cov_id - 2) + count + 2) + subfix)
                    # print(f"{prefix + str(7 * (cov_id - 2) + count + 2) + subfix}")
                    f, c, w, h = item.size()
                    pruned_num = int(self.compress_rate[cov_id - 1] * f)
                    ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                    zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in range(len(ind)):
                        zeros[ind[i], 0, 0, 0] = 1.
                    self.mask[self.param_per_cov * (cov_id - 2) + index + 4] = zeros  # covolutional weight
                    item.data = item.data * self.mask[self.param_per_cov * (cov_id - 2) + index + 4]
                    count += 1
                else: #others
                    self.mask[self.param_per_cov * (cov_id - 2) + index + 4] = torch.squeeze(zeros)
                    item.data = item.data * self.mask[self.param_per_cov * (cov_id - 2) + index + 4]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == (cov_id-1) * self.param_per_cov + 4:
                break
            if index not in self.mask:
                continue
            item.data = item.data * self.mask[index].to(self.device)#prune certain weight


class mask_resnet_110:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  arch="resnet_110_convwise"):
        params = self.model.parameters()
        # prefix = "rank_conv/"+arch+"/rank_conv"
        # prefix = "rank_conv/" + arch + "/rank_conv_hrank" #Hrank (manually by seulki)
        prefix = "rank_conv/" + arch + "/rank_conv_w"  #seulki's idea (manually by seulki)
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id*param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)

                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.

                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            elif index > (cov_id-1)*param_per_cov and index < cov_id*param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id*self.param_per_cov:
                break
            item.data = item.data * self.mask[index].to(self.device)#prune certain weight


class mask_resnet_50:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  arch="resnet_50_convwise"):
        params = self.model.parameters()
        # prefix = "rank_conv/"+arch+"/rank_conv"
        # prefix = "rank_conv/" + arch + "/rank_conv_hrank" #Hrank (manually by seulki)
        prefix = "rank_conv/" + arch + "/rank_conv_w"  #seulki's idea (manually by seulki)
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):
            if index == cov_id * param_per_cov:
                break

            if index == (cov_id-1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id
                zeros = torch.zeros(f, 1, 1, 1).to(self.device)#.cuda(self.device[0])#.to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            elif index > (cov_id-1) * param_per_cov and index < cov_id * param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            item.data = item.data * self.mask[index]#prune certain weight
   
