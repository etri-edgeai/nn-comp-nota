for index, item in enumerate(params):  # index 0 to 257 , item: weight & bias
    if cov_id == 1:  # pre-layer
        if index == 0:  # conv
            rank = np.load(prefix + str(cov_id) + subfix)

            f, c, w, h = item.size()
            pruned_num = int(self.compress_rate[cov_id - 1] * f)
            ind = np.argsort(rank)[pruned_num:]  # preserved filter id

            zeros = torch.zeros(f, 1, 1, 1).to(self.device)
            for i in range(len(ind)):
                zeros[ind[i], 0, 0, 0] = 1.
            self.mask[index] = zeros  # covolutional weight
            item.data = item.data * self.mask[index]
        else:  # others
            self.mask[index] = torch.squeeze(zeros)
            item.data = item.data * self.mask[index]

    else:  # inception_block
        if index in [0, 4, 8, 12, 16, 20, 24]:  # conv
            print(prefix + str(cov_id) + '_' + self.tp_list[[0, 4, 8, 12, 16, 20, 24].index(index)] + subfix)
        else:  # others
            print(f'non_conv')  
