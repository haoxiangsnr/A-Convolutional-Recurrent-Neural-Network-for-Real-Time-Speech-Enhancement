import torch
from torch.nn.utils.rnn import pad_sequence

# Copy from https://github.com/YangYang/CRNN_mapping_baseline/blob/master/utils/loss_utils.py
def mse_loss():
    def loss_function(est, label, nframes):
        """
        计算真实的MSE
        :param est: 网络输出
        :param label: label
        :param nframes: 每个batch中的真实帧长
        :return:loss
        """
        EPSILON = 1e-7
        with torch.no_grad():
            mask_for_loss_list = []
            # 制作掩码
            for frame_num in nframes:
                mask_for_loss_list.append(torch.ones(frame_num, label.size()[2], dtype=torch.float32))
            # input: list of tensor
            # output: B T F
            mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True).cuda()
        # 使用掩码计算真实值
        masked_est = est * mask_for_loss
        masked_label = label * mask_for_loss
        loss = ((masked_est - masked_label) ** 2).sum() / mask_for_loss.sum() + EPSILON
        return loss
    return loss_function