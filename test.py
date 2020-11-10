import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
# 自己加的
from matplotlib import pyplot as plt
import matplotlib.cm as cm

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/teddy/vgg',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    epoch_minus = []

    for inputs, count, name in dataloader:
        # print(type(name))
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(temp_minu)

            '''
            输出密度图并保存
            为了更平滑的可视化结果，对结果进行归一化
            '''
            dm = outputs.squeeze().detach().cpu().numpy()
            # print(dm)
            # 根据issue中的提示，需要对outputs进行归一化
            dm_nor = (dm-np.min(dm))/(np.max(dm)-np.min(dm))
            # print(dm_nor)
            plt.imshow(dm_nor, cmap=cm.jet)
            plt.savefig("/tmp/img/"+name[0])
            print(name[0]+" ok!")

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
