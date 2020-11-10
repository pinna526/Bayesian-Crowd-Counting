from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from  models.vgg import vgg19
from datasets.crowd import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
# 自己加的
from matplotlib import pyplot as plt
import matplotlib.cm as cm

'''每加载一份样本调用一次这个函数'''
def train_collate(batch):
    # print("regression Trainer ---> train_collate")
    transposed_batch = list(zip(*batch))    # zip() 一对一二对二三对三的列表元组
    images = torch.stack(transposed_batch[0], 0)    #叠加纬度0
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

class RegTrainer(Trainer):
    def setup(self):
        print("regression Trainer ---> setup")
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'val']}
        # pytorch的dataloaders
        self.dataloaders = {x: DataLoader(self.datasets[x], # 传入的数据集
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),  # 每个batch有多少个样本
                                          shuffle=(True if x == 'train' else False),    # 洗牌
                                          num_workers=args.num_workers*self.device_count,   #几个进程来处理
                                          pin_memory=(True if x == 'train' else False)) # 是否拷贝到cuda的固定内存中
                            for x in ['train', 'val']}
        print()
        self.model =vgg19() # 用vgg19模型
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        print("regression Trainer ---> train")
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()


    '''
    train_epoch训练次数，这里训练1000次
    初始化loss,mae,mse，时间和模型，开始新一轮训练
    '''
    def train_eopch(self):
        print("regression Trainer ---> train_eopch")
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode


        '''
        遍历洗牌后的数据进行训练，乱序
        inputs, points, targets, st_sizes已经在前面的train_collate()定义
        '''
        # Iterate over data.
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):

            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            # print("points:")
            # print(points)
            targets = [t.to(self.device) for t in targets]
            # print("targets:")
            # print(targets)

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                # outputs可以转换为64*64的矩阵，可以表示密度图，但是和论文里的不符
                # 【model是什么----vgg19模型】
                # 【输出查看Inputs是什么----tensor矩阵】
                # print(inputs)
                # 【输出查看Outputs是什么----tensor矩阵】
                # print(outputs)

                '''
                针对每一次训练，输出图像，发现都是64*64大小
                这里用的是层层卷积处理好的数据
                无法获取图像名称，而且已经被洗牌，所以顺序对不上
                '''
                # dm = outputs.squeeze().detach().cpu().numpy()
                # dm_nor = (dm-np.min(dm))/(np.max(dm)-np.min(dm)) # 归一化
                # plt.imshow(dm_nor, cmap=cm.jet)
                # 这里img都被数据代替，无法获取名字，所以用Num计数
                # plt.savefig("D:\研究生\BayesCrowdCounting\\" + str(num))
                # print("ok!")

                '''
                先验概率和损失
                在前面已经定义
                self.post_prob = Post_Prob(args.sigma,args.crop_size,……)
                self.criterion = Bay_Loss(args.use_background, self.device)
                '''
                prob_list = self.post_prob(points, st_sizes)
                loss = self.criterion(prob_list, targets, outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count

                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        '''
        训练完一轮后在这里输出loss,mse,mae……的平均值
        '''
        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time()-epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        print("regression Trainer ---> val_epoch")
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)

                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))



