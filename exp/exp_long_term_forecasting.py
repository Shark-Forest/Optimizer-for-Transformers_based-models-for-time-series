import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import sys

# 确保项目根目录被添加
current_script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch.cuda.amp import GradScaler, autocast
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
# 注释掉注意力层导入（如果仅用于可视化）
# from layers.SelfAttention_Family import AttentionLayer  # 导入注意力层类

# --------------------------
# 注释掉可视化相关导入
# --------------------------
# from models.visualization import (
#     visualize_forward_attn,
#     visualize_backward_attn,
#     global_forward_attn_without_weight,  # 前向无加权基准全局变量
#     global_backward_attn_without_weight  # 反向无加权基准全局变量
# )

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # 注释掉可视化频率参数
        # self.visualize_freq = getattr(args, 'visualize_freq', 1)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach()
                true = batch_y.detach()
                loss = criterion(pred, true)
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # 注释掉注意力层初始化方法
    # def init_attention_layers(self, epoch, use_weight):
    #     """初始化所有AttentionLayer的可视化参数"""
    #     # 遍历模型所有子模块，兼容DataParallel包装的模型
    #     model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
    #     for module in model.modules():
    #         if isinstance(module, AttentionLayer):
    #             module.set_epoch(epoch)
    #             module.use_weight = use_weight
    #             module.visualize_freq = self.visualize_freq  # 同步可视化频率
    #             module.visualize_count = 0
    #             # print(f"[可视化初始化] AttentionLayer设置：epoch={epoch}, use_weight={use_weight}")

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = GradScaler()

        # 注释掉注意力初始化相关代码
        # 初始化第0轮（无加权基准）
        # self.init_attention_layers(epoch=0, use_weight=False)

        for epoch in range(self.args.train_epochs):
            current_epoch = epoch + 1  # 1-based索引
            # 注释掉注意力权重相关变量
            # use_weight = (epoch > 0)  # 第0轮无加权，后续加权
            # self.init_attention_layers(current_epoch, use_weight)

            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 前向传播
                if self.args.use_amp:
                    with autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # 打印训练进度
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {current_epoch} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                # 反向传播
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # 轮次结束统计
            print(f"Epoch: {current_epoch} cost time: {time.time() - epoch_time:.4f}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(f"Epoch: {current_epoch}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            adjust_learning_rate(model_optim, current_epoch, self.args)

            # --------------------------
            # 注释掉所有可视化相关逻辑
            # --------------------------
            # if (epoch % self.visualize_freq == 0):
            #     # 适配DataParallel模型
            #     model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            #     
            #     # 遍历所有AttentionLayer，提取并可视化注意力
            #     for module in model.modules():
            #         if isinstance(module, AttentionLayer):
            #             # 1. 可视化前向注意力
            #             # （前向注意力在AttentionLayer.forward中已计算，此处补充基准缓存逻辑）
            #             global global_forward_attn_without_weight
            #             if not use_weight and global_forward_attn_without_weight is None:
            #                 # 第0轮缓存无加权前向基准
            #                 if hasattr(module, 'last_forward_attn'):  # 需在AttentionLayer中保存last_forward_attn
            #                     visualize_forward_attn(
            #                         attn_matrix=module.last_forward_attn,
            #                         epoch=current_epoch,
            #                         prefix="forward_without_weight"
            #                     )
            #                     global_forward_attn_without_weight = module.last_forward_attn
            #
            #             # 2. 可视化反向注意力（从key_projection提取）
            #             if hasattr(module.key_projection, 'backward_attn') and module.key_projection.backward_attn is not None:
            #                 backward_attn = module.key_projection.backward_attn.cpu().detach().numpy()
            #                 
            #                 # 缓存反向无加权基准
            #                 global global_backward_attn_without_weight
            #                 if not use_weight and global_backward_attn_without_weight is None:
            #                     visualize_backward_attn(
            #                         attn_sharp_matrix=backward_attn,
            #                         epoch=current_epoch,
            #                         prefix="backward_without_weight"
            #                     )
            #                     global_backward_attn_without_weight = backward_attn
            #                 # 可视化反向加权图
            #                 elif use_weight:
            #                     visualize_backward_attn(
            #                         attn_sharp_matrix=backward_attn,
            #                         epoch=current_epoch,
            #                         prefix="backward_with_weight"
            #                     )

        # 加载最佳模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                preds.append(outputs)
                trues.append(batch_y)
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], trues[0][0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], preds[0][0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            f.write('\n\n')

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

        return