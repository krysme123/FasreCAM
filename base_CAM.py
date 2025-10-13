import torch
import torch.nn.functional as F
import numpy as np
import cv2
from abc import ABC, abstractmethod


class BaseCAM(ABC):
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.fmap = []
        self.grad = []
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, mdl, inp, out):
        self.fmap.append(out.detach())

    def _backward_hook(self, mdl, grad_input, grad_output):
        self.grad.append(grad_output[0].detach())

    @abstractmethod
    def generate_cam(self, fmap, grad, score=None):
        """子类实现"""
        pass

    def __call__(self, x, class_idx=None):
        """x: (B,C,H,W) 已归一化"""
        self.fmap.clear()
        self.grad.clear()

        logits = self.model(x)          # 运行这一行代码的时候，会自动调用 def _forward_hook，计算得到 self.fmap
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        else:
            # 关键：将class_idx移动到logits所在设备
            class_idx = class_idx.to(logits.device)  # 新增这一行，统一设备

        self._input = x             # 传入重要参数
        self._idx = class_idx       # 传入重要参数
        # self._logits = logits

        score = logits.gather(1, class_idx.view(-1, 1)).sum()   # 把 logits 中索引为 class_idx 的元素取出来，记为 score
        self.model.zero_grad()
        score.backward(retain_graph=True)       # 运行这一行代码的时候，会自动调用 def _backward_hook，计算得到 self.grad，尺寸和 self.fmap 保持一致

        fmap = self.fmap[0]
        grad = self.grad[0] if self.grad else None
        cam = self.generate_cam(fmap, grad, score)              # 根据 fmap/grad/score 计算 fmap ！
        return cam                   # (B,1,H,W)

    @staticmethod
    def resize_cam(cam, size):              # 把 fmap 尺寸的 cam resize 回原始图片的尺寸
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.amin(dim=[2, 3], keepdim=True), cam.amax(dim=[2, 3], keepdim=True)
        cam = (cam - cam_min).div(cam_max - cam_min + 1e-8)
        return cam                   # (B,1,H,W)

    @staticmethod
    def overlay(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """image: (H,W,3) uint8 0-255"""
        cam = cam.squeeze().cpu().detach().numpy()                  # squeeze 去除尺寸为 1 的 维度
        cam = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        out = heatmap * alpha + image * (1-alpha)
        # print("overlay CAM range:", cam.min().item(), cam.max().item())
        return out.astype(np.uint8)


class FASRECAM(BaseCAM):
    @torch.no_grad()
    def _logit(self, x):
        out = self.model.fc(self.model.avgpool(x).flatten(1)).view(1, -1)
        non_idx_logits = torch.cat([out[:, :self._idx], out[:, self._idx + 1:]], dim=1)
        target_logit = out.gather(1, self._idx.view(-1, 1)).squeeze(1)
        non_target_mean = torch.mean(non_idx_logits)
        return target_logit, non_target_mean  # (B,)

    def generate_cam(self, fmap, grad=None, score=None):
        [batch, channel, height, width] = fmap.shape
        hw = height * width
        device = next(self.model.parameters()).device
        score_temp = torch.zeros((batch, 1, height, width)).to(device)
        weights_grad = grad.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)

        # 预计算原始非目标均值
        _, non_target_original = self._logit(fmap)

        # 用于收集分布数据的列表
        ratio1_list = []
        ratio2_list = []

        for i in range(hw):
            fmap_temp = fmap.clone()
            fmap_temp[:, :, i // width, i % width] = 0
            target_temp, non_target_temp = self._logit(fmap_temp)

            ratio1 = (score - target_temp)
            ratio2 = (non_target_original - non_target_temp)


            ratio1_np = ratio1.detach().cpu().numpy()
            ratio1_list.append(np.atleast_1d(ratio1_np))  # 转为一维数组

            ratio2_np = ratio2.detach().cpu().numpy()
            ratio2_list.append(np.atleast_1d(ratio2_np))  # 转为一维数组

            score_temp[:, 0, i // width, i % width] = torch.abs(ratio1 * ratio2)

        # 生成CAM
        cam = (weights_grad * (fmap * score_temp)).mean(1, keepdim=True)
        cam = torch.relu(cam)
        return cam.detach()




