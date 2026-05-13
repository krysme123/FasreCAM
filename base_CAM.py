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
    def __init__(self, model, target_layer, patch_size=3):
        super().__init__(model, target_layer)
        self.patch_size = patch_size  # 扰动区域大小（默认3×3）

    # 1. 批量计算logit
    @torch.no_grad()
    def _logit_batch(self, x):
        """
        x: (B,C,H,W) 批量特征图（B≥1）
        return:
            target_logit: (B,) 每个样本的目标类logit
            non_target_mean: (B,) 每个样本的非目标类均值
        """
        B = x.shape[0]
        # 批量池化+全连接
        avg_pool = self.model.avgpool(x)  # (B,C,1,1)
        flat = avg_pool.flatten(1)  # (B,C)
        out = self.model.fc(flat)  # (B, num_classes)

        # 批量处理非目标类logit
        target_logit = []
        non_target_mean = []
        for b in range(B):
            # 当前样本的目标类别
            idx_b = self._idx[b].item()
            # 非目标类logit
            non_idx_logits = torch.cat([
                out[b, :idx_b],
                out[b, idx_b + 1:]
            ], dim=0)
            target_logit.append(out[b, idx_b])
            non_target_mean.append(torch.mean(non_idx_logits))

        return torch.stack(target_logit, dim=0), torch.stack(non_target_mean, dim=0)

    # 2. 3×3区域置零扰动（核心，处理边界）
    def _perturb_3x3_patch(self, fmap, b, h_center, w_center):
        """
        对第b个样本的(h_center, w_center)为中心的3×3区域置零
        处理边界：边缘像素只扰动有效区域
        """
        H, W = fmap.shape[2], fmap.shape[3]
        # 计算3×3区域的上下左右边界（避免越界）
        h_start = max(0, h_center - self.patch_size // 2)
        h_end = min(H, h_center + self.patch_size // 2 + 1)
        w_start = max(0, w_center - self.patch_size // 2)
        w_end = min(W, w_center + self.patch_size // 2 + 1)

        # 3×3区域置零
        fmap_perturbed = fmap.clone()
        fmap_perturbed[b, :, h_start:h_end, w_start:w_end] = 0
        return fmap_perturbed

    # 3. 生成CAM
    def generate_cam(self, fmap, grad=None, score=None):
        B, C, H_f, W_f = fmap.shape  # B≥1（单样本B=1，批量B=32）
        device = fmap.device
        score_temp = torch.zeros((B, 1, H_f, W_f), device=device) 
        weights_grad = grad.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1) 

        target_original, non_target_original = self._logit_batch(fmap)
        safe_non_target_original = non_target_original + torch.sign(non_target_original) * 1e-6
        safe_score = score.squeeze(-1) + torch.sign(score.squeeze(-1)) * 1e-6  # (B,)

        # 用于分布分析
        ratio1_list = []
        ratio2_list = []

        # 遍历所有像素作为3×3中心（兼容单样本）
        for b in range(B):  # 单样本时b=0
            for h_center in range(H_f):
                for w_center in range(W_f):
                    # 3×3区域置零扰动
                    fmap_perturbed = self._perturb_3x3_patch(fmap, b, h_center, w_center)
                    # 计算扰动后logit
                    target_temp, non_target_temp = self._logit_batch(fmap_perturbed)
                    target_temp_b = target_temp[b]
                    non_target_temp_b = non_target_temp[b]

                    # 计算比率
                    ratio1_b = (safe_score[b] - target_temp_b)
                    ratio2_b = (safe_non_target_original[b] - non_target_temp_b)

                    # 记录数据
                    ratio1_list.append(np.atleast_1d(ratio1_b.detach().cpu().numpy()))
                    ratio2_list.append(np.atleast_1d(ratio2_b.detach().cpu().numpy()))

                    # 3×3区域权重赋值
                    h_start = max(0, h_center - self.patch_size // 2)
                    h_end = min(H_f, h_center + self.patch_size // 2 + 1)
                    w_start = max(0, w_center - self.patch_size // 2)
                    w_end = min(W_f, w_center + self.patch_size // 2 + 1)
                    score_temp[b, 0, h_start:h_end, w_start:w_end] = torch.abs(ratio1_b * ratio2_b)


        # 生成热力图（兼容单样本）
        cam = (weights_grad * (fmap * score_temp)).mean(1, keepdim=True)  # (B,1,H_f,W_f)
        cam = torch.relu(cam)
        return cam.detach()


