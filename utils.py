from base_CAM import FASRECAM
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import cv2



METHOD_MAP = {
    'FASRECAM': FASRECAM,
}


def get_cam(method, model, target_layer):
    return METHOD_MAP[method](model, target_layer)


def get_cam_map(image_tensor,  model, target_layer, method='GradCAM', class_idx=None, feature_dimension=2048):
    """
    获取CAM热力图（数值形式）

    参数:
        image_tensor: 输入图像张量 (1, C, H, W)
        model: 目标模型
        class_idx: 目标类别索引
        method: CAM方法名称
        feature_dimension: DiffCAM的特征维度

    返回:
        cam_map: CAM热力图 (1, 1, H, W)
    """
    device = next(model.parameters()).device
    # 如果传入的id是整数 转成张量给base——cam
    class_idx = torch.tensor([class_idx], device=image_tensor.device)
    x = image_tensor.to(device)
    # 使用现有的CAM生成流程
    cam_obj = get_cam(method, model, target_layer)

    if method == 'DiffCAM':
        dummy_reference = torch.randn(100, feature_dimension).to(device)
        cam_obj.set_reference_features(dummy_reference)
    # 生成CAM
    cam = cam_obj(x, class_idx=class_idx)

    # 调整到原始图像尺寸并归一化
    # cam 是模型生成的原始类激活图，其尺寸通常与模型最后一个卷积层的输出特征图尺寸一致（例如常见的 7×7 或 14×14），远小于输入图像的尺寸。
    # image_tensor 是原始图像数据（未经过预处理的图像），image_tensor.shape[:2] 表示获取原始图像的高度和宽度（例如 (224, 224) 或 (480, 640)）。
    cam = cam_obj.resize_cam(cam, size=image_tensor.shape[-2:])
    return cam


def load_image(path, img_size=224):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(path).convert('RGB')
    return tfm(img).unsqueeze(0), np.array(img)


def visualize(image_path, model, target_layer, method='GradCAM', class_idx=None, feature_dimension=2048):
    device = next(model.parameters()).device
    x, raw = load_image(image_path)
    x = x.to(device)
    cam_obj = get_cam(method, model, target_layer)

    cam = cam_obj(x, class_idx=class_idx)
    cam = cam_obj.resize_cam(cam, size=raw.shape[:2])
    vis = cam_obj.overlay(raw, cam)
    return vis    # uint8 RGB

def visualize_without_resize(image_path, model, target_layer, method='GradCAM', class_idx=None, feature_dimension=2048):
    device = next(model.parameters()).device
    x, raw = load_image(image_path)
    x = x.to(device)
    cam_obj = get_cam(method, model, target_layer)
    cam = cam_obj(x, class_idx=class_idx)

    cam = cam_obj.resize_cam(cam, size=raw.shape[:2])       # ############# resize!!!!
    # cam = cam_obj.resize_cam(cam, size=[128, 128])       # ############# resize!!!!
    # cam = cam_obj.resize_cam(cam, size=[7, 7])       # ############# resize!!!!

    cam = cam.squeeze().cpu().detach().numpy()  # squeeze 去除尺寸为 1 的 维度
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # vis = cam_obj.overlay(raw, cam)
    return heatmap    # uint8 RGB

