from torchvision import models
from utils import visualize
import matplotlib.pyplot as plt
import torch
plt.rcParams['font.size'] = 18

if __name__ == "__main__":
    # 判断使用何种平台
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Currently using Cuda.\n")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Currently using MPS.\n")
    else:
        device = torch.device('cpu')
        print("Currently using CPU!")

    model = models.resnet50(weights=True).eval().to(device)
    target_layer = model.layer4[-1]  # 常用

    #  ############################################### 方法在单张图片上的效果
    cam_list = ['FASRECAM']
    img_path = 'image/castle.png'
    for m in cam_list:
        print('start:', m)
        vis = visualize(img_path, model, target_layer, method=m)
        plt.imshow(vis)
        plt.title(m)
        plt.axis('off')
        # plt.savefig(f'./CAM_image/{m}.png', dpi=150)
        plt.savefig(f'./{m}.png', dpi=150)
        # plt.show()