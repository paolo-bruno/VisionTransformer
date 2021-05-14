import time

import torch
import torch.nn.functional as F
from ViT import VisionTransformer
from HVT import HierarchicalVisualTransformer
from PIL import Image
import ViT_utils as Utils
from torch.utils.data import DataLoader
from torchvision import datasets

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


def validate_on_ilsvrc2012_img_val(cfg: str):
    """
    Run the validation phase on images inside ILSVRC2012_img_val.tar

    """
    imagenet_val = datasets.imagenet.ImageNet(
        "./data",
        split='val',
        transform=Utils.ViT_transform[cfg]
    )

    val_loader = DataLoader(imagenet_val, batch_size=16, shuffle=False)

    model = VisionTransformer(**Utils.cfg[cfg]).to(device=device)
    # add weights ported from official Google JAX impl
    model.load_state_dict(Utils.adapt_state_dict(torch.load('data/' + Utils.weigths_path[cfg])))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum()
            print("Partial accuracy val {:.2f}".format(correct / total))

    print("Accuracy val: {:.2f}".format(correct / total))


def predict(filename: str, cfg: str):
    model = VisionTransformer(**Utils.cfg[cfg])
    model.load_state_dict(Utils.adapt_state_dict(torch.load('data/' + Utils.weigths_path[cfg])))
    model.eval()

    img = Image.open(filename).convert('RGB')
    transform = Utils.ViT_transform[cfg]
    tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

    with torch.no_grad():
        start = time.time_ns()
        out = model(tensor)
        finish = time.time_ns()
    print(f'{finish - start} ns')
    probabilities = F.softmax(out[0], dim=0)

    # Get imagenet class mappings
    with open("data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    k = 5
    top5_prob, top5_cls_id = torch.topk(probabilities, k, sorted=True)
    for i in range(k):
        print(categories[top5_cls_id[i]], top5_prob[i].item())


def test_hvt(filename: str):
    model = HierarchicalVisualTransformer()
    model.eval()

    img = Image.open(filename).convert('RGB')
    transform = Utils.ViT_transform['vit_base_patch16_224']
    tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

    with torch.no_grad():
        start = time.time_ns()
        out = model(tensor)
        finish = time.time_ns()
    print(f'{finish-start} ns')
    probabilities = F.softmax(out[0], dim=0)

    # Get imagenet class mappings
    with open("data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    k = 5
    top5_prob, top5_cls_id = torch.topk(probabilities, k, sorted=True)
    for i in range(k):
        print(categories[top5_cls_id[i]], top5_prob[i].item())


if __name__ == '__main__':
    # predict('data/cat_luna.jpeg', 'data/jx_vit_base_p16_224-80ecf9dd.pth')
    # predict('data/cat_luna.jpeg', 'vit_large_patch16_224')
    # validate_on_ilsvrc2012_img_val('vit_large_patch16_224')
    test_hvt('data/cat_luna.jpeg')
