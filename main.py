import torch
import torch.nn.functional as F
from ViT import VisionTransformer
from PIL import Image
import ViT_utils as Utils


def predict(filename: str, weights_path: str):
    model = VisionTransformer(**Utils.cfg['vit_base_patch16_224'])
    model.load_state_dict(Utils.adapt_state_dict(torch.load(weights_path)))
    model.eval()

    img = Image.open(filename).convert('RGB')
    transform = Utils.ViT_transform
    tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

    with torch.no_grad():
        out = model(tensor)
    probabilities = F.softmax(out[0], dim=0)

    # Get imagenet class mappings
    with open("data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    k = 5
    top5_prob, top5_cls_id = torch.topk(probabilities, k, sorted=True)
    for i in range(k):
        print(categories[top5_cls_id[i]], top5_prob[i].item())


if __name__ == '__main__':
    predict('data/cat_luna.jpeg', 'data/jx_vit_base_p16_224-80ecf9dd.pth')

