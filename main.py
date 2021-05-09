# This is an implementation of the VisionTransformer (ViT)
# described in paper: https://arxiv.org/abs/2010.11929
import torch
from ViT import VisionTransformer


def main():
    model = VisionTransformer()
    x = torch.randn((1, 3, 224, 224))
    y = model(input)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
