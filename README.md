# VisionTransformer & Hierarchical Visual Transformer
Implementation of the VisionTransformer (ViT) described in paper: https://arxiv.org/abs/2010.11929

Implementation of the Scalable Visual Transformers with Hierarchical Pooling (HVT) described in paper: https://arxiv.org/abs/2103.10619

The HVT model is suitable for the use with the timm library train.py script.
To use the training script directly, add "from hierarchical_visual_transformer import *" in "timm.models.__init__"