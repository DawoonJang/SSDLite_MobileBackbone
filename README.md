# SSDLite with MobileNetV3 and MobileDet 
This repo is Tensorflow2/Keras implementation of MobileNetV3 and MobileDet SSD Lite. I found that strong data augmentation such as mosaic, mixup and color jitter caused in poor mAP, Backbone network for mobile and lite version of ssd might be too small to cover it.

There are some differences from the original in that I use upperbounded ReLU by 6 (ReLU6) in backbone instead of ReLU and First activation function of SE modules follows inverted bottleneck block's activation that it is included in.

## Performance
All models are trained at coco 2017 train 118k and evaluated at coco 2017 val 5k

Model | Lr schedule  | max learning rate | BatchSize | warmup epochs | total epochs | kernel regulaization | optimizer | Loss | Params | Precision | mAP |
| ------------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |
MobileNetV3Small SSDLite | CosineDecay with warmup | 2e-1 | 256 | 4 | 600 | 1e-5 | Gradient Centralization SGDM | Focal, SmoothL1 | 1.7M | FP32 | 15.2 |
MobileNetV3Small SSDLite | CosineDecay with warmup | 2e-1 | 256 | 4 | 600 | 1e-5 | Gradient Centralization SGDM | Focal, BalanceL1 | 1.7M | FP32 | 15.6 |
MobileNetV3Small SSDLite | CosineDecay with warmup | 2e-1 | 256 | 4 | 600 | 1e-5 | Gradient Centralization SGDM | Focal, BalanceL1 | 1.7M | FP16 | 15.5 |
MobileNetV3Large SSDLite | CosineDecay with warmup | 5e-2 | 64 | 4 | 600 | 2e-5 | Gradient Centralization SGDM | Focal, SmoothL1 | 3.2M | FP32 | 21.1 |
MobileNetV3Large SSDLite | CosineDecay with warmup | 1e-1 | 128 | 4 | 600 | 2e-5 | Gradient Centralization SGDM | Focal, BalanceL1 | 3.2M | FP16 | 21.4 |

## Update
1. [22/06/14] Update: Quality Focal Loss and mixed precision training
2. [22/06/16] Update: Mosaic
3. [22/06/17] Update: BalanceL1 Loss

## Inference Examples
<img width="49%" src="https://user-images.githubusercontent.com/89026839/173187633-05a4711c-7d6b-4352-a217-234fabb5691d.jpg"/> <img width="49%" src="https://user-images.githubusercontent.com/89026839/173187669-3a385015-9412-4db7-8f4d-4e2ed1be0480.jpg"/>

## Reference
1. Searching for MobileNetV3 https://arxiv.org/abs/1905.02244

2. MobileDets: Searching for Object Detection Architectures for Mobile Accelerators https://arxiv.org/abs/2004.14525

3. Generalized Focal Loss: Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection https://arxiv.org/abs/2006.04388v1

4. BalanceL1 loss is from Libra R-CNN: Towards Balanced Learning for Object Detection https://arxiv.org/abs/1904.02701

5. Gradient Centralization: A New Optimization Technique for Deep Neural Networks https://arxiv.org/abs/2004.01461