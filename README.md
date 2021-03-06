## SSDLite with MobileBackbone
This repo is Tensorflow2/Keras implementation of MobileBackbone such as MobileNetV3 and MobileDet SSD Lite. I found that strong data augmentation such as mosaic, mixup and color jitter caused in poor mAP, Backbone network for mobile and lite version of ssd might be too small to cover it.

There are some differences from the original in that I used upperbounded ReLU by 6 (ReLU6) in backbone instead of ReLU and First activation function of SE modules follows inverted bottleneck block's activation that it is included in.

Mixed precision training could reduce training time by 60% retaining mAP performance

## Performance
All models are trained at coco 2017 train 118k and evaluated at coco 2017 val 5k

Model | Lr schedule  | max learning rate | BatchSize | total epochs | kernel regulaization | optimizer | Loss | Input Size | Params[M] | Training Precision | FLOPs[G] | mAP 0.5:0.95@0.05 |
| ------------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
MobileNetV3Small SSDLite | CosineDecay with warmup | 2e-1 | 256 | 600 | 1e-5 | Gradient Centralization SGDM | Focal, SmoothL1 |320x320| 1.7 | FP16 | 0.32 | 15.2 |
MobileNetV3Small SSDLite | CosineDecay with warmup | 2e-1 | 256 | 600 | 1e-5 | Gradient Centralization SGDM | Focal, BalanceL1 |320x320| 1.7 | FP16 | 0.32 | 15.5 |
MobileNetV3Large SSDLite | CosineDecay with warmup | 1e-1 | 128 | 600 | 2e-5 | Gradient Centralization SGDM | Focal, SmoothL1 |320x320| 3.2 | FP16 | 1.01 | 21.1 |
MobileNetV3Large SSDLite | CosineDecay with warmup | 1e-1 | 128 | 600 | 2e-5 | Gradient Centralization SGDM | Focal, BalanceL1 |320x320| 3.2 | FP16 | 1.01 |  21.4 |
MobileDetCPU SSDLite | CosineDecay with warmup | 2e-1 | 128 | 600 | 2e-5 | Gradient Centralization SGDM | Focal, SmoothL1 |320x320| 4.1 | FP16 | 0.93 | 22.5 |
MobileDetCPU SSDLite | CosineDecay with warmup | 2e-1 | 128 | 600 | 2e-5 | Gradient Centralization SGDM | Focal, BalanceL1 |320x320| 4.1 | FP16 | 0.93 | 22.7 |

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
