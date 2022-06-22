# MobileNetV3-MobileDet-SSD-Lite
This repo is Tensorflow2/Keras implementation of MobileNetV3 and MobileDet SSD Lite. I found that strong data augmentation such as mosaic, mixup and color jitter caused in poor mAP, Backbone network for mobile and lite version of ssd might be too small to cover it. 

## Update
1. [22/06/14] Update: Quality Focal Loss and mixed precision training
2. [22/06/16] Update: Mosaic
3. [22/06/17] Update: BalanceL1 Loss

## Inference Examples
<img width="49%" src="https://user-images.githubusercontent.com/89026839/173187633-05a4711c-7d6b-4352-a217-234fabb5691d.jpg"/> <img width="49%" src="https://user-images.githubusercontent.com/89026839/173187669-3a385015-9412-4db7-8f4d-4e2ed1be0480.jpg"/>

## Reference
### Ref
1. Searching for MobileNetV3 https://arxiv.org/abs/1905.02244

2. MobileDets: Searching for Object Detection Architectures for Mobile Accelerators https://arxiv.org/abs/2004.14525

3. Generalized Focal Loss: Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection https://arxiv.org/abs/2006.04388v1
