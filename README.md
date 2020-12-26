# QuasiVD

QuasiVD: Efficient Dual-Frame Smoke Detection Method for IoT Edge

The code will be available soon.

# Environment
Python 3.6

Pytorch 1.3+

# Experiments

Comparison with other methods on our dataset.

Model | Backbone | Input | Params | Flops | Latency | fps | Hardware | mAP50
--- |--- |--- |--- |--- |--- |--- |--- |---
Yolov4 | Darknet | Single-frame | 64.3M | 45.7G | 15.5ms | 64.5 | RTX2080Ti | 69.70
EfficientDet-D0 | EfficientNet-B0 | Single-frame | 3.9M | 2.58G | 5.3ms | 188.2 | RTX2080Ti | 62.73
EfficientDet-D1 | EfficientNet-B1 | Single-frame | 6.6M | 3.96G | 7.4ms | 134.7 | RTX2080Ti | 66.59
EfficientDet-D2 | EfficientNet-B2 | Single-frame | 8.1M | 4.95G | 8.6ms | 116.2 | RTX2080Ti | 81.29
EfficientDet-D3 | EfficientNet-B3 | Single-frame | 12.0M | 8.21G | 12.3ms | 81.2 | RTX2080Ti | 84.32
EfficientDet-D4 | EfficientNet-B4 | Single-frame | 20.7M | 13.9G | 17.2ms | 58.2 | RTX2080Ti | 85.16
EfficientDet-D5 | EfficientNet-B5 | Single-frame | 33.7M | 21.8G | 22.7ms | 44.0 | RTX2080Ti | 86.26
EfficientDet-D6 | EfficientNet-B6 | Single-frame | 51.9M | 36.3G | 30.6ms | 32.7 | RTX2080Ti | 86.92
CenterNet | Mobilenetv3 | Single-frame | 1.48M | 3.96G | 3.4ms | 296.9 | RTX2080Ti | 85.64
QuasiVD (Proposed) | Mobilenetv3 | Dual-frame | 1.48M | 3.96G | 3.4ms | 291.6 | RTX2080Ti | 90.35
QuasiVD (Proposed) | Mobilenetv3 | Dual-frame | 1.48M | 3.96G | 312.5ms | 3.2 | Jetson | Nano | 90.35
QuasiVD (Proposed) | Mobilenetv3 | Dual-frame | - | - | 277.8ms | 3.6 | Jetson | Nano | (FP16) | 90.35



# Visualization

Some cases and QuasiVD middle terms visualization: (a) input images, (b) frame difference, (c) motion-aware mask , (d) weakly guided attention , and (e) detection results. Among these cases, “positive” cases containing smoke targets

![ ](visualization/visualization.png)

# Contact
If you have any question, please feel free to contact me (Yichao Cao, caoyichao@seu.edu.cn). Thanks :-)

