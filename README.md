
# Enhanced Object Detection in Snowy Scenes via Adaptive Restoration-Assisted Learning and Large Kernel Refinement
Enhanced Object Detection in Snowy Scenes viaAdaptive Restoration-Assisted Learning and Large Kernel Refinement

## Official Pytorch implementation of ["Enhanced Object Detection in Snowy Scenes via Adaptive Restoration-Assisted Learning and Large Kernel Refinement"]

## Abstract
> Object detection in snowy weather poses significant challenges due to degraded image quality and obscured features. This paper introduces a unified detection framework that integrates image restoration tasks with object detection to mitigate weather interference. The proposed model consists of an Adaptive Multi-Level Feature Restoration (AMR) Branch for noise suppression, a Large Kernel Refinement (LKR) Module for cross-task integration, and a Cross-Scale Feature Integration (CFI) Module for feature aggregation. A large-scale snowy weather image dataset, srSnow, is constructed to support training and testing. Experimental results demonstrate that our method achieves mean Average Precision (mAP) improvements of 3.80, 2.10, and 1.86 percentage points on the rSnow, RSOD, and Snowy-weather real-world datasets, respectively, validating its effectiveness in snowy weather object detection.
## Environment
```
conda create -n SnowNet python=3.7
conda activate SnowNet
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Install dependencies
```
pip install -r requirements.txt
```

## Best path weights
The best path weights are stored in Baidu Cloud Drive and can be obtained via the following link: https://pan.baidu.com/s/1Nt8gz8JVAYg9PdBNIJahow
Extraction code: 1234

## Datasets
RSOD:https://pan.baidu.com/s/1dCTJf4NQ4u7Ai-gK9fYuiA&nbsp
pwd=r5u8


Snowy-weather：https://universe.roboflow.com/weatherdetection/snowy-weather

Of course, you can use the dataset that we have prepared.
Here are the links to the datasets that we have compiled.

https://pan.baidu.com/s/1YsbyQBW5h0V_M_NWjlOPmg?
pwd=1234

## Test

```
python map-rsod.py
python map-sw.py
```
## License
The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file.<br>
