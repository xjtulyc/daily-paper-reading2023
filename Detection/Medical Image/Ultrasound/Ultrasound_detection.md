# Object Detection for Ultrasound

## 1. Review

### 1.1. Deep Learning in Medical Ultrasound Analysis: A Review - 2019

cite

> Liu S, Wang Y, Yang X, et al. Deep learning in medical ultrasound analysis: a review[J]. Engineering, 2019, 5(2): 261-275.

超声相较于MRI和CT的优势

> Over the decades, it has been demonstrated that US has several major advantages over other medical imaging modalities such as X-ray, magnetic resonance imaging (MRI), and computed tomography (CT), including its non-ionizing radiation, portability, accessibility, and cost effectiveness.

深度学习在超声分析中的应用

> As noted earlier, current applications of deep learning techniques in US analysis mainly involve three types of tasks: classification, detection, and segmentation for various anatomical structures or tissues, such as the breast, prostate, liver, heart, and fetus.

在检测任务中，关注``Tumors or lesions``/``Fetus``/``Cardiac``

### 1.2. Towards Clinical Application of Artificial Intelligence in Ultrasound Imaging - 2021

cite

> Komatsu M, Sakai A, Dozen A, et al. Towards clinical application of artificial intelligence in ultrasound imaging[J]. Biomedicines, 2021, 9(7): 720.

可解释性是ai应用在医学影像中的重要环节

> Examiners need to understand and explain the rationale for diagnosis to patients objectively for obtaining informed consent in constructing valid AI-based US diagnostic technologies in clinical practice.

在成像原理方面，有两个方面需要注意

> US image quality improvement and acoustic shadow detection

## 2. 方法

### 2.1. A Deep Learning Framework for Single-Sided Sound Speed Inversion in Medical Ultrasound - 2019

提出了一种基于深度学习的方法，快速将原始超声图像转化为组织声速图（tissue sound speed map）

### 2.2. Adaptive and Compressive Beamforming Using Deep Learning for Medical Ultrasound - 2020

提出了一种基于深度学习的方法，快速从射频数据中生成高质量超声图像

### 2.3. Automatic 3D Ultrasound Segmentation of Uterus Using Deep Learning - 2021

cite
> Behboodi B, Rivaz H, Lalondrelle S, et al. Automatic 3D ultrasound segmentation of uterus using deep learning[C]//2021 IEEE International Ultrasonics Symposium (IUS). IEEE, 2021: 1-4.

使用2D-UNet分割3D子宫

### 2.4. Bimodal Automated Carotid Ultrasound Segmentation using Geometrically Constrained Convolutional Neural Networks - 2019

cite

> Azzopardi C, Camilleri K P, Hicks Y A. Bimodal automated carotid ultrasound segmentation using geometrically constrained deep neural networks[J]. IEEE Journal of Biomedical and Health Informatics, 2020, 24(4): 1004-1015.

用于颈动脉超声分割的方法

### 2.5. Detection of Lines and Boundaries in Speckle Images—Application to Medical Ultrasound - 1999

> Czerwinski R N, Jones D L, O'Brien W D. Detection of lines and boundaries in speckle images-application to medical ultrasound[J]. IEEE transactions on medical imaging, 1999, 18(2): 126-136.

主要贡献

> This paper describes an approach to boundary detection in ultrasound speckle based on an image enhancement technique.

### 2.6. Edge Detection in Medical Ultrasound Images Using Adjusted Canny Edge Detection Algorithm - 2016

> Nikolic M, Tuba E, Tuba M. Edge detection in medical ultrasound images using adjusted Canny edge detection algorithm[C]//2016 24th Telecommunications Forum (TELFOR). IEEE, 2016: 1-4.

使用改进canny算法进行边缘检测

### 2.7. Key-frame Guided Network for Thyroid Nodule Recognition using Ultrasound Videos - 2022

> Wang Y, Li Z, Cui X, et al. Key-frame Guided Network for Thyroid Nodule Recognition Using Ultrasound Videos[C]//Medical Image Computing and Computer Assisted Intervention–MICCAI 2022: 25th International Conference, Singapore, September 18–22, 2022, Proceedings, Part IV. Cham: Springer Nature Switzerland, 2022: 238-247.

可以回归关键帧位置的甲状腺检测系统

### 2.8. Low-Memory CNNs Enabling Real-Time Ultrasound Segmentation Towards Mobile Deployment - 2020

> Vaze S, Xie W, Namburete A I L. Low-memory CNNs enabling real-time ultrasound segmentation towards mobile deployment[J]. IEEE Journal of Biomedical and Health Informatics, 2020, 24(4): 1059-1069.

使用可分离卷积和知识蒸馏的实时监测系统

关于实时的定义

> In this work, we use the widely accepted figure of 30 fps to define ‘real-time’ processing [8].

### 2.9. Medical Breast Ultrasound Image Segmentation by Machine Learning - 2019

> Xu Y, Wang Y, Yuan J, et al. Medical breast ultrasound image segmentation by machine learning[J]. Ultrasonics, 2019, 91: 1-9.

乳腺组织分割

### 2.10. Transfer Learning U-Net Deep Learning for Lung Ultrasound Segmentation - 2021

> Cheng D, Lam E Y. Transfer learning U-Net deep learning for lung ultrasound segmentation[J]. arXiv preprint arXiv:2110.02196, 2021.

迁移学习+UNet+肺部分割

### 2.11. Ultrasound segmentation using U-Net: learning from simulated data and testing on real data - 2019

> Behboodi B, Rivaz H. Ultrasound segmentation using u-net: learning from simulated data and testing on real data[C]//2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE, 2019: 6628-6631.

使用模拟数据训练Unet

