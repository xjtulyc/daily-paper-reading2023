# Pool of Papers

2023-03-08

## 1. \[Robustness - S&P \] Towards Evaluating the Robustness of Neural Networks

定义了神经网络的鲁棒性：在对抗样本攻击下的抗干扰能力。

## 2. \[US + NeRF \] Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging

将NeRF用于2D超声图像重建。

## 3. \[ Video Classification - MICCAI 22 - P 88 \] Contrastive Transformer-Based Multiple Instance Learning for Weakly Supervised Polyp Frame Detection

编码视频为token后使用对比学习增强难样本 - 结肠视频

## 4. \[ Video Segmentation - MICCAI 22 - P 99 \] Lesion-Aware Dynamic Kernel for Polyp Segmentation

使用自适应池化增强U型分割网络 - 结肠视频

## 5. \[ Video Segmentation - MICCAI 22 - P 110 \] Stepwise Feature Fusion: Local Guides Global

使用多尺度Transformer分割 - 结肠视频

2023-03-18

## 1. \[ XAI - MICCAI 22 - P 121 \] Stay Focused - Enhancing Model Interpretability Through Guided Feature Training

对于数据的预处理（加了个blur），使得热力图集中于手术器材上

## 2. \[ Single View Depths Estimating - MICCAI 22 - P 130 \] On the Uncertain Single-View Depths in Colonoscopies

自监督方法估计直肠深度，在估计的时候加入了高斯分布的置信度和深度

## 3. \[ 多模态息肉分割 - MICCAI 22 - P 141 \] Toward Clinically Assisted Colorectal Polyp Recognition via Structured Cross-Modal Representation Consistency

多模态光源拍摄的图像的息肉分割

## 4. \[ 结合了少量文本信息的结肠分割 - MICCAI 22 - P 151 \] TGANet: Text-Guided Attention for Improved Polyp Segmentation

首先回归有无息肉和息肉尺寸的信息，利用这些信息辅助分割

## 5. \[ 3D Image Encoder - MICCAI 22 - P 163\] SATr: Slice Attention with Transformer for Universal Lesion Detection

3D图像时一种和视频类似，但不同于视频的图像（可以利用上下文信息）

## 6. \[ 单个患者的ct、mri融合 - MICCAI 22 - P 175\] MAL: Multi-modal Attention Learning for Tumor Diagnosis Based on Bipartite Graph and Multiple Branches

利用二分图对不同模态之间的相关性进行建模，然后通过注意力学习和多分支网络在特征空间中进行模态融合。

## 7. \[ 有序模式树 脑网络分析 - MICCAI 22 - P 186 \] Optimal Transport Based Ordinal Pattern Tree Kernel for Brain Disease Diagnosis

提出了一种新颖的有序模式树( OPT )方法，利用脑网络中边权重的有序模式关系来表示网络的连接模式。在OPT中，节点通过有序边连接，使得节点具有层次结构。脑网络中边权重的变化会影响有序边，导致OPT的差异。我们进一步利用最优运输距离来衡量OPT配对上节点之间的运输成本。基于这些最优传输距离，我们开发了一种新的图核，称为基于最优传输的有序模式树核，用于衡量配对脑网络之间的相似性。

## 8. \[ FL - MICCAI 22 - P 196\] Dynamic Bank Learning for Semi-supervised Federated Image Diagnosis with Class Imbalance

尽管半监督联邦学习( FL )在医学图像诊断中取得了一些进展，但在实际应用中，未标记客户端之间的类分布不平衡问题仍然没有解决。在本文中，我们研究了一个实际但具有挑战性的类不平衡半监督FL ( imFed-Semi )问题，该问题允许所有客户端只有未标记数据，而服务器只有少量标记数据

