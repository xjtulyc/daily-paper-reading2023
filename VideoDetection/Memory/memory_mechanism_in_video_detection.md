# Memory Mechanism in Video Detection

## 1. Introduction

### 1.1. Memory

Recent state-of-the-art Video Object Segmentation (VOS) methods use attention to link representations of past frames stored in the ``feature memory`` with features extracted from the newly observed query frame which needs to be segmented. Despite the high performance of these methods, they require a ``large amount of GPU memory`` to store past frame representations. In practice, they usually struggle to handle videos longer than a minute on consumer-grade hardware.

### 1.2. Related Work


## 2. Paper List

| **Item Type**   | **Publication Year**                               | **Author**                                                                                           | **Title**                                                                                      | **Publication Title**                                                        | **DOI**                         |
|:---------------:|:--------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|:-------------------------------:|
| preprint        | 2022                                               | Cheng, Ho Kei; Schwing, Alexander G\.                                                                | XMem: Long\-Term Video Object Segmentation with an Atkinson\-Shiffrin Memory Model             |                                                                              |                                 |
| journalArticle  | 2022                                               | Xu, Xiaohao; Wang, Jinglu; Li, Xiao; Lu, Yan                                                         | Reliable Propagation\-Correction Modulation for Video Object Segmentation                      | Proceedings of the AAAI Conference on Artificial Intelligence                | 10\.1609/aaai\.v36i3\.20200     |
| conferencePaper | 2019                                               | Duarte, Kevin; Rawat, Yogesh; Shah, Mubarak                                                          | CapsuleVOS: Semi\-Supervised Video Object Segmentation Using Capsule Routing                   | 2019 IEEE/CVF International Conference on Computer Vision \(ICCV\)           | 10\.1109/ICCV\.2019\.00857      |
| conferencePaper | 2021                                               | Duke, Brendan; Ahmed, Abdalla; Wolf, Christian; Aarabi, Parham; Taylor, Graham W\.                   | SSTVOS: Sparse Spatiotemporal Transformers for Video Object Segmentation                       | 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition \(CVPR\) | 10\.1109/CVPR46437\.2021\.00585 |
| conferencePaper | 2021                                               | Ge, Wenbin; Lu, Xiankai; Shen, Jianbing                                                              | Video Object Segmentation Using Global and Instance Embedding Learning                         | 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition \(CVPR\) | 10\.1109/CVPR46437\.2021\.01656 |
| conferencePaper | 2020                                               | Huang, Xuhua; Xu, Jiarui; Tai, Yu\-Wing; Tang, Chi\-Keung                                            | Fast Video Object Segmentation With Temporal Aggregation Network and Dynamic Template Matching | 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition \(CVPR\) | 10\.1109/CVPR42600\.2020\.00890 |
| conferencePaper | 2021                                               | Liang, Shuxian; Shen, Xu; Huang, Jianqiang; Hua, Xian\-Sheng                                         | Video Object Segmentation with Dynamic Memory Networks and Adaptive Object Alignment           | 2021 IEEE/CVF International Conference on Computer Vision \(ICCV\)           | 10\.1109/ICCV48922\.2021\.00796 |
| conferencePaper | 2021                                               | Mao, Yunyao; Wang, Ning; Zhou, Wengang; Li, Houqiang                                                 | Joint Inductive and Transductive Learning for Video Object Segmentation                        | 2021 IEEE/CVF International Conference on Computer Vision \(ICCV\)           | 10\.1109/ICCV48922\.2021\.00953 |
| journalArticle  || Liang, Yongqing; Li, Xin; Jafari, Navid; Chen, Qin | Video Object Segmentation with Adaptive Feature Bank and Uncertain\-Region Reﬁnement                 |                                                                                                |                                                                              |                                 |
| journalArticle  || Yang, Zongxin; Wei, Yunchao; Yang, Yi              | Associating Objects with Transformers for Video Object Segmentation                                  |                                                                                                |                                                                              |                                 |
| journalArticle  || Cheng, Ho Kei; Tai, Yu\-Wing; Tang, Chi\-Keung     | Rethinking Space\-Time Networks with Improved Memory Coverage for Efﬁcient Video Object Segmentation |                                                                                                |                                                                              |                                 |
| conferencePaper | 2019                                               | Oh, Seoung Wug; Lee, Joon\-Young; Xu, Ning; Kim, Seon Joo                                            | Video Object Segmentation Using Space\-Time Memory Networks                                    | 2019 IEEE/CVF International Conference on Computer Vision \(ICCV\)           | 10\.1109/ICCV\.2019\.00932      |
| conferencePaper | 2020                                               | Zhang, Yizhuo; Wu, Zhirong; Peng, Houwen; Lin, Stephen                                               | A Transductive Approach for Video Object Segmentation                                          | 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition \(CVPR\) | 10\.1109/CVPR42600\.2020\.00698 |
| preprint        | 2020                                               | Li, Yu; Shen, Zhuoran; Shan, Ying                                                                    | Fast Video Object Segmentation using the Global Context Module                                 |                                                                              |                                 |


### 2.1. XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model

Author: [Ho Kei Cheng](https://hkchengrex.github.io/), [Alexander Schwing](https://www.alexander-schwing.de/)

[[arXiv]](https://arxiv.org/abs/2207.07115) [[PDF]](https://arxiv.org/pdf/2207.07115.pdf) [[Project Page]](https://hkchengrex.github.io/XMem/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RXK5QsUo2-CnOiy5AOSjoZggPVHOPh1m?usp=sharing)

![framework](https://imgur.com/ToE2frx.jpg)

We frame Video Object Segmentation (VOS), first and foremost, as a *memory* problem.
Prior works mostly use a single type of feature memory. This can be in the form of network weights (i.e., online learning), last frame segmentation (e.g., MaskTrack), spatial hidden representation (e.g., Conv-RNN-based methods), spatial-attentional features (e.g., STM, STCN, AOT), or some sort of long-term compact features (e.g., AFB-URR).

Methods with a short memory span are not robust to changes, while those with a large memory bank are subject to a catastrophic increase in computation and GPU memory usage. Attempts at long-term attentional VOS like AFB-URR compress features eagerly as soon as they are generated, leading to a loss of feature resolution.

Our method is inspired by the Atkinson-Shiffrin human memory model, which has a *sensory memory*, a *working memory*, and a *long-term memory*. These memory stores have different temporal scales and complement each other in our memory reading mechanism. It performs well in both short-term and long-term video datasets, handling videos with more than 10,000 frames with ease.

## 2.2. Reliable Propagation-Correction Modulation for Video Object Segmentation

Author: Xu, Xiaohao; Wang, Jinglu; Li, Xiao; Lu, Yan

[URL](https://ojs.aaai.org/index.php/AAAI/article/view/20200) [Code](https://github.com/JerryX1110/RPCMVOS)

![Picture1](https://user-images.githubusercontent.com/65257938/145016835-3c4be820-c55d-4eb4-b7f5-b8a012ee0f8c.png)

**Error propagation** is a general but crucial problem in **online semi-supervised video object segmentation**. We aim to **suppress error propagation through a correction mechanism with high reliability**. 

The key insight is **to disentangle the correction from the conventional mask propagation process with reliable cues**. 

We **introduce two modulators, propagation and correction modulators,** to separately perform channel-wise re-calibration on the target frame embeddings according to local temporal correlations and reliable references respectively. Specifically, we assemble the modulators with a cascaded propagation-correction scheme. This avoids overriding the effects of the reliable correction modulator by the propagation modulator. 

Although the reference frame with the ground truth label provides reliable cues, it could be very different from the target frame and introduce uncertain or incomplete correlations. We **augment the reference cues by supplementing reliable feature patches to a maintained pool**, thus offering more comprehensive and expressive object representations to the modulators. In addition, a reliability filter is designed to retrieve reliable patches and pass them in subsequent frames. 

Our model achieves **state-of-the-art performance on YouTube-VOS18/19 and DAVIS17-Val/Test** benchmarks. Extensive experiments demonstrate that the correction mechanism provides considerable performance gain by fully utilizing reliable guidance.

### 2.3. CapsuleVOS: Semi-Supervised Video Object Segmentation Using Capsule Routing

Author: Duarte, Kevin; Rawat, Yogesh; Shah, Mubarak

[URL](https://ieeexplore.ieee.org/document/9010040/) [Code](https://github.com/KevinDuarte/CapsuleVOS)

In this work we propose a capsule-based approach for semi-supervised video object segmentation. Current video object segmentation methods are frame-based and often require optical flow to capture temporal consistency across frames which can be difficult to compute. To this end, we propose a video based capsule network, CapsuleVOS, which can segment several frames at once conditioned on a reference frame and segmentation mask. This conditioning is performed through a novel routing algorithm for attention-based efficient capsule selection. We address two challenging issues in video object segmentation: 1) segmentation of small objects and 2) occlusion of objects across time. The issue of segmenting small objects is addressed with a zooming module which allows the network to process small spatial regions of the video. Apart from this, the framework utilizes a novel memory module based on recurrent networks which helps in tracking objects when they move out of frame or are occluded. The network is trained end-to-end and we demonstrate its effectiveness on two benchmark video object segmentation datasets; it outperforms current offline approaches on the Youtube-VOS dataset while having a run-time that is almost twice as fast as competing methods. The code is publicly available at https://github.com/KevinDuarte/CapsuleVOS.