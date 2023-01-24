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

### 2.2. Reliable Propagation-Correction Modulation for Video Object Segmentation

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

### 2.4. SSTVOS: Sparse Spatiotemporal Transformers for Video Object Segmentation

Author: Duke, Brendan; Ahmed, Abdalla; Wolf, Christian; Aarabi, Parham; Taylor, Graham W.

[Code (not released)](https://github.com/dukebw/SSTVOS) [Paper](https://arxiv.org/abs/2101.08833)

![SSTVOS](https://github.com/dukebw/SSTVOS/blob/main/imgs/sstvos-architecture.png "SSTVOS")

### 2.5. Video Object Segmentation Using Global and Instance Embedding Learning

Author: Wenbin Ge; Xiankai Lu; Jianbing Shen

[URL](https://ieeexplore.ieee.org/document/9577683)

### 2.6. Fast Video Object Segmentation With Temporal Aggregation Network and Dynamic Template Matching

Author: Xuhua Huang, Jiarui Xu, Yu-Wing Tai, Chi-Keung Tang

[URL](https://arxiv.org/abs/2007.05687)

Significant progress has been made in Video Object Segmentation (VOS), the video object tracking task in its finest level. While the VOS task can be naturally decoupled into image semantic segmentation and video object tracking, significantly much more research effort has been made in segmentation than tracking. In this paper, we introduce "tracking-by-detection" into VOS which can coherently integrate segmentation into tracking, by proposing a new temporal aggregation network and a novel dynamic time-evolving template matching mechanism to achieve significantly improved performance. Notably, our method is entirely online and thus suitable for one-shot learning, and our end-to-end trainable model allows multiple object segmentation in one forward pass. We achieve new state-of-the-art performance on the DAVIS benchmark without complicated bells and whistles in both speed and accuracy, with a speed of 0.14 second per frame and J&F measure of 75.9% respectively.

### 2.7. Video Object Segmentation with Dynamic Memory Networks and Adaptive Object Alignment

Author: Liang, Shuxian; Shen, Xu; Huang, Jianqiang; Hua, Xian-Sheng

[Code (not released)](https://github.com/liang4sx/dmn-aoa) [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Video_Object_Segmentation_With_Dynamic_Memory_Networks_and_Adaptive_Object_ICCV_2021_paper.pdf)

In this paper, we propose a novel solution for object-matching based semi-supervised video object segmentation, where the target object masks in the first frame are provided. Existing object-matching based methods focus on the matching between the raw object features of the current frame and the first/previous frames. However, two issues are still not solved by these object-matching based methods. As the appearance of the video object changes drastically over time, 1) unseen parts/details of the object present in the current frame, resulting in incomplete annotation in the first annotated frame (e.g., view/scale changes). 2) even for the seen parts/details of the object in the current frame, their positions change relatively (e.g., pose changes/camera motion), leading to a misalignment for the object matching. To obtain the complete information of the target object, we propose a novel object-based dynamic memory network that exploits visual contents of all the past frames. To solve the misalignment problem caused by position changes of visual contents, we propose an adaptive object alignment module by incorporating a region translation function that aligns object proposals towards templates in the feature space. Our method achieves state-of-the-art results on latest benchmark datasets DAVIS 2017 (J of 81.4% and F of 87.5% on the validation set) and YouTube-VOS (the overall score of 82.7% on the validation set) with a very efficient inference time (0.16 second/frame on DAVIS 2017 validation set). Code is available at: https://github.com/liang4sx/DMN-AOA.

### 2.8. Joint Inductive and Transductive Learning for Video Object Segmentation

Author: Mao, Yunyao; Wang, Ning; Zhou, Wengang; Li, Houqiang

[Code](https://github.com/maoyunyao/JOINT) [Paper](https://arxiv.org/pdf/2108.03679.pdf)

![JOINT overview figure](https://github.com/maoyunyao/JOINT/blob/main/pytracking/.figs/architecture.png)

### 2.9. Video Object Segmentation with Adaptive Feature Bank and Uncertain-Region

[\[NeurIPS Page\]](https://proceedings.neurips.cc//paper_files/paper/2020/hash/234833147b97bb6aed53a8f4f1c7a7d8-Abstract.html)
[\[Paper\]](https://proceedings.neurips.cc/paper/2020/file/234833147b97bb6aed53a8f4f1c7a7d8-Paper.pdf)
[\[Supplementary\]](https://proceedings.neurips.cc/paper/2020/file/234833147b97bb6aed53a8f4f1c7a7d8-Supplemental.zip)


![](https://github.com/xmlyqing00/AFB-URR/blob/main/assets/pipeline.png)

### 2.10. Associating Objects with Transformers for Video Object Segmentation

[Code](https://github.com/yoxu515/aot-benchmark/) [Paper]()

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/decoupling-features-in-hierarchical/semi-supervised-video-object-segmentation-on-15)](https://paperswithcode.com/sota/semi-supervised-video-object-segmentation-on-15?p=decoupling-features-in-hierarchical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/associating-objects-with-scalable/video-object-segmentation-on-youtube-vos)](https://paperswithcode.com/sota/video-object-segmentation-on-youtube-vos?p=associating-objects-with-scalable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/associating-objects-with-scalable/semi-supervised-video-object-segmentation-on-18)](https://paperswithcode.com/sota/semi-supervised-video-object-segmentation-on-18?p=associating-objects-with-scalable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/associating-objects-with-scalable/semi-supervised-video-object-segmentation-on-1)](https://paperswithcode.com/sota/semi-supervised-video-object-segmentation-on-1?p=associating-objects-with-scalable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/associating-objects-with-scalable/visual-object-tracking-on-davis-2017)](https://paperswithcode.com/sota/visual-object-tracking-on-davis-2017?p=associating-objects-with-scalable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/associating-objects-with-scalable/visual-object-tracking-on-davis-2016)](https://paperswithcode.com/sota/visual-object-tracking-on-davis-2016?p=associating-objects-with-scalable)

A modular reference PyTorch implementation of AOT series frameworks:
- **DeAOT**: Decoupling Features in Hierachical Propagation for Video Object Segmentation (NeurIPS 2022, Spotlight) [[OpenReview](https://openreview.net/forum?id=DgM7-7eMkq0)][[PDF](https://arxiv.org/pdf/2210.09782.pdf)]
<img src="https://github.com/yoxu515/aot-benchmark/blob/main/source/overview_deaot.png" width="90%"/>

- **AOT**: Associating Objects with Transformers for Video Object Segmentation (NeurIPS 2021, Score 8/8/7/8) [[OpenReview](https://openreview.net/forum?id=hl3v8io3ZYt)][[PDF](https://arxiv.org/abs/2106.02638)]
<img src="https://github.com/yoxu515/aot-benchmark/blob/main/source/overview.png" width="90%"/>

An extension of AOT, [AOST](https://arxiv.org/abs/2203.11442) (under review), is available now. AOST is a more robust and flexible framework, supporting run-time speed-accuracy trade-offs.

### 2.11. Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation

[Ho Kei Cheng](https://hkchengrex.github.io/), Yu-Wing Tai, Chi-Keung Tang

NeurIPS 2021

[[arXiv]](https://arxiv.org/abs/2106.05210) [[PDF]](https://arxiv.org/pdf/2106.05210) [[Project Page]](https://hkchengrex.github.io/STCN/) [[Papers with Code]](https://paperswithcode.com/task/semi-supervised-video-object-segmentation) [[Code]](https://github.com/hkchengrex/STCN/edit/main/README.md)

### 2.12. Video Object Segmentation using Space-Time Memory Networks

Seoung Wug Oh, Joon-Young Lee, Ning Xu, Seon Joo Kim

[[Code]](https://github.com/seoungwugoh/STM/edit/master/README.md)
[[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html)

[![Video Object Segmentation using Space-Time Memory Networks (ICCV 2019)](https://img.youtube.com/vi/vVZiBEDmgIU/0.jpg)](https://www.youtube.com/watch?v=vVZiBEDmgIU "Video Object Segmentation using Space-Time Memory Networks (ICCV 2019)")

### 2.13. A Transductive Approach for Video Object Segmentation

Author: Yizhuo Zhang, Zhirong Wu, Houwen Peng, Stephen Lin

[[Code]](https://github.com/microsoft/transductive-vos.pytorch)
[[Paper]](https://arxiv.org/abs/2004.07193)

Semi-supervised video object segmentation aims to separate a target object from a video sequence, given the mask in the first frame. Most of current prevailing methods utilize information from additional modules trained in other domains like optical flow and instance segmentation, and as a result they do not compete with other methods on common ground. To address this issue, we propose a simple yet strong transductive method, in which additional modules, datasets, and dedicated architectural designs are not needed. Our method takes a label propagation approach where pixel labels are passed forward based on feature similarity in an embedding space. Different from other propagation methods, ours diffuses temporal information in a holistic manner which take accounts of long-term object appearance. In addition, our method requires few additional computational overhead, and runs at a fast ∼37 fps speed. Our single model with a vanilla ResNet50 backbone achieves an overall score of 72.3 on the DAVIS 2017 validation set and 63.1 on the test set. This simple yet high performing and efficient method can serve as a solid baseline that facilitates future research. Code and models are available at https://github.com/microsoft/transductive-vos.pytorch.

### 2.14. Fast Video Object Segmentation using the Global Context Module

Author: Yu Li (1), Zhuoran Shen (2), Ying Shan (1) ((1) Tencent PCG Applied Research Center, (2) The University of Hong Kong)

[Code (Not Released)](https://github.com/cmsflash/global-context-module) [[Paper]](https://arxiv.org/abs/2001.11243)

![](https://github.com/cmsflash/global-context-module/blob/master/illustration.png)
