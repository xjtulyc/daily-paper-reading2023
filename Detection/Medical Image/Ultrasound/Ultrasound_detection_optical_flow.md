# Object Detection for Ultrasound with Optical Flow / Motion Estimation Method

## 1. Review

### ~~1.1. A Pilot Study on Convolutional Neural Networks for Motion Estimation From Ultrasound Images - 2020~~

> Evain E, Faraz K, Grenier T, et al. A pilot study on convolutional neural networks for motion estimation from ultrasound images[J]. IEEE transactions on ultrasonics, ferroelectrics, and frequency control, 2020, 67(12): 2565-2573.

回顾了用于超声的运动估计网络

## 2. Method

### 2.1. Automatic Fetal Ultrasound Standard Plane Recognition Based on Deep Learning and IIoT - 2021

> Pu B, Li K, Li S, et al. Automatic fetal ultrasound standard plane recognition based on deep learning and IIoT[J]. IEEE Transactions on Industrial Informatics, 2021, 17(11): 7771-7780.

使用RNN结合帧间信息，胎儿检测；将光流作为辅助信息（运动表示）输入网络

### 2.2. Deep Learning-Based Pneumothorax Detection in Ultrasound Videos - 2019

> Mehanian C, Kulhare S, Millin R, et al. Deep learning-based pneumothorax detection in ultrasound videos[C]//Smart Ultrasound Imaging and Perinatal, Preterm and Paediatric Image Analysis: First International Workshop, SUSI 2019, and 4th International Workshop, PIPPI 2019, Held in Conjunction with MICCAI 2019, Shenzhen, China, October 13 and 17, 2019, Proceedings 4. Springer International Publishing, 2019: 74-82.

光流信息作为辅助信息

### 2.3. Displacement Estimation in Ultrasound Elastography Using Pyramidal Convolutional Neural Network - 2020

> Tehrani A K Z, Rivaz H. Displacement estimation in ultrasound elastography using pyramidal convolutional neural network[J]. IEEE transactions on ultrasonics, ferroelectrics, and frequency control, 2020, 67(12): 2629-2639.

一种多尺度的光流估计网络（主打实时性和小体积），并对于光流网络进行了总结

### 2.4. End-to-End Real-time Catheter Segmentation with Optical Flow-Guided Warping during Endovascular Intervention - 2020

> Nguyen A, Kundrat D, Dagnino G, et al. End-to-end real-time catheter segmentation with optical flow-guided warping during endovascular intervention[C]//2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020: 9967-9973.

光流辅助信息

### ~~2.5. Exploiting flow dynamics for super-resolution in contrast-enhanced ultrasound~~

> Solomon O, van Sloun R J G, Wijkstra H, et al. Exploiting flow dynamics for superresolution in contrast-enhanced ultrasound[J]. IEEE transactions on ultrasonics, ferroelectrics, and frequency control, 2019, 66(10): 1573-1586.

超声成像中引入光流（微泡的流动模型）提升图像质量

### 2.6. Feature Extraction of Kidney Tissue Image Based on Ultrasound Image Segmentation - 2021

> Lian J, Zhang M, Jiang N, et al. Feature extraction of kidney tissue image based on ultrasound image segmentation[J]. Journal of Healthcare Engineering, 2021, 2021.

手动设计肾脏光流特征 传统方法

### 2.7. Image-Based 3D Ultrasound Reconstruction with Optical Flow via Pyramid Warping Network - 2021

> Xie Y, Liao H, Zhang D, et al. Image-based 3D ultrasound reconstruction with optical flow via pyramid warping network[C]//2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2021: 3539-3542.

多尺度光流信息融合

### 2.8. Improving ultrasound video classification: an evaluation of novel deep learning methods in echocardiography - 2020

> Howard J P, Tan J, Shun-Shin M J, et al. Improving ultrasound video classification: an evaluation of novel deep learning methods in echocardiography[J]. Journal of medical artificial intelligence, 2020, 3.

光流辅助信息

### 2.9. SIAMESE NETWORKS WITH LOCATION PRIOR FOR LANDMARK TRACKING IN LIVER ULTRASOUND SEQUENCES - 2019

> Gomariz A, Li W, Ozkan E, et al. Siamese networks with location prior for landmark tracking in liver ultrasound sequences[C]//2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019). IEEE, 2019: 1757-1760.

siamase net. + 光流辅助用于肝脏超声目标跟踪
