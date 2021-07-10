# Awesome vision transformer
A curated list of vision transformer related resources, including survey, paper, source code, etc.
Maintainer: Murufeng

We are looking for a maintainer! Let me know (2923204420@qq.com) if interested.

Please feel free to pull requests or open an issue to add papers and source codes.

## Table of Contents

### [Awesome Survey](#Survey)
### [优秀论文解读](#论文解读)
### [Papers](#paper) 
#### [ViT系列变种](#ViT系列变种)
- [魔改算子](#operator)
- [Local & Hierarchical & multi-scale](#Local)
- [Transformer+卷积结合](#CNN)
- [Transformer模型压缩轻量化处理](#compress)
- [DETR变种](#detr)

#### [MLP系列]

- [MLP-Mixer](https://github.com/murufeng/Awesome_vision_transformer/blob/main/model/MLP/MLP-Mixer.py)

- [ResMLP](https://github.com/murufeng/Awesome_vision_transformer/blob/main/model/MLP/ResMLP-fb.py)

- RepMLP

- gMLP 

- Spatial shift($S^{2}$) MLP

- ViP(Vision Permute-MLP)

#### [Transformer+各类task迁移](#task)
- [1.目标检测（Object-Detection）](#1.目标检测)
- [2.超分辨率（Super-Resolution）](#2.超分辨率)
- [3.图像分割、语义分割(Segmentation)](#3.图像分割)
- [4.GAN/生成式/对抗式(GAN/Generative/Adversarial)](#4.GAN)
- [5.track](#5.track)
- [6.video](#6.video)
- [7.多模态结合](#7.multimodel)
- [8.人体姿态估计](#8.姿态估计)
- [9.神经网络架构搜索NAS](#9.NAS)
- [10.人脸识别](#10.Face)
- [11.行人重识别](#11.ReID)
- [12.密集人群检测](#12.Crowd)
- [13.医学图像处理](#13.medical)
- [14.图像风格迁移](#14.imagestyle)
- [15.low level vision(去噪，去雨，复原，去模糊等等)](#15.low_level_vision)

#### [其它](#其它)

### [模型代码复现](#code)
- [MLP](#mlp)
- [Attention](#attention)
- [Transformer](#code_transformer)


<a name="Survey"></a>
### Awesome Survey
  - [A Survey of Transformers](https://arxiv.org/abs/2106.04554)
    - 论文作者&单位: 复旦大学邱锡鹏团队; Tianyang Lin, Yuxin Wang, Xiangyang Liu, Xipeng Qiu
    - 时间: 2021.6.08
  - [A Survey on Visual Transformer](https://arxiv.org/abs/2012.12556)  
    - 论文作者&单位: 华为诺亚方舟; Kai Han, Yunhe Wang, Hanting Chen, etc
    - 2021.1.30
  - [Transformers in Vision: A Survey](https://arxiv.org/abs/2101.01169) 
    - 论文作者:Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, Mubarak Shah
    - 时间: 2021.1.4

<a name="论文解读"></a>

### 论文解读
- [颜水成团队提出VOLO屠榜CV任务，无需额外训练数据，首次在ImageNet 上达到87.1%](https://mp.weixin.qq.com/s?__biz=MzU2NDExMzE5Nw==&mid=2247510980&idx=1&sn=884b30ccdff2c6e7f44f0bc65c262199&chksm=fc4d1186cb3a989008b5cd276c6438edd39c36a82b7575d08ac2b9bf367be759d3fae16fd075&token=1173831238&lang=zh_CN#rd)
- [分层级联Transformer！苏黎世联邦提出TransCNN: 显著降低了计算/空间复杂度！](https://mp.weixin.qq.com/s?__biz=MzU2NDExMzE5Nw==&mid=2247509814&idx=2&sn=4f83ae4f12cf95b61b346b0effe45ea5&chksm=fc4d1d74cb3a9462a7cfdee53c5d9be59666ece6f1d836005d9f3d9da51a8dd113a66eff92f0&token=2050016138&lang=zh_CN#rd)
- [登上更高峰！颜水成、程明明团队开源ViP，引入三维信息编码机制，无需卷积与注意力](https://mp.weixin.qq.com/s?__biz=MzU2NDExMzE5Nw==&mid=2247510615&idx=3&sn=d0f60016ed9dc27624ffe092c3181a52&chksm=fc4d1015cb3a990371d8a9da87c93ed6711952b8cb14db9fc0e85f8ffc947465cf229d6e97c7&token=1173831238&lang=zh_CN#rd)
- [清华鲁继文团队提出DynamicViT：一种高效的动态稀疏化Token的ViT](https://mp.weixin.qq.com/s?__biz=MzU2NDExMzE5Nw==&mid=2247509696&idx=2&sn=1fe82423987a0519413782b830d469d4&chksm=fc4d1c82cb3a95943bee993c5363c47830b3d8e292d31aa93380b6f2721094740fe97c47679b&token=2050016138&lang=zh_CN#rd)
- [并非所有图像都值16x16个词--- 清华&华为提出一种自适应序列长度的动态ViT](https://mp.weixin.qq.com/s?__biz=MzU2NDExMzE5Nw==&mid=2247509514&idx=1&sn=c30c5514a76185038da274b4c6876289&chksm=fc4d1c48cb3a955eef07b72043e38a18828d6c7c784bd7560f97c66ede9fb7f67bc7e652712a&token=2050016138&lang=zh_CN#rd)
- [注意力可以使MLP完全替代CNN吗？ 未来有哪些研究方向？](https://mp.weixin.qq.com/s?__biz=MzU2NDExMzE5Nw==&mid=2247509411&idx=1&sn=20e63c8f160bc7929bc052455ea80603&chksm=fc4d1fe1cb3a96f73501568926cc5d7e7442f3bc466fb52ee606c27e316c3d0538f083908d7e&token=2050016138&lang=zh_CN#rd)
- [超越Swin Transformer！谷歌提出了收敛更快、鲁棒性更强、性能更强的NesT](https://mp.weixin.qq.com/s?__biz=MzU2NDExMzE5Nw==&mid=2247509059&idx=3&sn=1e0af034d262e7b24105dab2b6e348d2&chksm=fc4d1e01cb3a9717ceb25f7dc35cc1e91f1642862abfd0884699f007da902683457f54f8994c&token=2050016138&lang=zh_CN#rd)

<a name="paper"></a>
### Paper(最新，最受关注的)
- 超越Swin，Transformer屠榜三大视觉任务！微软推出新作：Focal Self-Attention [paper](https://arxiv.org/abs/2107.00641)
- Multi-Scale Densenet续作？搞定Transformer降采样，清华联合华为开源动态ViT！ [paper](https://arxiv.org/abs/2105.15075)
- CSWin Transformer：具有十字形窗口的视觉Transformer主干 [paper](https://arxiv.org/abs/2107.00652)
-  Faceboo提出 [Early Convolutions Help Transformers See Better](https://arxiv.org/abs/2106.14881)
- VOLO: Vision Outlooker for Visual Recognition
  - 论文链接: [https://arxiv.org/abs/2106.13112](https://arxiv.org/abs/2106.13112)
  - 代码地址: [https://github.com/sail-sg/volo](https://github.com/sail-sg/volo)
  - 作者团队:颜水成，冯佳时  无需任何额外训练数据，首次在ImageNet数据集上实现87.1%精度
- Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition
    - paper链接：[https://arxiv.org/abs/2106.12368](https://arxiv.org/abs/2106.12368)
    - 代码地址：[https://github.com/Andrew-Qibin/VisionPermutator](https://github.com/Andrew-Qibin/VisionPermutator)
- [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560)
- [CAT: Cross Attention in Vision Transformer](https://arxiv.org/abs/2106.05786)
- [CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/abs/2106.04803)
  - 作者单位：谷歌大佬;Zihang Dai, Hanxiao Liu, Quoc V. Le, Mingxing Tan
- [Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/abs/2106.03650)
- [Container: Context Aggregation Network](https://arxiv.org/abs/2106.01401)
- [Aggregating Nested Transformers](https://arxiv.org/abs/2105.12723)
- [X-volution: On the unification of convolution and self-attention](https://arxiv.org/abs/2106.02253)
- [Video Swin Transformer](https://arxiv.org/abs/2106.13230)

<a name="ViT系列变种"></a>
#### ViT系列变种
<a name="operator"></a>
#### 魔改算子
- **[VTs]**: [Visual Transformers: Token-based Image Representation and Processing for Computer Vision](https://arxiv.org/abs/2006.03677)]
- **[So-ViT]**: So-ViT: Mind Visual Tokens for Vision Transformer 
    - [paper](https://arxiv.org/abs/2104.10935)
    - [code](https://github.com/jiangtaoxie/So-ViT)
- **[Token Labeling]** Token Labeling: Training a 85.5% Top-1 Accuracy Vision Transformer with 56M Parameters on ImageNet 
   - [paper](https://arxiv.org/abs/2104.10858)
   - [code](https://github.com/zihangJiang/TokenLabeling)
- **[LeViT]** [LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference](https://arxiv.org/abs/2104.01136)]
- **[CrossViT]** [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://arxiv.org/abs/2103.14899)
- **[CeiT]** [Incorporating Convolution Designs into Visual Transformers](https://arxiv.org/abs/2103.11816)
- **[DeepViT]** [DeepViT: Towards Deeper Vision Transformer](https://arxiv.org/abs/2103.11886)
- **[TNT]** Transformer in Transformer 
   - [paper](https://arxiv.org/abs/2103.00112)
   - [code](https://github.com/huawei-noah/noah-research/tree/master/TNT)
- **[T2T-ViT]** Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet 
   - [paper](https://arxiv.org/abs/2101.11986)
   - [code](https://github.com/yitu-opensource/T2T-ViT)
- **[BoTNet]** [Bottleneck Transformers for Visual Recognition](https://arxiv.org/abs/2101.11605)]
- **[Visformer]** Visformer: The Vision-friendly Transformer 
   - [paper](https://arxiv.org/abs/2104.12533)
   - [code](https://github.com/danczs/Visformer)
- **[ConTNet]** ConTNet: Why not use convolution and transformer at the same time? 
   - [paper](https://arxiv.org/abs/2104.13497)
   - [code](https://github.com/yan-hao-tian/ConTNet)
- **[DeiT]**: Training data-efficient image transformers & distillation through attention 
   - [paper](https://arxiv.org/abs/2012.12877)
   - [code](https://github.com/facebookresearch/deit)

<a name="Local"></a>
#### Local & Hierarchical & multi-scale
- **[Twins]** Twins: Revisiting Spatial Attention Design in Vision Transformers
  - [paper](https://arxiv.org/abs/2104.13840)
  - [code](https://github.com/Meituan-AutoML/Twins)
- [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560)
- **[GasHis-Transformer]** [GasHis-Transformer: A Multi-scale Visual Transformer Approach for Gastric Histopathology Image Classification](https://arxiv.org/abs/2104.14528) 
- **[Vision Transformer]** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (**ICLR**)
  - [paper](https://arxiv.org/abs/2010.11929)
  - [code](https://github.com/google-research/vision_transformer)
- **[RegionViT]** [Regional-to-Local Attention for Vision Transformers](https://arxiv.org/abs/2106.02689)
- **[PVT]** [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)] 
   - [code](https://github.com/whai362/PVT)
- **[FPT]** Feature Pyramid Transformer (**CVPR**) 
  - [paper](https://arxiv.org/abs/2007.09451)
  - [code](https://github.com/ZHANGDONG-NJUST/FPT)
- **[PiT]** Rethinking Spatial Dimensions of Vision Transformers 
  - [[paper](https://arxiv.org/abs/2103.16302)
  - [code](https://github.com/naver-ai/pit)
- **[CoaT]** Co-Scale Conv-Attentional Image Transformers
  - [paper](https://arxiv.org/abs/2104.06399)
  - [code](https://github.com/mlpc-ucsd/CoaT)
- **[LocalViT]** LocalViT: Bringing Locality to Vision Transformers
  - [paper](https://arxiv.org/abs/2104.05707)
  - [code](https://github.com/ofsoundof/LocalViT)
- **[Swin Transformer]** Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
  - [paper](https://arxiv.org/abs/2103.14030)
  - [code](https://github.com/microsoft/Swin-Transformer)
- **[DPT]** Vision Transformers for Dense Prediction 
  - [paper](https://arxiv.org/abs/2103.13413)
  - [code](https://github.com/intel-isl/DPT)
- **[MViT]** [Mask Vision Transformer for Facial Expression Recognition in the wild](https://arxiv.org/abs/2106.04520)]


<a name="CNN"></a>
#### Transformer+卷积结合
- **[Shuffle Transformer]** [Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/abs/2106.03650)
- **[TransCNN]** Transformer in Convolutional Neural Networks\
  - [paper](https://arxiv.org/abs/2106.03180)
  - [code](https://github.com/yun-liu/TransCNN)
- **[ResT]** ResT: An Efficient Transformer for Visual Recognition
  - [paper](https://arxiv.org/abs/2105.13677)
  - [code](https://github.com/wofmanaf/ResT)
- **[CPVT]** Do We Really Need Explicit Position Encodings for Vision Transformers? 
   - [paper](https://arxiv.org/abs/2102.10882)
   - [code](https://github.com/Meituan-AutoML/CPVT)
- **[ConViT]** [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/abs/2103.10697) 
- **[CoaT]** Co-Scale Conv-Attentional Image Transformers
  - [paper](https://arxiv.org/abs/2104.06399)
  - [code](https://github.com/mlpc-ucsd/CoaT)
- **[CvT]** CvT: Introducing Convolutions to Vision Transformers 
  - [paper](https://arxiv.org/abs/2103.15808)
  - [code](https://github.com/leoxiaobin/CvT)
- **[ConTNet]** ConTNet: Why not use convolution and transformer at the same time?
  - [paper](https://arxiv.org/abs/2104.13497)
  - [code](https://github.com/yan-hao-tian/ConTNet)
- **[CeiT]** [Incorporating Convolution Designs into Visual Transformers](https://arxiv.org/abs/2103.11816)
- **[BoTNet]** [Bottleneck Transformers for Visual Recognition](https://arxiv.org/abs/2101.11605)
- **[CPTR]** [CPTR: Full Transformer Network for Image Captioning](https://arxiv.org/abs/2101.10804)

<a name="compress"></a>
#### Transformer模型压缩轻量化处理
- **[DynamicViT]**: Efficient Vision Transformers with Dynamic Token Sparsification
  - [paper](https://arxiv.org/abs/2106.02034)
  - [code](https://dynamicvit.ivg-research.xyz/)
- **[DVT]** [Not All Images are Worth 16x16 Words: Dynamic Vision Transformers with Adaptive Sequence Length](https://arxiv.org/abs/2105.15075)
- **[LeViT]** [LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference](https://arxiv.org/abs/2104.01136)


<a name="detr"></a>
### DETR变种
- **[UP-DETR]** [UP-DETR: Unsupervised Pre-training for Object Detection with Transformers (**CVPR**)](https://arxiv.org/abs/2011.09094)
- **[Deformable DETR]** Deformable DETR: Deformable Transformers for End-to-End Object Detection (**ICLR**)
  - [paper](https://arxiv.org/abs/2010.04159)
  - [code](https://github.com/fundamentalvision/Deformable-DETR)
- **[DETR]** End-to-End Object Detection with Transformers (**ECCV**) 
  - [paper](https://arxiv.org/abs/2005.12872)
  - [code](https://github.com/facebookresearch/detr)
- **[Meta-DETR]** Meta-DETR: Few-Shot Object Detection via Unified Image-Level Meta-Learning 
  - [paper](https://arxiv.org/abs/2103.11731)
  - [code](https://github.com/ZhangGongjie/Meta-DETR)
- **[DA-DETR]** [DA-DETR: Domain Adaptive Detection Transformer by Hybrid Attention](https://arxiv.org/abs/2103.17084)
- **[DETReg]** Unsupervised Pretraining with Region Priors for Object Detection
  - [paper](https://arxiv.org/abs/2106.04550)
  - [code](https://amirbar.net/detreg)
  
<a name="task"></a>
#### Transformer+各类task迁移
<a name="1.目标检测"></a>
#### Transformer+目标检测
- **[Pointformer]** [3D Object Detection with Pointformer](https://arxiv.org/abs/2012.11409) 
- **[ViT-FRCNN]** [Toward Transformer-Based Object Detection](https://arxiv.org/abs/2012.09958)
- [Oriented Object Detection with Transformer](https://arxiv.org/abs/2106.03146)
- **[YOLOS]** You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection 
  - [paper](https://arxiv.org/abs/2106.00666)
  - [code](https://github.com/hustvl/YOLOS)
- **[COTR]** [COTR: Convolution in Transformer Network for End to End Polyp Detection](https://arxiv.org/abs/2105.10925)
- **[TransVOD]** End-to-End Video Object Detection with Spatial-Temporal Transformers 
  - [paper](https://arxiv.org/abs/2105.10920)
  - [code](https://github.com/SJTU-LuHe/TransVOD)
- **[CAT]** [CAT: Cross-Attention Transformer for One-Shot Object Detection](https://arxiv.org/abs/2104.14984)
- **[M2TR]** [M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection](https://arxiv.org/abs/2104.09770)
- [Transformer Transforms Salient Object Detection and Camouflaged Object Detection](https://arxiv.org/abs/2104.10127)
- **[SSTN]** [SSTN: Self-Supervised Domain Adaptation Thermal Object Detection for Autonomous Driving](https://arxiv.org/abs/2103.03150) 
- **[TSP-FCOS]** [Rethinking Transformer-based Set Prediction for Object Detection](https://arxiv.org/abs/2011.10881)
- **[ACT]** [End-to-End Object Detection with Adaptive Clustering Transformer](https://arxiv.org/abs/2011.09315)
- **[PED]** [DETR for Pedestrian Detection](https://arxiv.org/abs/2012.06785)
- **[DPT]** Vision Transformers for Dense Prediction
  - [paper](https://arxiv.org/abs/2103.13413)
  - [code](https://github.com/intel-isl/DPT)

<a name="2.超分辨率"></a>
#### 2.超分辨率（Super-Resolution)
- **[TTSR]** Learning Texture Transformer Network for Image Super-Resolution (**CVPR**)
  - [paper](https://arxiv.org/abs/2006.04139)
  - [code](https://github.com/researchmm/TTSR)

<a name="3.图像分割"></a>

#### 3. 图像分割、语义分割(Segmentation)
- [Fully Transformer Networks for Semantic ImageSegmentation](https://arxiv.org/abs/2106.04108)
- **[TransVOS]**  [TransVOS: Video Object Segmentation with Transformers](https://arxiv.org/abs/2106.00588)
- **[SegFormer]** SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers 
  - [paper](https://arxiv.org/abs/2105.15203)
  - [code](https://github.com/NVlabs/SegFormer)
- **[VisTR]** [End-to-End Video Instance Segmentation with Transformers (**CVPR**)](https://arxiv.org/abs/2011.14503)
- **[Trans2Seg]**  Segmenting Transparent Object in the Wild with Transformer 
  - [paper](https://arxiv.org/abs/2101.08461)
  - [code](https://github.com/xieenze/Trans2Seg)
- SETR ：Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers
  - 作者单位：复旦, 牛津大学, 萨里大学, 腾讯优图, Facebook
  - 主页：[https://fudan-zvg.github.io/SETR/](https://fudan-zvg.github.io/SETR/)
  - 代码：[https://github.com/fudan-zvg/SETR](https://github.com/fudan-zvg/SETR)
  - 论文：[https://arxiv.org/abs/2012.15840](https://arxiv.org/abs/2012.15840)

<a name="4.GAN"></a>
#### 4.GAN/生成式/对抗式(GAN/Generative/Adversarial)
- **[GANsformer]**  Generative Adversarial Transformers
  - [链接](https://arxiv.org/abs/2103.01209)
  - [code](https://github.com/dorarad/gansformer)

- **[TransGAN]**: Two Transformers Can Make One Strong GAN
  - [链接](https://arxiv.org/abs/2102.07074)
  - [Code](https://github.com/VITA-Group/TransGAN)
  
- **[AOT-GAN]** Aggregated Contextual Transformations for High-Resolution Image Inpainting 
  - [paper](https://arxiv.org/abs/2104.01431) 
  - [code](https://github.com/researchmm/AOT-GAN-for-Inpainting)
  
  
<a name="5.track"></a>
#### 5.track
- **[STGT]** [Spatial-Temporal Graph Transformer for Multiple Object Tracking](https://arxiv.org/abs/2104.00194)
- [Transformer Tracking](https://arxiv.org/abs/2103.15436)
- **[TransCenter]** [TransCenter: Transformers with Dense Queries for Multiple-Object Tracking](https://arxiv.org/abs/2103.15145)
- **[TrackFormer]** [TrackFormer: Multi-Object Tracking with Transformers](https://arxiv.org/abs/2101.02702)
- **[TransTrack]** TransTrack: Multiple-Object Tracking with Transformer
  - [paper](https://arxiv.org/abs/2012.15460) 
  - [code](https://github.com/PeizeSun/TransTrack)


<a name="6.video"></a>
#### 6.video
- [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
- Anticipative Video Transformer
  - [paper](https://arxiv.org/abs/2106.02036)
  - [code](http://facebookresearch.github.io/AVT)
- **[TimeSformer]** [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
- **[VidTr]** [VidTr: Video Transformer Without Convolutions](https://arxiv.org/abs/2104.11746)
- **[ViViT]** [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)
- **[VTN]** [Video Transformer Network](https://arxiv.org/abs/2102.00719)
- **[VisTR]** [End-to-End Video Instance Segmentation with Transformers (**CVPR**)](https://arxiv.org/abs/2011.14503)
- **[STTN]** Learning Joint Spatial-Temporal Transformations for Video Inpainting (**ECCV**)
  - [paper](https://arxiv.org/abs/2007.10247)
  - [code](https://github.com/researchmm/STTN)


<a name="7.multimodel"></a>
#### 7.多模态结合
- ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision
  - 论文：https://arxiv.org/abs/2102.03334
  - 代码：https://github.com/dandelin/ViLT

<a name="8.姿态估计"></a>
#### 8.人体姿态估计
- **[TransPose]** [TransPose: Towards Explainable Human Pose Estimation by Transformer](https://arxiv.org/abs/2012.14214)
- **[TFPose]** [TFPose: Direct Human Pose Estimation with Transformers](https://arxiv.org/abs/2103.15320)
- [Lifting Transformer for 3D Human Pose Estimation in Video](https://arxiv.org/abs/2103.14304)

<a name="9.NAS"></a>
#### 9.神经网络架构搜索NAS
- **[BossNAS]** BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search 
  - [paper](https://arxiv.org/abs/2103.12424)
  - [code](https://github.com/changlin31/BossNAS)
- [Vision Transformer Architecture Search](https://arxiv.org/abs/2106.13700)

<a name="10.Face"></a>
#### 10.人脸识别
- **[FaceT]**: [Learning to Cluster Faces via Transformer](https://arxiv.org/abs/2104.11502)

<a name="11.ReID"></a>
#### 11.行人重识别
- **[TransReID]** [TransReID: Transformer-based Object Re-Identification](https://arxiv.org/abs/2102.04378)

<a name="12.Crowd"></a>
#### 12.密集人群检测
- **[TransCrowd]** TransCrowd: Weakly-Supervised Crowd Counting with Transformer 
  - [paper](https://arxiv.org/abs/2104.09116)
  - [code](https://github.com/dk-liang/TransCrowd)

<a name="13.medical"></a>
#### 13.医学图像处理
- **[SUNETR]** [SUNETR: Transformers for 3D Medical Image Segmentation](https://arxiv.org/abs/2103.10504)] 
- **[U-Transformer]** [U-Net Transformer: Self and Cross Attention for Medical Image Segmentation](https://arxiv.org/abs/2103.06104)
- **[MedT]** Medical Transformer: Gated Axial-Attention for Medical Image Segmentation 
  - [paper](https://arxiv.org/abs/2102.10662)
  - [code](https://github.com/jeya-maria-jose/Medical-Transformer)
- **[TransUNet]** TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation 
  - [paper](https://arxiv.org/abs/2102.04306)
  - [code](https://github.com/Beckschen/TransUNet)

<a name="14.imagestyle"></a>
#### 14.图像风格迁移
- StyTr2: Unbiased Image Style Transfer with Transformers
  - [paper](https://arxiv.org/abs/2105.14576)

<a name="15.low_level_vision"></a>
#### 15.low level vision(去噪，去雨，复原，去模糊等等)
- **[IPT]** [Pre-Trained Image Processing Transformer (**CVPR**)](https://arxiv.org/abs/2012.00364)
- **[SDNet]**: mutil-branch for single image deraining using swin
  - [paper](https://arxiv.org/abs/2105.15077)
  - [code](https://github.com/H-tfx/SDNet)
- Uformer: A General U-Shaped Transformer for Image Restoration
  - [paper](https://arxiv.org/abs/2106.03106)
  - [code](https://github.com/ZhendongWang6/Uformer)

<a name="其它"></a>
#### 其它
- [Chasing Sparsity in Vision Transformers:An End-to-End Exploration](https://arxiv.org/abs/2106.04533)
- [MViT: Mask Vision Transformer for Facial Expression Recognition in the wild](https://arxiv.org/abs/2106.04520)
- **[CPTR]** [CPTR: Full Transformer Network for Image Captioning](https://arxiv.org/abs/2101.10804)
- Learn to Dance with AIST++: Music Conditioned 3D Dance Generation
  - [paper](https://arxiv.org/abs/2101.08779)
  - [code](https://google.github.io/aichoreographer/)
- [Deepfake Video Detection Using Convolutional Vision Transformer](https://arxiv.org/abs/2102.11126)
- [Training Vision Transformers for Image Retrieval](https://arxiv.org/abs/2102.05644)


<a name="code"></a>
### 模型代码复现

<a name="mlp"></a>
### MLP
- [MLP-Mixer](https://github.com/murufeng/Awesome_vision_transformer/blob/main/model/MLP/MLP-Mixer.py)

- [ResMLP](https://github.com/murufeng/Awesome_vision_transformer/blob/main/model/MLP/ResMLP-fb.py)

- RepMLP

- gMLP 

- Spatial shift($S^{2}) MLP

- ViP(Vision Permute-MLP)

<a name="attention"></a>
### Attention

- 待更新

<a name="code_transformer"></a>
### Transformer

- 待更新
