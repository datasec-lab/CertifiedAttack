# CertifiableBlackboxAttack

This is the official implementation of the paper "certified attack".

#### This project includes the ready-to-run models/attacks/defenses and the corresponding tutorials:

- [x] Certifiable Attack

- [x] Empirical Attacks:
  
  - 14 black-box attacks from [BlackboxBench](https://github.com/SCLBD/BlackboxBench) repository: [NES](https://arxiv.org/abs/1804.08598), [ZOSignSGD](https://openreview.net/forum?id=BJe-DsC5Fm), [Bandit-prior](https://arxiv.org/abs/1807.07978), [ECO attack](https://arxiv.org/abs/1905.06635), [SimBA](https://arxiv.org/abs/1905.07121), [SignHunter](https://openreview.net/forum?id=SygW0TEFwH), [Sqaure attack](https://arxiv.org/abs/1912.00049), [Boundary attack](https://arxiv.org/abs/1712.04248), [OPT attack](https://arxiv.org/abs/1807.04457), [Sign-OPT](https://arxiv.org/abs/1909.10773), [GeoDA](https://arxiv.org/abs/2003.06468), [HSJA](https://arxiv.org/abs/1904.02144), [Sign Flip](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2336_ECCV_2020_paper.php), [RayS](https://arxiv.org/abs/2006.12792).
  
  - 2 additional unrestricted black-box attacks: [PointWise](https://arxiv.org/abs/1805.09190), [SparseEvo](https://arxiv.org/abs/2202.00091)

- [x] Defense methods:
  
  - [Blacklight Black-box Attack Detection](https://www.usenix.org/conference/usenixsecurity22/presentation/li-huiying)
  
  - [Randomized Pre-processing Defense](https://arxiv.org/abs/2104.11470)
  
  - [Randomized Post-processing Defense](https://arxiv.org/abs/1811.02054)
  
  - [TRADES Adversarial Training](https://arxiv.org/abs/1901.08573)

- [x] Image classification models and datasets from this [repository](https://github.com/hysts/pytorch_image_classification):
  
  - 6 Datasets: MNIST, FashionMNIST, KMNIST, CIFAR10, CIFAR100, ImageNet
  
  - 9 Models:  [VGG](https://arxiv.org/abs/1409.1556), [ResNet](https://arxiv.org/abs/1512.03385),  [ResNet-preact](https://arxiv.org/abs/1603.05027), [WRN](https://arxiv.org/abs/1605.07146), [DenseNet](https://arxiv.org/abs/1608.06993), [PyramidNet](https://arxiv.org/abs/1610.02915), [ResNeXt](https://arxiv.org/abs/1611.05431), [shake-shake](https://arxiv.org/abs/1705.07485), [SENet](https://arxiv.org/abs/1709.01507)

##### To set up the environment for this program, follow these steps:

1. Install Anaconda & Pytorch
   
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

2. Install required packages:
   
   ```
   pip install fvcore thop tensorboard 
   ```

3. For Certified Attacks
   
   ```
   pip install matplotlib scipy==1.11 statsmodels einops transformers accelerate pytorch_fid ema-pytorch torchviz
   ```

##### Run the code