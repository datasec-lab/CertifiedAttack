# Certifiable Black-box Attack

This is the official implementation of the paper "Certifiable Black-Box Attacks with Randomized Adversarial Examples: Breaking Defenses with Provable Confidence".

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

Note: All the experimental settings used in the paper are stored in './configs',  you can also customize your config files in './configs' to meet your requirements. The default configuration for all the attacks/defenses and models are stored in './pytorch_image_classification/config/defaults.py'.



**Example 1**: Run the Certifiable Attack against VGG model on CIFAR10

```
python attack.py --config "./configs/attack/cifar10/untargeted/unrestricted/vgg_CertifiedAttack.yaml" device cuda:0
```

**Example 2**: Run the Certifiable Attack against ResNet model on CIFAR100 under Blacklight

```
python attack.py --config "./configs/attack/cifar100_blacklight/untargeted/unrestricted/resnet_CertifiedAttack.yaml" device cuda:0
```

**Example 3**: Run the Certifiable Attack against ResNet model on ImageNet under RAND Pre-processing defense

```
python attack.py --config "./configs/attack/imagenet_RAND/untargeted/unrestricted/resnet_CertifiedAttack.yaml" device cuda:0
```

**Example 4**: Run the Certifiable Attack against WRN model on CIFAR10 under RAND Post-processing defense

```
python attack.py --config "./configs/attack/cifar10_post_RAND/untargeted/unrestricted/wrn_CertifiedAttack.yaml" device cuda:0
```

**Example 5**: Run the Certified Attack against Adversarial Trained VGG model on CIFAR10

```
python attack.py --config "./configs/attack/cifar10_AT/untargeted/unrestricted/resnet_CertifiedAttack_l2.yaml" device cuda:0
```

**Example 6**: Train the ResNext model on CIFAR10

```
python train.py --config "./configs/cifar10/resnext.yaml"
```

**Example 7**: Evaluate the trained VGG model on CIFAR10

```
python evaluate.py --config "./configs/evaluate/vgg.yaml"
```

**Example 8**: Train the ResNet model on CIFAR100 using Adversarial Training

```
python train.py --config "./configs/AT/cifar100/resnet_linf.yaml" device "cuda:2"
```