#adversarial training with TRADES
nohup python train.py --config "./configs/AT/cifar10/resnet_linf.yaml" device "cuda:0" >./nohup_logs/cifar10_AT_resnet_linf.yaml &
nohup python train.py --config "./configs/AT/cifar10/resnet_l2.yaml" device "cuda:1" >./nohup_logs/cifar10_AT_resnet_l2.yaml &
nohup python train.py --config "./configs/AT/cifar100/resnet_linf.yaml" device "cuda:2" >./nohup_logs/cifar100_AT_resnet_linf.yaml &
nohup python train.py --config "./configs/AT/cifar100/resnet_l2.yaml" device "cuda:3" >./nohup_logs/cifar100_AT_resnet_l2.yaml &
