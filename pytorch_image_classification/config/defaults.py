from .config_node import ConfigNode

config = ConfigNode()

config.device = 'cuda'
# cuDNN

config.defense = ConfigNode()
config.defense.blacklight=False
config.defense.sigma=0.0
config.defense.post_sigma=0.0
config.defense.TRADES=False

config.attack = ConfigNode()
config.attack.name='Square'
config.attack.epsilon=8.0
config.attack.p = 'linf'
config.attack.max_loss_queries=10000
config.attack.ub = 255.0
config.attack.lb = 0.0
config.attack.target= False
config.attack.target_type="median"
config.attack.test_sample=-1
config.attack.test_sample_seed=42


config.attack.CertifiedAttack=ConfigNode()
config.attack.CertifiedAttack.diffusion=False
config.attack.CertifiedAttack.pdf="Gaussian"
config.attack.CertifiedAttack.MonteNum=50
config.attack.CertifiedAttack.query_batch=50
config.attack.CertifiedAttack.p=0.9
config.attack.CertifiedAttack.pdf_args=[-1,0.01]
config.attack.CertifiedAttack.norm=2
config.attack.CertifiedAttack.initialization="bin_search"
config.attack.CertifiedAttack.shifting="geo"

config.attack.NES=ConfigNode()
config.attack.NES.fd_eta=2.0
config.attack.NES.lr=30.0
config.attack.NES.q=15

config.attack.ZOSignSGD=ConfigNode()
config.attack.ZOSignSGD.fd_eta=2.0
config.attack.ZOSignSGD.lr=2.5
config.attack.ZOSignSGD.q=30

config.attack.Bandit=ConfigNode()
config.attack.Bandit.lr= 30
config.attack.Bandit.fd_eta= 2.0
config.attack.Bandit.prior_lr= 0.1
config.attack.Bandit.prior_size= 20
config.attack.Bandit.data_size=32
config.attack.Bandit.prior_exploration=0.1

config.attack.Sign=ConfigNode()
config.attack.Sign.fd_eta=8.0

config.attack.Simple=ConfigNode()
config.attack.Simple.delta=15

config.attack.Parsimonious=ConfigNode()
config.attack.Parsimonious.EOT=1
config.attack.Parsimonious.block_size=4
config.attack.Parsimonious.block_batch_size=64

config.attack.Square=ConfigNode()
config.attack.Square.p_init=0.05

config.attack.SignOPT=ConfigNode()
config.attack.SignOPT.alpha=0.2
config.attack.SignOPT.beta=0.1
config.attack.SignOPT.svm=False
config.attack.SignOPT.momentum=0
config.attack.SignOPT.k=200

config.attack.HSJ=ConfigNode()
config.attack.HSJ.gamma=50
config.attack.HSJ.stepsize_search="geometric_progression"
config.attack.HSJ.max_num_evals=10000
config.attack.HSJ.init_num_evals=100
config.attack.HSJ.EOT=1

config.attack.GeoDA=ConfigNode()
config.attack.GeoDA.sub_dim=30
config.attack.GeoDA.tol=0.0001
config.attack.GeoDA.alpha=0.0002
config.attack.GeoDA.mu=0.6
config.attack.GeoDA.search_space="sub"
config.attack.GeoDA.grad_estimator_batch_size=256

config.attack.Opt=ConfigNode()
config.attack.Opt.alpha=0.2
config.attack.Opt.beta=0.001

config.attack.Evolutionary=ConfigNode()
config.attack.Evolutionary.sub=2

config.attack.SignFlip=ConfigNode()
config.attack.resize_factor=1.0

config.attack.RayS=ConfigNode()

config.attack.Boundary=ConfigNode()
config.attack.Boundary.steps=25000
config.attack.Boundary.spherical_step=0.01
config.attack.Boundary.source_step=0.01
config.attack.Boundary.source_step_convergance=1e-7
config.attack.Boundary.step_adaptation=1.5
config.attack.Boundary.update_stats_every_k=10

config.attack.SparseEvo=ConfigNode()
config.attack.SparseEvo.n_pix = 4
config.attack.SparseEvo.pop_size = 10
config.attack.SparseEvo.cr = 0.5
config.attack.SparseEvo.mu = 0.04
config.attack.SparseEvo.seed = 0


config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False

config.dataset = ConfigNode()
config.dataset.name = 'CIFAR10'
config.dataset.dataset_dir = '/mnt/data/imagenet'
config.dataset.image_size = 32
config.dataset.n_channels = 3
config.dataset.n_classes = 10
config.dataset.normalize = False

config.model = ConfigNode()
# options: 'cifar', 'imagenet'
# Use 'cifar' for small input images
config.model.type = 'cifar'
config.model.name = 'resnet_preact'
config.model.init_mode = 'kaiming_fan_out'
config.model.normalize_layer = True

config.model.vgg = ConfigNode()
config.model.vgg.n_channels = [64, 128, 256, 512, 512]
config.model.vgg.n_layers = [2, 2, 3, 3, 3]
config.model.vgg.use_bn = True

config.model.resnet = ConfigNode()
config.model.resnet.depth = 110  # for cifar type model
config.model.resnet.n_blocks = [2, 2, 2, 2]  # for imagenet type model
config.model.resnet.block_type = 'basic'
config.model.resnet.initial_channels = 16

config.model.resnet_preact = ConfigNode()
config.model.resnet_preact.depth = 110  # for cifar type model
config.model.resnet_preact.n_blocks = [2, 2, 2, 2]  # for imagenet type model
config.model.resnet_preact.block_type = 'basic'
config.model.resnet_preact.initial_channels = 16
config.model.resnet_preact.remove_first_relu = False
config.model.resnet_preact.add_last_bn = False
config.model.resnet_preact.preact_stage = [True, True, True]

config.model.wrn = ConfigNode()
config.model.wrn.depth = 28  # for cifar type model
config.model.wrn.initial_channels = 16
config.model.wrn.widening_factor = 10
config.model.wrn.drop_rate = 0.0

config.model.densenet = ConfigNode()
config.model.densenet.depth = 100  # for cifar type model
config.model.densenet.n_blocks = [6, 12, 24, 16]  # for imagenet type model
config.model.densenet.block_type = 'bottleneck'
config.model.densenet.growth_rate = 12
config.model.densenet.drop_rate = 0.0
config.model.densenet.compression_rate = 0.5

config.model.pyramidnet = ConfigNode()
config.model.pyramidnet.depth = 272  # for cifar type model
config.model.pyramidnet.n_blocks = [3, 24, 36, 3]  # for imagenet type model
config.model.pyramidnet.initial_channels = 16
config.model.pyramidnet.block_type = 'bottleneck'
config.model.pyramidnet.alpha = 200

config.model.resnext = ConfigNode()
config.model.resnext.depth = 29  # for cifar type model
config.model.resnext.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnext.initial_channels = 64
config.model.resnext.cardinality = 8
config.model.resnext.base_channels = 4

config.model.shake_shake = ConfigNode()
config.model.shake_shake.depth = 26  # for cifar type model
config.model.shake_shake.initial_channels = 96
config.model.shake_shake.shake_forward = True
config.model.shake_shake.shake_backward = True
config.model.shake_shake.shake_image = True

config.model.se_resnet_preact = ConfigNode()
config.model.se_resnet_preact.depth = 110  # for cifar type model
config.model.se_resnet_preact.initial_channels = 16
config.model.se_resnet_preact.se_reduction = 16
config.model.se_resnet_preact.block_type = 'basic'
config.model.se_resnet_preact.initial_channels = 16
config.model.se_resnet_preact.remove_first_relu = False
config.model.se_resnet_preact.add_last_bn = False
config.model.se_resnet_preact.preact_stage = [True, True, True]

config.train = ConfigNode()
config.train.checkpoint = ''
config.train.resume = False
config.train.use_apex = False
# optimization level for NVIDIA apex
# O0 = fp32
# O1 = mixed precision
# O2 = almost fp16
# O3 = fp16
config.train.precision = 'O0'
config.train.batch_size = 128
config.train.subdivision = 1
# optimizer (options: sgd, adam, lars, adabound, adaboundw)
config.train.optimizer = 'sgd'
config.train.base_lr = 0.1
config.train.momentum = 0.9
config.train.nesterov = True
config.train.weight_decay = 1e-4
config.train.no_weight_decay_on_bn = False
config.train.gradient_clip = 0.0
config.train.start_epoch = 0
config.train.seed = 0
config.train.val_first = True
config.train.val_period = 1
config.train.val_ratio = 0.0
config.train.use_test_as_val = True

config.train.output_dir = 'experiments/exp00'
config.train.log_period = 100
config.train.checkpoint_period = 10

config.train.use_tensorboard = False
config.tensorboard = ConfigNode()
config.tensorboard.train_images = False
config.tensorboard.val_images = False
config.tensorboard.model_params = False

# optimizer
config.optim = ConfigNode()
# Adam
config.optim.adam = ConfigNode()
config.optim.adam.betas = (0.9, 0.999)
# LARS
config.optim.lars = ConfigNode()
config.optim.lars.eps = 1e-9
config.optim.lars.threshold = 1e-2
# AdaBound
config.optim.adabound = ConfigNode()
config.optim.adabound.betas = (0.9, 0.999)
config.optim.adabound.final_lr = 0.1
config.optim.adabound.gamma = 1e-3

# scheduler
config.scheduler = ConfigNode()
config.scheduler.epochs = 160
# warm up (options: none, linear, exponential)
config.scheduler.warmup = ConfigNode()
config.scheduler.warmup.type = 'none'
config.scheduler.warmup.epochs = 0
config.scheduler.warmup.start_factor = 1e-3
config.scheduler.warmup.exponent = 4
# main scheduler (options: constant, linear, multistep, cosine, sgdr)
config.scheduler.type = 'multistep'
config.scheduler.milestones = [80, 120]
config.scheduler.lr_decay = 0.1
config.scheduler.lr_min_factor = 0.001
config.scheduler.T0 = 10
config.scheduler.T_mul = 1.

# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.num_workers = 2
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = False
config.train.dataloader.non_blocking = False

# validation data loader
config.validation = ConfigNode()
config.validation.batch_size = 256
config.validation.dataloader = ConfigNode()
config.validation.dataloader.num_workers = 2
config.validation.dataloader.drop_last = False
config.validation.dataloader.pin_memory = False
config.validation.dataloader.non_blocking = False

# distributed
config.train.distributed = False
config.train.dist = ConfigNode()
config.train.dist.backend = 'nccl'
config.train.dist.init_method = 'env://'
config.train.dist.world_size = -1
config.train.dist.node_rank = -1
config.train.dist.local_rank = 0
config.train.dist.use_sync_bn = False

config.augmentation = ConfigNode()
config.augmentation.use_random_crop = True
config.augmentation.use_random_horizontal_flip = True
config.augmentation.use_cutout = False
config.augmentation.use_random_erasing = False
config.augmentation.use_dual_cutout = False
config.augmentation.use_mixup = False
config.augmentation.use_ricap = False
config.augmentation.use_cutmix = False
config.augmentation.use_label_smoothing = False

config.augmentation.random_crop = ConfigNode()
config.augmentation.random_crop.padding = 4
config.augmentation.random_crop.fill = 0
config.augmentation.random_crop.padding_mode = 'constant'

config.augmentation.random_horizontal_flip = ConfigNode()
config.augmentation.random_horizontal_flip.prob = 0.5

config.augmentation.cutout = ConfigNode()
config.augmentation.cutout.prob = 1.0
config.augmentation.cutout.mask_size = 16
config.augmentation.cutout.cut_inside = False
config.augmentation.cutout.mask_color = 0
config.augmentation.cutout.dual_cutout_alpha = 0.1

config.augmentation.random_erasing = ConfigNode()
config.augmentation.random_erasing.prob = 0.5
config.augmentation.random_erasing.area_ratio_range = [0.02, 0.4]
config.augmentation.random_erasing.min_aspect_ratio = 0.3
config.augmentation.random_erasing.max_attempt = 20

config.augmentation.mixup = ConfigNode()
config.augmentation.mixup.alpha = 1.0

config.augmentation.ricap = ConfigNode()
config.augmentation.ricap.beta = 0.3

config.augmentation.cutmix = ConfigNode()
config.augmentation.cutmix.alpha = 1.0

config.augmentation.label_smoothing = ConfigNode()
config.augmentation.label_smoothing.epsilon = 0.1

config.tta = ConfigNode()
config.tta.use_resize = True
config.tta.use_center_crop = True
config.tta.resize = 256

# test config
config.test = ConfigNode()
config.test.checkpoint = ''
config.test.output_dir = ''
config.test.batch_size = 256
# test data loader
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False



def get_default_config():
    return config.clone()
