import yacs.config
from attacks.score.nes_attack import NESAttack
from attacks.score.bandit_attack import BanditAttack
from attacks.score.zo_sign_sgd_attack import ZOSignSGDAttack
from attacks.score.sign_attack import SignAttack
from attacks.score.simple_attack import SimpleAttack
from attacks.score.square_attack import SquareAttack
from attacks.score.parsimonious_attack import ParsimoniousAttack

from attacks.decision.sign_opt_attack import SignOPTAttack
from attacks.decision.hsja_attack import HSJAttack
from attacks.decision.geoda_attack import GeoDAttack
from attacks.decision.opt_attack import OptAttack
from attacks.decision.evo_attack import EvolutionaryAttack
from attacks.decision.sign_flip_attack import SignFlipAttack
from attacks.decision.rays_attack import RaySAttack
from attacks.decision.boundary_attack import BoundaryAttack
from attacks.sparse_attack.spaevo_attack import SpaEvoAtt
from attacks.sparse_attack.pointwise_attack import PointWiseAtt
from attacks.certified_attack.certifiedattack import CertifiedAttack
from attacks.certified_attack.pdf_functions import *
def get_attack(config: yacs.config.CfgNode):

    if config.test.batch_size!=1:
        print("The batch size for attack evaluation must be 1! Otherwise the evaluation is incorrect!")
        raise NotImplementedError

    if config.attack.name == "CertifiedAttack":

        if config.attack.CertifiedAttack.pdf == 'Gaussian':
            noise_distribution = Gaussian
        elif config.attack.CertifiedAttack.pdf == 'Laplace':
            noise_distribution = Laplace_iid
        elif config.attack.CertifiedAttack.pdf == 'Cauthy':
            '''variance=0.25, args.pdf_args = [-1,0.01969]'''
            noise_distribution = cauthy_iid
        elif config.attack.CertifiedAttack.pdf == 'Hsecant':
            '''variance=0.25, args.pdf_args = [-1, 0.1592]'''
            noise_distribution = H_secant
        elif config.attack.CertifiedAttack.pdf == 'GeneralExp':
            '''variance=0.25, args.pdf_args = [-1, 0.2909,1.5]
               variance=0.25, args.pdf_args = [-1, 0.4092, 3]'''
            noise_distribution = Gen_normal
        else:
            raise NotImplementedError

        if config.attack.CertifiedAttack.diffusion:
            from attacks.certified_attack.denoising_diffusion_pytorch import Unet, GaussianDiffusion
            model_ = Unet(
                dim=32,
                dim_mults=(1, 2, 4, 8),
                flash_attn=True
            )

            diffusion = GaussianDiffusion(
                model_,
                image_size=32,
                timesteps=500,
                sampling_timesteps=50  # number of steps
            ).cuda()

            pre_train = torch.load('./attacks/certified_attack/diffusion_models/model-100.pt')
            diffusion.load_state_dict(pre_train['model'])
        else:
            diffusion = None



        attacker=CertifiedAttack(num_classes=config.dataset.n_classes,
                           query_batch_size=config.attack.CertifiedAttack.query_batch,
                           N=config.attack.CertifiedAttack.MonteNum,
                           p=config.attack.CertifiedAttack.p,
                           input_size=config.dataset.image_size*config.dataset.image_size*config.dataset.n_channels,
                           pdf_args=config.attack.CertifiedAttack.pdf_args,
                           pdf=noise_distribution,
                           norm=config.attack.CertifiedAttack.norm,
                           initialization=config.attack.CertifiedAttack.initialization,
                           shifting=config.attack.CertifiedAttack.shifting,
                           diffusion=diffusion,
                           device=config.device,
                           max_query=config.attack.max_loss_queries,
                           epsilon=config.attack.epsilon,
                           pert_norm=config.attack.p,
                           blacklight=config.defense.blacklight,
                           rand_sigma=config.defense.sigma,
                           post_sigma=config.defense.post_sigma)


    elif config.attack.name=="PointWise":
        from pytorch_image_classification.datasets import create_dataset
        dataset = create_dataset(config, is_train=False)
        if config.attack.test_sample>0:
            torch.manual_seed(config.attack.test_sample_seed)
            indices = torch.randperm(len(dataset)).tolist()[:config.attack.test_sample]
            dataset = torch.utils.data.Subset(dataset, indices)

        attacker=PointWiseAtt(targeted=config.attack.target,dataset=dataset,query=config.attack.max_loss_queries,blacklight=config.defense.blacklight,device=config.device,rand_sigma=config.defense.sigma,post_sigma=config.defense.post_sigma)

    elif config.attack.name == "SparseEvo":

        from pytorch_image_classification.datasets import create_dataset
        dataset = create_dataset(config, is_train=False)
        if config.attack.test_sample>0:
            torch.manual_seed(config.attack.test_sample_seed)
            indices = torch.randperm(len(dataset)).tolist()[:config.attack.test_sample]
            dataset = torch.utils.data.Subset(dataset, indices)

        attack_config = {
            "max_query": config.attack.max_loss_queries,
            "n": config.attack.SparseEvo.n_pix,
            "pop_size": config.attack.SparseEvo.pop_size,
            "cr": config.attack.SparseEvo.cr,
            "mu": config.attack.SparseEvo.mu,
            "seed": config.attack.SparseEvo.seed,
            "target": config.attack.target,
            "device": config.device,
            "blacklight": config.defense.blacklight,
            "dataset":dataset,
            "rand_sigma":config.defense.sigma,
            "post_sigma":config.defense.post_sigma
        }
        attacker = SpaEvoAtt(**attack_config)

    elif config.attack.name=="NES":
        attack_config= {
            "batch_size": config.test.batch_size,
            "name": "NES",
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "fd_eta": config.attack.NES.fd_eta,
            "lr": config.attack.NES.lr,
            "q": config.attack.NES.q,
            "max_loss_queries": config.attack.max_loss_queries,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight":config.defense.blacklight,
            "sigma":config.defense.sigma,
            "post_sigma":config.defense.post_sigma
        }
        attacker = NESAttack(**attack_config)
    elif config.attack.name=="ZOSignSGD":
        attack_config={
            "batch_size" : config.test.batch_size,
            "name": "zosignsgd",
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "fd_eta": config.attack.ZOSignSGD.fd_eta,
            "lr": config.attack.ZOSignSGD.lr,
            "q": config.attack.ZOSignSGD.q,
            "max_loss_queries": config.attack.max_loss_queries,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma

        }
        attacker = ZOSignSGDAttack(**attack_config)

    elif config.attack.name=="Bandit":
        attack_config= {
            "batch_size" : config.test.batch_size,
            "name": "Bandit",
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "lr": config.attack.Bandit.lr,
            "fd_eta": config.attack.Bandit.fd_eta,
            "prior_lr": config.attack.Bandit.prior_lr,
            "prior_size": config.attack.Bandit.prior_size,
            "data_size": config.dataset.image_size,
            "prior_exploration": config.attack.Bandit.prior_exploration,
            "max_loss_queries": config.attack.max_loss_queries,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
          }
        attacker = BanditAttack(**attack_config)
    elif config.attack.name=="Sign":
        attack_config={
            "batch_size" : config.test.batch_size,
            "name": "Sign",
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "fd_eta": config.attack.Sign.fd_eta,
            "max_loss_queries": config.attack.max_loss_queries,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
          }
        attacker = SignAttack(**attack_config)
    elif config.attack.name=="Simple":
        attack_config = {
            "batch_size":config.test.batch_size,
            "epsilon":config.attack.epsilon,
            "p":config.attack.p[1:],
            "delta":config.attack.Simple.delta,
            "max_loss_queries":config.attack.max_loss_queries,
            "name":"SimBA",
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
        }
        attacker = SimpleAttack(**attack_config)

    elif config.attack.name=="Parsimonious":
        attack_config={
            "batch_size": config.test.batch_size,
            "name": "ECO",
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "max_loss_queries": config.attack.max_loss_queries,
            "EOT": config.attack.Parsimonious.EOT,
            "block_size": config.attack.Parsimonious.block_size,
            "block_batch_size": config.attack.Parsimonious.block_batch_size,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type": config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
        }
        attacker = ParsimoniousAttack(**attack_config)

    elif config.attack.name=="Square":
        attack_config={
            "batch_size" : config.test.batch_size,
            "name": "Square",
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "p_init": config.attack.Square.p_init,
            "max_loss_queries": config.attack.max_loss_queries,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
          }
        attacker = SquareAttack(**attack_config)

    elif config.attack.name=="SignOPT":
        attack_config={
            "batch_size": config.test.batch_size,
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "alpha": config.attack.SignOPT.alpha,
            "beta": config.attack.SignOPT.beta,
            "svm": config.attack.SignOPT.svm,
            "momentum": config.attack.SignOPT.momentum,
            "max_queries": config.attack.max_loss_queries,
            "k": config.attack.SignOPT.k,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
          }
        attacker = SignOPTAttack(**attack_config)
    elif config.attack.name=="HSJ":
        attack_config={
            "batch_size": config.test.batch_size,
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "max_queries": config.attack.max_loss_queries,
            "gamma": config.attack.HSJ.gamma,
            "stepsize_search": config.attack.HSJ.stepsize_search,
            "max_num_evals": config.attack.HSJ.max_num_evals,
            "init_num_evals": config.attack.HSJ.init_num_evals,
            "EOT": config.attack.HSJ.EOT,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
          }
        attacker= HSJAttack(**attack_config)
    elif config.attack.name == "GeoDA":
        attack_config={
            "batch_size": config.test.batch_size,
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "max_queries": config.attack.max_loss_queries,
            "sub_dim": config.attack.GeoDA.sub_dim,
            "tol": config.attack.GeoDA.tol,
            "alpha": config.attack.GeoDA.alpha,
            "mu": config.attack.GeoDA.mu,
            "search_space": config.attack.GeoDA.search_space,
            "grad_estimator_batch_size": config.attack.GeoDA.grad_estimator_batch_size,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
          }
        attacker = GeoDAttack(**attack_config)
    elif config.attack.name == "Opt":
        attack_config={
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "alpha": config.attack.Opt.alpha,
            "beta": config.attack.Opt.beta,
            "max_queries": config.attack.max_loss_queries,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
          }
        attacker = OptAttack(**attack_config)
    elif config.attack.name == "Evolutionary":
        attack_config={
            "batch_size": config.test.batch_size,
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "max_queries": config.attack.max_loss_queries,
            "sub": config.attack.Evolutionary.sub,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
          }
        attacker = EvolutionaryAttack(**attack_config)
    elif config.attack.name == "SignFlip":
        attack_config= {
            "batch_size": config.test.batch_size,
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "resize_factor": config.attack.resize_factor,
            "max_queries": config.attack.max_loss_queries,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
        }
        attacker = SignFlipAttack(**attack_config)
    elif config.attack.name == "RayS":
        attack_config= {
            "batch_size": config.test.batch_size,
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "max_queries": config.attack.max_loss_queries,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
        }
        attacker = RaySAttack(**attack_config)
    elif config.attack.name == "Boundary":
        attack_config={
            "batch_size": config.test.batch_size,
            "epsilon": config.attack.epsilon,
            "p": config.attack.p[1:],
            "max_queries": config.attack.max_loss_queries,
            "steps": config.attack.Boundary.steps,
            "spherical_step": config.attack.Boundary.spherical_step,
            "source_step": config.attack.Boundary.source_step,
            "source_step_convergance": config.attack.Boundary.source_step_convergance,
            "step_adaptation": config.attack.Boundary.step_adaptation,
            "update_stats_every_k": config.attack.Boundary.update_stats_every_k,
            "ub": config.attack.ub,
            "lb": config.attack.lb,
            "target": config.attack.target,
            "target_type":config.attack.target_type,
            "device":config.device,
            "blacklight": config.defense.blacklight,
            "sigma": config.defense.sigma,
            "post_sigma":config.defense.post_sigma
        }
        attacker = BoundaryAttack(**attack_config)

    else:
        raise NotImplementedError

    return attacker