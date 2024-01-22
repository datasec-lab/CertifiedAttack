# from art.defences.preprocessor import FeatureSqueezing
# from art.estimators.classification import PyTorchClassifier,BlackBoxClassifier,BlackBoxClassifierNeuralNetwork
# from art.estimators.estimator import BaseEstimator
# from art.attacks.evasion import *
import argparse
import os
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture

from tqdm import tqdm
from pdf_functions import *
import time

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
from attacks.utils.compute import l2_proj_maker

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--model_path', type=str, default='./model_saved/ResNet101_CIFAR10_our_2,0.707_best.pth')
parser.add_argument('--attack_method',type=str,choices=["NES","ZOSignSGD","Bandit","Sign","Simple", "Square", "Parsimonious", "DPDAttack", "SignOPT", "HSJ", "GeoDA","Opt", "Evolutionary","SignFlip", "RayS", "Boundary"])
parser.add_argument('--samples_begin', type=int, default=0)
parser.add_argument('--samples_end', type=int, default=500)

args = parser.parse_args()

if __name__ == '__main__':

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # torch.cuda.set_device(int(args.gpu))

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)

    checkpoint = torch.load(args.model_path)
    try:
        model.load_state_dict(checkpoint)
    except:
        model.load_state_dict(checkpoint['state_dict'])

    criterion = CrossEntropyLoss().cuda()

    print('start')
    correct=0
    total=0
    model.eval()


    queries=[]
    times=[]
    distances=[]
    cert_suc=0
    cert_total=0
    ave_dis=[]

    if args.attack_method=="NES":
        attack_config= {
            "model":model,
            "batch_size": args.batch,
            "name": "NES",
            "epsilon": 8/255,
            "p": "inf",
            "fd_eta": 1.6/255,
            "lr": 1.6/255,
            "q": 15,
            "max_loss_queries": 10000,
            "ub":1.0,
            "lb":0.0

        }
        attacker = NESAttack(**attack_config)
    elif args.attack_method=="ZOSignSGD":
        attack_config={
            "model":model,
            "batch_size" : args.batch,
            "name": "zosignsgd",
            "epsilon": 8/255,
            "p": "inf",
            "fd_eta": 1.6/255,
            "lr": 1.6/255,
            "q": 30,
            "max_loss_queries": 10000,
            "ub": 1.0,
            "lb": 0.0
          }
        attacker = ZOSignSGDAttack(**attack_config)

    elif args.attack_method=="Bandit":
        attack_config= {
            "model":model,
            "batch_size" : args.batch,
            "name": "Bandit",
            "epsilon": 8/255,
            "p": "inf",
            "lr": 1.6/255,
            "fd_eta": 1.6/255,
            "prior_lr": 0.1,
            "prior_size": 20,
            "data_size": 32,
            "prior_exploration": 0.1,
            "max_loss_queries": 10000,
            "ub": 1.0,
            "lb": 0.0
          }
        attacker = BanditAttack(**attack_config)
    elif args.attack_method=="Sign":
        attack_config={
            "model": model,
            "batch_size" : args.batch,
            "name": "Sign",
            "epsilon": 8/255,
            "p": "inf",
            "fd_eta": 8/255,
            "max_loss_queries": 10000,
            "ub": 1.0,
            "lb": 0.0
          }
        attacker = SignAttack(**attack_config)
    elif args.attack_method=="Simple":
        attack_config = {
            "model": model,
            "batch_size":args.batch,
            "epsilon":8/255,
            "p":"inf",
            "delta":1/255,
            "max_loss_queries":10000,
            "name":"SimBA",
            "ub": 1.0,
            "lb": 0.0
        }
        attacker = SimpleAttack(**attack_config)
    elif args.attack_method=="Square":
        attack_config={
            "model":model,
            "batch_size" : args.batch,
            "name": "Square",
            "epsilon": 8/255,
            "p": "inf",
            "p_init": 0.05,
            "max_loss_queries": 10000,
            "ub": 1.0,
            "lb": 0.0
          }
        attacker = SquareAttack(**attack_config)
    elif args.attack_method=="Parsimonious":
        attack_config={
            "model":model,
            "batch_size" : args.batch,
            "name": "ECO",
            "epsilon": 8/255,
            "p": "inf",
            "max_loss_queries": 10000,
            "EOT": 1,
            "block_size": 4,
            "block_batch_size": args.batch,
            "ub": 1.0,
            "lb": 0.0
          }
        attacker = ParsimoniousAttack(**attack_config)
    else:
        raise NotImplementedError

    times=[]
    ave_dis=[]
    adv=0
    total=0
    num =0
    # np.save('./query_num', num)

    data_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    for i, (x_batch, y_batch) in enumerate(tqdm(data_loader)):

        # print('attack sample {}'.format(i))
        # (x, y) = test_dataset[i]
        # file_name='./adv_samples/{}_{}_init_{}_L{}_sigma{}_p{}_{}{}'.format(args.dataset,args.initialization,args.shifting,args.norm,args.pdf_args[1], args.p, args.pdf, args.note)+'/sample{}.np.npy'.format(i)
        x_batch = x_batch.numpy().astype(np.float32)
        # y_batch=torch.tensor([y])
        time_begin=time.time()



        if args.attack_method in ["SignOPTAttack", "HSJAttack", "GeoDAttack", "OptAttack", "EvolutionaryAttack",
                                     "SignFlipAttack", "RaySAttack", "BoundaryAttack"]:
            logs_dict,AE = attacker.run(x_batch, y_batch, model, target=False, dset=test_dataset)
        else:
            logs_dict,AE = attacker.run(x_batch, y_batch)

        # AE = attacker.run(x.unsqueeze(0).numpy().astype(np.float32), torch.tensor([y]))
        time_end=time.time()

        # AE=AE.cpu().numpy()
        # diffs = AE - x_batch
        # output=model(torch.from_numpy(AE).cuda().float())
        # pred = torch.max(output, 1)[1]
        # if pred!=y_batch:
        #     adv+=1
        #     times.append(time_end - time_begin)
        #     ave_dis.append(np.linalg.norm(diffs.reshape(1, -1), ord=2, axis=1))
        # total+=1
        #
        #
        #
        # if not os.path.exists('./adv_samples/{}/'.format(args.attack_method)):
        #     os.mkdir('./adv_samples/{}/'.format(args.attack_method))
        # np.save('./adv_samples/{}/sample{}.np.npy'.format(args.attack_method,i),AE)
        # np.save('./results/{}_times'.format(args.attack_method),times)
        # np.save('./results/{}_pert'.format(args.attack_method),ave_dis)
        # print('asr: {}, average time: {}, average pert: {}'.format(adv/total*100,np.mean(times),np.mean(ave_dis)))
        # print(np.load('./query_num.npy')/total)
        print(attacker.result())
