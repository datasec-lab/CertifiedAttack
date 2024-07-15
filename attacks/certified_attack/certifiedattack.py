import os

import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
import torch.nn as nn
import torchvision
import time
from attacks.certified_attack.diffusion_model import denoise
import torchviz


#### for blacklight
from attacks.certified_attack.probabilistic_fingerprint import *

def get_tracker(query, window_size, hash_kept, roundto, step_size, workers):
    tracker = InputTracker(query, window_size, hash_kept, round=roundto, step_size=step_size, workers=workers)
    LOGGER.info("Blacklight detector created.")
    return tracker

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t

class vertical_vector(nn.Module):
    def __init__(self):
        super(vertical_vector,self).__init__()
    def forward(self, vector, pos_samples,x,x_adv):

        vectors=vector.repeat((pos_samples.shape[0], 1, 1, 1))
        sim1=torch.cosine_similarity(vectors.view(vectors.shape[0],-1), pos_samples.view(pos_samples.shape[0],-1))
        loss1=-torch.mean(torch.sqrt(1-torch.square(sim1)))
        c=x-x_adv
        sim2=torch.cosine_similarity(vector.view(1,-1),c.view(1,-1))
        loss2=-sim2

        return loss1+loss2


class perceptual_criteria(nn.Module):
    def __init__(self):
        super(perceptual_criteria, self).__init__()
        resnet34 = torchvision.models.resnet34(pretrained=True)
        self.resnet34 = nn.Sequential(*list([resnet34.conv1, resnet34.bn1, resnet34.relu, resnet34.maxpool,
                                             resnet34.layer1, resnet34.layer2, resnet34.layer3, resnet34.layer4, resnet34.avgpool]))

        for param in self.resnet34.parameters():
            param.requires_grad = False  # Freeze the model parameters

        self.mse = nn.MSELoss()

    def forward(self, adv, org):
        # Ensure model is in eval mode to keep batchnorm and dropout layers fixed
        self.resnet34.eval()

        # Compute the features
        adv_features = self.resnet34(adv)
        org_features = self.resnet34(org)

        # Compute the loss
        loss = self.mse(adv_features, org_features)
        return loss


class CertifiedAttack(object):

    def __init__(self, num_classes: int,query_batch_size:int, N: int, p: float, input_size:int, pdf_args,pdf,norm,initialization,shifting,diffusion=None,device='cuda:0',max_query=10000,epsilon=8.0, pert_norm="inf",blacklight=False,rand_sigma=0,post_sigma=0):
        self.num_classes = num_classes
        self.batch_size=query_batch_size
        self.pdf=pdf
        self.N=N
        self.pdf_args=pdf_args
        self.p=p
        self.input_size=input_size
        if norm==-1:
            self.norm=np.inf
        else:
            self.norm=norm
        self.initialization=initialization
        self.shifting=shifting
        self.diffusion=diffusion
        self.device=torch.device(device)
        self.max_query=max_query
        self.epsilon=epsilon/255.0
        self.pert_norm=pert_norm
        self.query_list=[]
        self.success_list=[]
        self.certified_success=[]
        self.empirical_distance=[]
        self.mean_distance=[]
        self.RPQ_nums=[]
        self.rand_sigma=rand_sigma
        self.post_sigma=post_sigma
        self.post_noise=0

        self.blacklight=blacklight
        self.blacklight_detection=0
        self.blacklight_count=0
        self.blacklight_cover=[]
        self.blacklight_query_to_detect=0
        self.blacklight_query_to_detect_list=[]
        self.RAND_noise=0
    def result(self):
        result={
            'average_num_queries':np.mean(self.query_list),
            'failure_rate':1-np.sum(self.success_list)/len(self.success_list),
            'certified_acc':np.sum(self.certified_success)/len(self.certified_success),
            'mean_distance':np.mean(self.mean_distance),
            'distance':np.mean(self.empirical_distance),
            'blacklight_detection_rate':self.blacklight_detection/len(self.query_list) if self.blacklight else None,
            'blacklight_coverage':np.mean(self.blacklight_cover) if self.blacklight else None,
            'blacklight_query_to_detect': np.mean(self.blacklight_query_to_detect_list) if self.blacklight else None,
            'RPQ': np.mean(self.RPQ_nums)
        }
        return result
    def initial_adv(self, x,y,x_adv=None,step_size = 3/255, eps=18/255):

        adv = torch.randn(x.shape).to(self.device)

        with torch.enable_grad():
            adv.data = x_adv.data
            adv.requires_grad = True
            criterion = perceptual_criteria()
            criterion.to(self.device)


            epsilons = self.noise_sampling(self.N, self.pdf_args).view(self.N,x.shape[1],x.shape[2],x.shape[3])
            for t in range(10):

                # Assuming 'adv' is the original adversarial example tensor with requires_grad set to True
                adv1 = adv.clone().requires_grad_(True)

                num=self.N
                losses=0
                for i in range(np.ceil(num / self.batch_size).astype(int)):
                    this_batch_size = min(self.batch_size, num)
                    num -= this_batch_size
                    batch = adv1.repeat((this_batch_size, 1, 1, 1))
                    noise = epsilons[i * self.batch_size:i * self.batch_size + this_batch_size].float().view(batch.shape)
                    noisy_input=torch.clip(batch+noise,0.0,1.0)
                    loss = criterion(normalize(noisy_input), normalize(x.clone().detach().repeat((this_batch_size,1,1,1))+noise))
                    losses+=loss
                losses.backward()

                adv.data = adv.data + step_size * adv.grad.sign()
                # adv.data=torch.clip(self.clip_adv(x,adv.data),0.0,1.0)
                adv.data = torch.where(adv.data > x.data + eps, x.data + eps, adv.data)
                adv.data = torch.where(adv.data < x.data - eps, x.data - eps, adv.data)
                adv.data.clamp_(0.0, 1.0)

                adv.grad.data.zero_()

        return adv


    def initial_adv_random(self, x,y):
        x_adv=torch.rand(x.shape).to(self.device)
        return x_adv

    def smoothed_query(self,x,x_adv,y):
        """using Monte Carlo method to compute the upper bound and lower bound of
          the probability
          x: the input image
          y: the label"""
        self.base_classifier.eval()

        # N = self.MC_Num
        num_classes = self.num_classes
        alpha = 0.001
        with torch.no_grad():

            x_adv = x_adv.to(self.device)
            # label = y.to(self.device)
            epsilons = self.noise_sampling(self.N, self.pdf_args)

            predictions_g = torch.tensor([]).to(self.device).long()
            pos_samples=torch.tensor([]).to(self.device).long()

            num = self.N
            for i in range(np.ceil(num / self.batch_size).astype(int)):
                this_batch_size = min(self.batch_size, num)
                num -= this_batch_size
                batch = x_adv.repeat((this_batch_size, 1, 1, 1))
                noise = epsilons[i * self.batch_size:i * self.batch_size + this_batch_size].float().view(batch.shape)
                # clip
                noisy_input=torch.clip(batch+noise,0.0,1.0)
                if self.diffusion:
                    #the denoise require input with range (-1,1), change to (0,1) in the next version
                    noisy_input=denoise(noise,batch,self.pdf_args[1],self.diffusion)
                logits=self.base_classifier(noisy_input+self.rand_sigma * torch.randn_like(noisy_input))

                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow(noisy_input[0].cpu().permute(1, 2, 0))
                # plt.axis('off')
                # if not os.path.exists('./paper_utils/visualization/sample'):
                #     os.mkdir('./paper_utils/visualization/sample')
                # img_index=len(os.listdir('./paper_utils/visualization/sample/'))
                # plt.savefig('./paper_utils/visualization/sample/{}.png'.format(img_index), dpi=1000)

                predictions = (logits+self.post_sigma*torch.randn_like(logits)).argmax(1)
                if pos_samples.shape[0]<1000:
                    pos_samples= torch.cat([pos_samples,noisy_input[predictions==y]],0)
                predictions_g = torch.cat([predictions_g, predictions], 0)
            pred = predictions_g.cpu().numpy()
            if pos_samples.shape[0]>1000:
                pos_samples=pos_samples[:1000]
            # print(pred)
            counts_g = np.zeros(num_classes)
            for i in range(num_classes):
                counts_g[i] = (pred == i).sum()
            # print(counts_g)
            NA=np.sum(counts_g)-counts_g[y]
            pABar = proportion_confint(NA, self.N, alpha=2 * alpha, method="beta")[0]

            if self.blacklight:
                self.blacklight_detect(noisy_input)
                if self.blacklight_count==0:
                    self.blacklight_query_to_detect+=1

            return pABar,pos_samples

    def blacklight_detect(self,noisy_input):
        ####for blacklight
        threshold = 25
        id = 0
        match_list = []
        for query in noisy_input.detach().cpu().numpy():
            match_num = self.tracker.add_img(query)
            match_list.append(match_num)
            if (match_num > threshold):
                # LOGGER.info(
                #     "Image: {}, max match: {}, attack_query: {}".format(id, match_num, match_num > threshold))
                self.blacklight_count += 1
                # print("blacklight_success:{}".format(self.blacklight_count))
            # print(query)
            id += 1
    def certify_move(self,x,x_adv,p_adv,pos_samples):

        # print(pos_samples.shape[0])
        if pos_samples.shape[0]==0:
            E=(x-x_adv).reshape(self.input_size)
            E=E/torch.linalg.norm(E,ord=self.norm)
            delta=E
            # print(1)
        else:
            # print(2)
            with torch.enable_grad():
                vert = vertical_vector()
                vert.to(self.device)
                vector=torch.randn_like(x).to(self.device)
                vector.requires_grad = True
                for t in range(20):
                    vert_loss=vert(vector,pos_samples.detach(),x.detach(),x_adv.detach())
                    # print(vert_loss)
                    vert_loss.backward()
                    vector.data = vector.data - 0.05 * vector.grad.sign()
                    vector.data.clamp_(0.0, 1.0)
                    vector.grad.data.zero_()
                    # print(torch.cosine_similarity(vector.view(1,-1),(x-x_adv).view(1,-1)))
                c=x-x_adv
                # print(torch.cosine_similarity(vector.view(1,-1),c.view(1,-1)))
                if torch.cosine_similarity(vector.view(1,-1),c.view(1,-1))<0:
                    vector=-vector
                delta=vector/torch.linalg.norm(vector.view(-1),ord=self.norm)


        epsilons = self.noise_sampling(self.N,self.pdf_args)
        delta_scaled=self.scale_optimization_binary(delta.view(-1),p_adv,epsilons,self.pdf_args)
        if delta_scaled=='failure':
            return 'failure'
        # print(delta_scaled)
        x_adv=x_adv+delta_scaled.view(x_adv.shape)
        x_adv.data.clamp_(0.0, 1.0)
        return x_adv

    def localization(self,x,y):
        t0 = time.time()
        query_count = 0
        p_adv=0

        if self.initialization == 'random':
            n = 0
            while p_adv < self.p and n < 85:
                x_adv = self.initial_adv_random(x, y)
                p_adv, pos_samples = self.smoothed_query(x, x_adv, y)
                query_count += 1
                n += 1
            if n == 85:
                x_adv=None
            print('initialized, queries: {}, time cost: {}'.format(query_count, time.time() - t0))
        elif self.initialization == 'smt_ssp':
            eps = 3 / 255
            x_adv=x
            while p_adv < self.p:
                if eps > 1:
                    x_adv=None
                    break
                x_adv = self.initial_adv(x, y, x_adv=x_adv, eps=eps)
                p_adv, pos_samples = self.smoothed_query(x, x_adv, y)
                query_count += 1
                eps = eps + 3 / 255

            print('initialized, queries: {}, time cost: {}'.format(query_count, time.time() - t0))
        elif self.initialization == 'bin_search':
            n = 0
            while p_adv < self.p and n < 85:
                x_adv = self.initial_adv_random(x, y)
                p_adv, pos_samples = self.smoothed_query(x, x_adv, y)
                query_count += 1
                n += 1
            if n>=85:
                x_adv=None
            else:
                max_iter=15
                tol=0.1
                low = x.clone()
                high = x_adv.clone()
                for _ in range(max_iter):
                    mid = (low + high) / 2
                    _p_adv, _pos_samples = self.smoothed_query(x, mid, y)
                    query_count +=1
                    if _p_adv>=self.p:
                        x_adv = mid
                        p_adv=_p_adv
                        pos_samples=_pos_samples
                        high = mid
                    else:
                        low = mid

                    # Check if the difference between the high and low is less than the tolerance
                    if torch.linalg.norm((high-low).reshape(self.input_size),ord=self.norm) <= tol:
                        break

        else:
            raise NotImplementedError

        return x_adv,query_count,p_adv,pos_samples

    def certify_shifting(self,x,y,x_adv,query_count,p_adv,pos_samples):
        last_adv=x_adv
        if self.shifting=='none':

            return x_adv, query_count

        ############# shifting ###############
        # np.save('./visualization/init_{}',x_adv.cpu().detach().numpy())
        iter=72
        while p_adv>=self.p:
            last_distance=torch.linalg.norm((x_adv-x).reshape(self.input_size),ord=self.norm)
            if self.shifting=='geo':
                x_adv=self.certify_move(x,x_adv,p_adv,pos_samples)
            else:
                print('undefined shifting method: {}'.format(self.shifting))
            if x_adv=='failure':
                return last_adv, query_count
            p_adv,pos_samples = self.smoothed_query(x,x_adv,y)
            query_count+=1
            distance=torch.linalg.norm((x_adv-x).reshape(self.input_size),ord=self.norm)
            radius=torch.linalg.norm((x_adv-last_adv).reshape(self.input_size),ord=self.norm)
            print('move_in, distance={}, p_adv= {}'.format(distance,p_adv))

            if torch.abs(last_distance-distance)<self.pdf_args[1]*0.1:
                # print(x_adv)
                return x_adv, query_count
            if p_adv<self.p:
                # print(last_adv)
                return last_adv, query_count
            if radius>distance:
                # print(x)
                return x, query_count
            if iter<1:
                return last_adv, query_count
            else:
                last_adv = x_adv
                iter-=1
    def adv_certify(self, x,y,model):
        self.base_classifier=model

        x = x.to(self.device)
        ############# initialization ###############
        x_adv,query_count,p_adv,pos_samples=self.localization(x,y)
        if x_adv is None:
            return None,query_count
        else:
            x_adv,query_count= self.certify_shifting(x,y, x_adv, query_count, p_adv, pos_samples)

            return x_adv,query_count


    def run(self,x,y,model):

        if self.blacklight:
            self.tracker = get_tracker(x, window_size= 20, hash_kept= 50, roundto= 50, step_size= 1, workers= 5)
            self.blacklight_count=0
            self.blacklight_query_to_detect=0

        adv_center, certify_query=self.adv_certify(x,y,model)
        attack_query=0
        query=certify_query*self.batch_size+attack_query
        if adv_center is not None:
            if query<self.max_query:
                self.certified_success.append(1)
                self.mean_distance.append(torch.linalg.norm((adv_center-x).reshape(self.input_size),ord=self.norm).cpu().data)
                self.RPQ_nums.append(certify_query)
            while query<self.max_query:
                noise=self.noise_sampling(1,self.pdf_args).float().view(adv_center.shape)
                noisy_input = torch.clip(adv_center + noise, 0.0, 1.0)
                self.RAND_noise=self.rand_sigma * torch.randn_like(noisy_input)
                logits=self.base_classifier(noisy_input+self.RAND_noise)
                self.post_noise =self.post_sigma*torch.randn_like(logits)
                predictions = (logits+self.post_noise).argmax(1)
                if self.blacklight:
                    self.blacklight_detect(noisy_input)
                    if self.blacklight_count==0:
                        self.blacklight_query_to_detect+=1
                attack_query+=1
                query = certify_query * self.batch_size + attack_query
                if predictions!=y:
                    if self.blacklight:
                        self.blacklight_cover.append(self.blacklight_count/query)
                        if self.blacklight_count>0:
                            self.blacklight_detection+=1
                            self.blacklight_query_to_detect_list.append(self.blacklight_query_to_detect)
                    self.query_list.append(query)
                    self.success_list.append(1)
                    self.empirical_distance.append(torch.linalg.norm((noisy_input-x).reshape(self.input_size),ord=self.norm).cpu().data)
                    print(self.result())
                    # import matplotlib.pyplot as plt
                    # plt.figure()
                    # plt.imshow(noisy_input[0].cpu().permute(1, 2, 0))
                    # plt.axis('off')
                    # plt.savefig('./paper_utils/visualization/sample/final.png', dpi=1000)
                    # plt.figure()
                    # plt.imshow(x[0].cpu().permute(1, 2, 0))
                    # plt.axis('off')
                    # sample_index = len(os.listdir('./paper_utils/visualization/'))
                    # plt.savefig('./paper_utils/visualization/sample/ori.png', dpi=1000)
                    # sample_index = len(os.listdir('./paper_utils/visualization/'))
                    # os.rename('./paper_utils/visualization/sample','./paper_utils/visualization/sample{}'.format(sample_index))
                    return noisy_input+self.RAND_noise
        self.certified_success.append(0)
        self.success_list.append(0)
        self.query_list.append(0)
        self.empirical_distance.append(0)
        print(self.result())
        return x+self.RAND_noise

    import torch

    def clip_adv(self,clean_data, adv_examples):
        """
        Clips adversarial examples to a norm ball of radius epsilon around the clean data.

        :param clean_data: PyTorch tensor of clean examples
        :param adv_examples: PyTorch tensor of adversarial examples
        :param epsilon: radius of the norm ball
        :param norm: type of norm ('l2' or 'inf')
        :return: clipped adversarial examples
        """
        if self.pert_norm == 'l2':
            # Calculate the L2 norm of the perturbation
            delta = adv_examples - clean_data
            norm_delta = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1, keepdim=True)
            # Ensure the norm is not zero to avoid division by zero
            norm_delta = torch.where(norm_delta > 0, norm_delta, torch.ones_like(norm_delta))
            # Scale factor should not exceed 1, so we clip it to [0, 1]
            scale = torch.min(self.epsilon / norm_delta, torch.ones_like(norm_delta))
            # Reshape scale to match the dimensions of delta
            scale = scale.view(-1, *([1] * (delta.dim() - 1)))
            clipped_adv = clean_data + delta * scale
        elif self.pert_norm == 'linf':
            # Clip the perturbation to [-epsilon, epsilon] and add to clean samples
            clipped_adv = torch.clamp(adv_examples - clean_data, -self.epsilon, self.epsilon) + clean_data
        else:
            raise ValueError("Norm must be 'l2' or 'inf'")

        return clipped_adv

    def noise_sampling(self,Sample_Num,args):
        """samping Sample_Num*input_size noises from the discrete pdf"""
        pdf, discrete_scaler = self.discrete_pdf_function(args)
        s_pdf = []
        s = 0
        for p in pdf:
            s = s + p
            s_pdf.append(s)

        epsilons = torch.rand(Sample_Num,self.input_size,device=self.device)
        left_bound = 0
        for i, v in enumerate(s_pdf):
            epsilons[torch.logical_and(epsilons < v, epsilons >= left_bound)] = i
            left_bound = v
        return (epsilons - pdf.shape[0] / 2) * discrete_scaler

    def scale_optimization_binary(self,delta,PA,epsilons,args):
        error=0.01
        max_iteration=20
        Lambda_init = torch.mean(torch.abs(epsilons))/2
        Lambda = Lambda_init
        delta_scaler = delta*Lambda_init
        diff_initial = self.compute_K_binary(PA, epsilons, delta_scaler,args)
        n = max_iteration
        if diff_initial >= 0:
            diff = diff_initial
            while diff >= 0 and n > 0:
                Lambda = Lambda * 2
                delta_scaler = delta*Lambda
                diff = self.compute_K_binary(PA,epsilons,delta_scaler,args)
                # print(diff)
                n -= 1
        else:
            diff = diff_initial
            while diff < 0 and n > 0:
                Lambda = Lambda * 1 / 2
                delta_scaler =  delta*Lambda
                diff = self.compute_K_binary(PA,epsilons, delta_scaler,args)
                # print(diff)
                n -= 1
        if n == 0:
            return 'failure'

        if Lambda > Lambda_init:
            Lambdaa = Lambda_init
            Lambdab = Lambda
        else:
            Lambdaa = Lambda
            Lambdab = Lambda_init

        n = max_iteration
        while np.abs(diff) > error or diff <0:
            if n==0:
                break
            Lambda = (Lambdaa + Lambdab) / 2
            delta_scaler = delta*Lambda
            diff = self.compute_K_binary(PA,epsilons,delta_scaler,args)
            if diff >= 0:
                Lambdaa = Lambda
            else:
                Lambdab = Lambda
            n -= 1
        if n == 0:
            return 'failure'
        else:
            return delta_scaler

    def compute_ta_tb_binary(self, PA, epsilons, delta, args):
        epsilons_delta = epsilons - delta
        gammas = torch.prod(self.pdf(epsilons_delta, args) / self.pdf(epsilons, args), dim=1)
        sorted_gammas = torch.sort(gammas, dim=0)[0]
        ta = sorted_gammas[np.ceil(len(gammas) * PA).astype(int)]
        return ta

    def compute_K_binary(self,PA,epsilons, delta,args):
        """"""
        ta= self.compute_ta_tb_binary(PA,epsilons, delta,args)
        epsilons_delta = epsilons + delta
        gammas = torch.prod(self.pdf(epsilons, args) / self.pdf(epsilons_delta, args), dim=1)
        P_ta = (gammas <= ta).sum() / len(gammas)
        K=P_ta-self.p
        return K.cpu()

    def discrete_pdf_function(self,args,discrete_range_factor=5):
        """prepare the discrete version of pdf function"""
        #estimate the sigma
        s=torch.linspace(-5,5,500)
        t=self.pdf(s,args)
        t=t/torch.sum(t*10/500)
        sigma=torch.sqrt(torch.sum(s**2*t*10/500))
        #prepare the discrete pdf
        s_=torch.linspace(-sigma*discrete_range_factor,sigma*discrete_range_factor,1000)
        discrete_pdf=self.pdf(s_,args)
        discrete_pdf=discrete_pdf/torch.sum(discrete_pdf*1)

        discrete_scaler=1/1000*2* sigma * discrete_range_factor

        return discrete_pdf.data, discrete_scaler