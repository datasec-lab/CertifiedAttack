import torch
import numpy as np
from attacks.sparse_attack.utils_se import *
from attacks.certified_attack.probabilistic_fingerprint import *

def get_tracker(query, window_size, hash_kept, roundto, step_size, workers):
    tracker = InputTracker(query, window_size, hash_kept, round=roundto, step_size=step_size, workers=workers)
    LOGGER.info("Blacklight detector created.")
    return tracker
# main attack
class SpaEvoAtt():
    def __init__(self,
                n = 4,  
                # 4, 16, 64, 256 only required for uni_rand: 4/(32*32) = 0.004 (CIFAR10)
                # 49, 196, 784, 3136 only required for uni_rand: 196/(224*224) = 0.004 (ImageNet)
                pop_size=10,
                cr=0.9,
                mu=0.01,
                seed = None,
                target=False,
                max_query=10000,
                blacklight=False,
                dataset=None,
                device="cuda:0",
                rand_sigma=0,
                post_sigma=0):

        self.n_pix = n # if uni_rand is used
        self.pop_size = pop_size
        self.cr = cr
        self.mu = mu
        self.seed = seed
        self.target = target
        self.device=torch.device(device)
        self.max_query=max_query
        self.dataset=dataset

        self.blacklight=blacklight
        self.blacklight_detection=0
        self.blacklight_count=0
        self.blacklight_count_total=0
        self.blacklight_cover_list=[]
        self.blacklight_total_sample=0
        self.blacklight_query_to_detect=0
        self.blacklight_query_to_detect_list=[]
        self.distances=[]
        self.query_list=[]
        self.rand_sigma=rand_sigma
        self.post_sigma=post_sigma
        self.post_noise=0
    def convert1D_to_2D(self,idx,wi):
        c1 = idx //wi
        c2 = idx - c1 * wi
        return c1, c2

    def convert2D_to_1D(self,x,y,wi):
        outp = x*wi + y
        return outp

    def masking(self,oimg,timg):
        xo = torch.abs(oimg-timg)
        d = torch.zeros(xo.shape[2],xo.shape[3]).bool().to(self.device)
        for i in range (xo.shape[1]):
            tmp = (xo[0,i]>0.).bool().to(self.device)
            d = tmp | d # "or" => + ; |
        
        wi = oimg.shape[2]
        p = np.where(d.int().cpu().numpy() == 1) # oimg -> reference;'0' => "same as oimg" '1' => 'same as timg'
        out = self.convert2D_to_1D(p[0],p[1],wi)

        return out # output pixel coordinates have value same as 'timg'

    def uni_rand(self,oimg,timg,olabel,tlabel):
    
        if self.seed != None:
            np.random.seed(self.seed)

        terminate = False
        nqry = 0
        wi = oimg.shape[2]
        he = oimg.shape[3]
        
        fit = torch.zeros(self.pop_size) + np.inf
        pop = []

        p1 = np.zeros(wi*he).astype(int)
        idxs = self.masking(oimg,timg) #[x for x in range(wi * he)]
        p1[idxs] = 1
        
        if p1.sum()<self.n_pix:
            self.n_pix = p1.sum()        

        for i in range(self.pop_size):
            n = self.n_pix
            cnt = 0
            j = 0
            while True:
                p = p1.copy()
                idx = np.random.choice(idxs, n, replace = False)
                p[idx] = 0
                nqry += 1
                fitness = self.feval(p,oimg,timg,olabel,tlabel)
                    
                if fitness < fit[i]:
                    pop.append(p)
                    fit[i] = fitness
                    break
                elif (n>1):
                    n -= 1
                elif (n == 1):
                    while j < len(idxs):
                        p[idxs[j]] = 0
                        nqry += 1
                        fitness = self.feval(p,oimg,timg,olabel,tlabel)

                        if fitness < fit[i]:
                            pop.append(p)
                            fit[i] = fitness
                            break
                        else:
                            j += 1
                    break

            if (j==len(idxs)-1):
                break
                
        if len(pop)<self.pop_size:
            for i in range(len(pop),self.pop_size):
                pop.append(p1)

        return pop,nqry,fit

    def recombine(self,p0,p1,p2):

        cross_points = np.random.rand(len(p1)) < self.cr # uniform random
        if not np.any(cross_points):
            cross_points[np.random.randint(0, len(p1))] = True
        trial = np.where(cross_points, p1, p2).astype(int)
        trial = np.logical_and(p0,trial).astype(int) 
        return trial

    def mutate(self,p):

        outp = p.copy()
        if p.sum() != 0:
            one = np.where(outp == 1)
            n_px = int(len(one[0])*self.mu)
            if n_px == 0:
                n_px = 1
            idx = np.random.choice(one[0],n_px,replace=False)
            outp[idx] = 0

        return outp

    def modify(self,pop,oimg,timg):
        wi = oimg.shape[2]
        img = timg.clone()
        p = np.where(pop == 0)
        c1,c2 = self.convert1D_to_2D(p[0],wi)
        img[:,:,c1,c2] = oimg[:,:,c1,c2]
        return img

    def feval(self,pop,oimg,timg,olabel,tlabel):

        xp = self.modify(pop,oimg,timg)
        l2 = torch.norm(oimg - xp).cpu().numpy()
        self.RAND_noise=self.rand_sigma * torch.randn_like(xp)
        logits=self.model(xp+self.RAND_noise)
        self.post_noise=self.post_sigma*torch.randn_like(logits)
        pred_label = (logits+self.post_noise).argmax(dim=1)

        if self.blacklight:
            self.blacklight_detect(xp)
            self.blacklight_count_total+=xp.shape[0]
            if self.blacklight_count==0:
                self.blacklight_query_to_detect+=1
        if self.target == True:
            if pred_label == tlabel:
                lc = 0
            else:
                lc = np.inf
        else:
            if pred_label != olabel:
                lc = 0
            else:
                lc = np.inf

        outp = l2 + lc
        return outp 


    def selection(self,x1,f1,x2,f2):

        xo = x1.copy()
        fo = f1
        if f2<f1:
            fo = f2
            xo = x2

        return xo,fo

    def result(self):
        return{
            "average_num_queries":np.mean(self.query_list),
            "failure_rate":0,
            "blacklight_detection_rate":self.blacklight_detection/self.blacklight_total_sample if self.blacklight_total_sample>0 else "None",
            "blacklight_coverage":np.mean(self.blacklight_cover_list),
            "blacklight_query_to_detect":np.mean(self.blacklight_query_to_detect_list),
            "distance":np.mean(self.distances)
        }
    def run(self,x,y,model):

        if not self.target and model(x).argmax(dim=1) != y:
            return x

        if self.blacklight:
            self.tracker = get_tracker(x, window_size=20, hash_kept=50, roundto=50, step_size=1, workers=5)
            self.blacklight_count = 0
            self.blacklight_query_to_detect = 0
            self.blacklight_count_total = 0
            self.blacklight_total_sample += 1
        oimg = x
        olabel = y
        seed = 0
        nquery = 0
        self.model = model
        if self.target:
            raise NotImplementedError
        else:
            init_mode = 'salt_pepper_att'  # 'gauss_rand' #'salt_pepper'
            timg, nqry, _ = gen_starting_point(model, oimg, olabel, seed, self.dataset, init_mode)
            tlabel = None
            nquery += nqry
        adv, nquery, distances = self.evo_perturb(oimg, timg, olabel, tlabel)
        self.query_list.append(nquery)
        self.distances.append(
            torch.linalg.norm((adv - x).reshape(x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]), ord=2).cpu().data)

        if self.blacklight:
            if self.blacklight_count > 0:
                self.blacklight_detection += 1
                self.blacklight_cover_list.append(self.blacklight_count / self.blacklight_count_total)
                self.blacklight_query_to_detect_list.append(self.blacklight_query_to_detect)
            print("blacklight detection rate: {}, cover: {}, query to detect: {} ".format(self.blacklight_detection/self.blacklight_total_sample if self.blacklight_total_sample>0 else "None",np.mean(self.blacklight_cover_list),np.mean(self.blacklight_query_to_detect_list)))


        print(self.result())
        return adv+self.rand_sigma * torch.randn_like(adv)

    def blacklight_detect(self,queries):
        ####for blacklight
        threshold = 25
        id = 0
        match_list = []
        for query in queries.detach().cpu().numpy():
            match_num = self.tracker.add_img(query)
            match_list.append(match_num)
            if (match_num > threshold):
                # LOGGER.info(
                #     "Image: {}, max match: {}, attack_query: {}".format(id, match_num, match_num > threshold))
                self.blacklight_count += 1
                # print("blacklight_success:{}".format(self.blacklight_count))
            # print(query)
            id += 1
    def evo_perturb(self,oimg,timg,olabel,tlabel):

        max_query=self.max_query
        # 0. variable init
        if self.seed != None:
            np.random.seed(self.seed)

        D = torch.zeros(max_query+500,dtype=int).to(self.device)
        wi = oimg.shape[3]
        he = oimg.shape[2]
        n_dims = wi * he

        # 1. population init
        idxs = self.masking(oimg,timg) #[x for x in range(wi * he)]
        if len(idxs)>1: # more than 1 diff pixel
            pop, nqry,fitness = self.uni_rand(oimg,timg,olabel,tlabel)
            
            if len(pop)>0:
                # 2. find the worst & best
                rank = np.argsort(fitness) 
                best_idx = rank[0].item()
                worst_idx = rank[-1].item()

                # ====== record ======
                D[:nqry] = l0(self.modify(pop[best_idx],oimg,timg),oimg)
                # ====================
                
                # 3. evolution
                while True:
                    # a. Crossover (recombine)
                    idxs = [idx for idx in range(self.pop_size) if idx != best_idx]
                    id1, id2 = np.random.choice(idxs, 2, replace = False)
                    offspring = self.recombine(pop[best_idx],pop[id1],pop[id2])

                    # b. mutation (diversify)
                    offspring = self.mutate(offspring)
                        
                    # c. fitness evaluation
                    fo = self.feval(offspring,oimg,timg,olabel,tlabel)
                        
                    # d. select
                    pop[worst_idx],fitness[worst_idx] = self.selection(pop[worst_idx],fitness[worst_idx],offspring,fo)
                        
                    # e. update best and worst
                    rank = np.argsort(fitness)
                    best_idx = rank[0].item()
                    worst_idx = rank[-1].item()

                    # ====== record ======
                    D[nqry] = l0(self.modify(pop[best_idx],oimg,timg),oimg)
                    nqry += 1 
                    # ====================
                    
                    if nqry % 5000 == 0:
                        print(pop[best_idx].sum().item(),nqry,self.model(self.modify(pop[best_idx],oimg,timg)).argmax(dim=1))
                    if nqry > max_query:
                        break
                
                # ====================

                adv = self.modify(pop[best_idx],oimg,timg)
            else:
                adv = timg
                D[:nqry] = l0(self.modify(pop[best_idx],oimg,timg),oimg)#len(self.masking(oimg,timg))
        else:
            adv = timg
            nqry = 1 # output purpose, not mean number of qry = 1
            D[0] = 1
            
        return adv, nqry, D[:nqry]

