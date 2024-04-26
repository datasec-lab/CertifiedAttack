import torch
import numpy as np
from attacks.sparse_attack.utils_se import *
import random
from attacks.certified_attack.probabilistic_fingerprint import *


def get_tracker(query, window_size, hash_kept, roundto, step_size, workers):
    tracker = InputTracker(query, window_size, hash_kept, round=roundto, step_size=step_size, workers=workers)
    LOGGER.info("Blacklight detector created.")
    return tracker
# main attack
class PointWiseAtt():

    def __init__(self,
        targeted,dataset,query,blacklight=False,device="cuda:0",rand_sigma=0,post_sigma=0):

        self.targeted = targeted
        self.dataset=dataset
        self.query=query
        self.device=torch.device(device)

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

    def check_adv_status(self,img,olabel,tlabel):
        is_adv = False

        adv_=torch.from_numpy(img).to(self.device)
        self.RAND_noise=self.rand_sigma * torch.randn_like(adv_)
        logits=self.model(adv_+self.RAND_noise)
        self.post_noise=self.post_sigma*torch.randn_like(logits)
        pred_label = (logits+self.post_noise).argmax(dim=1)
        if self.blacklight:
            self.blacklight_detect(adv_)
            self.blacklight_count_total+=adv_.shape[0]
            if self.blacklight_count==0:
                self.blacklight_query_to_detect+=1
        if self.targeted == True:
            if pred_label == tlabel:
                is_adv = True
        else:
            if pred_label != olabel:
                is_adv = True
        return is_adv

    def binary_search(self, x, index, adv_value, non_adv_value,shape,olabel,tlabel):
        nquery = 0
        for i in range(10):
            next_value = (adv_value + non_adv_value) / 2
            x[index] = next_value
            nquery += 1
            is_adversarial = self.check_adv_status(x.reshape(shape),olabel,tlabel)
            if is_adversarial:
                adv_value = next_value
            else:
                non_adv_value = next_value
        return adv_value,nquery


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

        if not self.targeted and model(x).argmax(dim=1) != y:
            return x
        if self.blacklight:
            self.tracker = get_tracker(x, window_size= 20, hash_kept= 50, roundto= 50, step_size= 1, workers= 5)
            self.blacklight_count=0
            self.blacklight_query_to_detect=0
            self.blacklight_count_total=0
            self.blacklight_total_sample += 1
        oimg=x
        olabel=y
        seed=0
        nquery = 0
        self.model=model
        if self.targeted:
            raise NotImplementedError
        else:
            init_mode = 'salt_pepper_att'  # 'gauss_rand' #'salt_pepper'
            timg, nqry, _ = gen_starting_point(model, oimg, olabel, seed, self.dataset, init_mode)
            tlabel = None
            nquery += nqry
        adv,nquery, distances=self.pw_perturb(oimg.cpu().numpy(),timg.cpu().numpy(),olabel,tlabel)
        self.query_list.append(nquery)
        adv=torch.from_numpy(adv.reshape(oimg.shape)).to(self.device)
        self.distances.append(torch.linalg.norm((adv-x).reshape(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]),ord=2).cpu().data)

        if self.blacklight:
            if self.blacklight_count>0:
                self.blacklight_detection+=1
                self.blacklight_cover_list.append(self.blacklight_count/self.blacklight_count_total)
                self.blacklight_query_to_detect_list.append(self.blacklight_query_to_detect)
            print("blacklight detection rate: {}, cover: {}, query to detect: {} ".format(self.blacklight_detection/self.blacklight_total_sample if self.blacklight_total_sample>0 else "None",np.mean(self.blacklight_cover_list),np.mean(self.blacklight_query_to_detect_list)))
        print(self.result())
        return adv+self.RAND_noise
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
    def pw_perturb(self,oimg,timg,olabel,tlabel):

        max_query=self.query
        shape = oimg.shape
        N = oimg.size
        start_qry = 0
        end_qry = 0

        # flatten an image
        original = oimg.copy().reshape(-1)
        x = timg.copy().reshape(-1)
        nquery = 0
        D = np.zeros(max_query+500).astype(int)
        d = l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))

        terminate = False
        #while True:
        while not terminate:
            # draw random shuffling of all indices
            indices = list(range(N))
            random.shuffle(indices)

            for index in indices:

                # change index
                old_value = x[index]
                new_value = original[index]
                if old_value == new_value:
                    continue
                x[index] = new_value

                nquery += 1
                is_adversarial = self.check_adv_status(x.reshape(shape),olabel,tlabel)

                # if adversarial, restart from there
                if is_adversarial:
                    distance = np.linalg.norm(original - x)
                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry]=d
                    d = l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
                    if nquery%200 == 0:
                        print('nqry = {}; Reset value to original -> new distance: {}; L0 = {}; pred label: {}' .format(nquery,distance,d,self.model(torch.from_numpy(x.reshape(shape)).to(self.device)).argmax(dim=1)))

                else:
                    x[index] = old_value

                if nquery>max_query:
                    terminate = True
                    break
            else:
                # no index was succesful
                terminate = True

        if nquery>max_query:
            terminate = True
        else:
            terminate = False

        while not terminate:
            # draw random shuffling of all indices
            indices = list(range(N))
            random.shuffle(indices)

            # whether that run through all values made any improvement
            improved = False

            for index in indices:
                # change index
                old_value = x[index]
                original_value = original[index]
                if old_value == original_value:
                    continue
                x[index] = original_value

                # check if still adversarial
                nquery += 1
                is_adversarial = self.check_adv_status(x.reshape(shape),olabel,tlabel)

                # if adversarial, no binary search needed
                if is_adversarial:  # pragma: no cover
                    distance = np.linalg.norm(original - x)

                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry]=d
                    d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))

                    improved = True
                else:
                    adv_value = old_value              # x
                    non_adv_value = original_value     # ori
                    best_adv_value,nqry = self.binary_search(x, index, adv_value, non_adv_value,shape,olabel,tlabel)
                    nquery += nqry

                    if old_value != best_adv_value:
                        x[index] = best_adv_value
                        improved = True

                        distance = np.linalg.norm(original - x)

                        start_qry = end_qry
                        end_qry = nquery
                        D[start_qry:end_qry]=d
                        d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
                        print('nquery = {}; Set value at {} from {} to {}'
                                         ' (original has {}) ->'
                                         ' new distance: {}; \npred label:{}; L0:{}'.format(nquery,
                                             index, old_value, best_adv_value,
                                             original_value, distance,
                                            self.model(torch.from_numpy(x.reshape(shape)).to(self.device)).argmax(dim=1) ,d))
                    else:
                        x[index]=old_value

                if nquery > max_query:
                    terminate = True
                    break
            if not improved:
                # no improvement for any of the indices
                terminate = True
                #break

        d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
        D[end_qry:nquery]=d

        return x,nquery, D[:nquery]

    def pw_perturb_multiple(self,oimg,timg,olabel,tlabel,npix=196,max_query=1000):

        shape = oimg.shape
        N = oimg.size
        start_qry = 0
        end_qry = 0

        # flatten an image
        original = oimg.copy().reshape(-1)
        x = timg.copy().reshape(-1)
        nquery = 0
        D = np.zeros(max_query+500).astype(int)
        d = l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
        ngroup = N//npix

        terminate = False
        #while True:
        while not terminate:
            # draw random shuffling of all indices
            indices = list(range(N))
            random.shuffle(indices)

            for index in range(ngroup):
                # change multiple pixels (a group)
                idx = indices[index*npix:(index+1)*npix]
                old_value = x[idx]
                new_value = original[idx]
                tmp = np.abs(old_value - new_value)
                if tmp.sum()==0:
                    continue
                x[idx] = new_value

                nquery += 1
                is_adversarial = self.check_adv_status(x.reshape(shape),olabel,tlabel)

                # if adversarial, restart from there
                if is_adversarial:
                    distance = np.linalg.norm(original - x)
                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry]=d
                    d = l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
                    if nquery%200 == 0:
                        print('nqry = {}; Reset value to original -> new distance: {}; L0 = {}; pred label: {}' .format(nquery,distance,d,self.model(torch.from_numpy(x.reshape(shape)).to(self.device)).argmax(dim=1)))

                else:
                    x[idx] = old_value

                if nquery>max_query:
                    terminate = True
                    break
            else:
                # no index (group) was succesful
                terminate = True

        if nquery>max_query:
            terminate = True
        else:
            terminate = False

        print('refine stage!')
        while not terminate:
            # draw random shuffling of all indices
            indices = list(range(N))
            random.shuffle(indices)

            # whether that run through all values made any improvement
            improved = False

            for index in range(ngroup):
                # change multiple pixels (a group)
                idx = indices[index*npix:(index+1)*npix]
                old_value = x[idx]
                original_value = original[idx]
                tmp = np.abs(old_value - original_value)
                if tmp.sum()==0:
                    continue
                x[idx] = original_value

                # check if still adversarial
                nquery += 1
                is_adversarial = self.check_adv_status(x.reshape(shape),olabel,tlabel)

                # if adversarial, no binary search needed
                if is_adversarial:  # pragma: no cover
                    distance = np.linalg.norm(original - x)

                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry]=d
                    d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))

                    improved = True
                else:
                    adv_value = old_value              # x
                    non_adv_value = original_value     # ori
                    #best_adv_value,nqry = self.binary_search(x, index, adv_value, non_adv_value,shape,olabel,tlabel)
                    best_adv_value,nqry = self.binary_search(x, idx, adv_value, non_adv_value,shape,olabel,tlabel)
                    nquery += nqry
                    tmp2 = old_value - best_adv_value

                    #if old_value != best_adv_value:
                    if tmp2.sum() != 0:
                        x[idx] = best_adv_value
                        improved = True

                        distance = np.linalg.norm(original - x)

                        start_qry = end_qry
                        end_qry = nquery
                        D[start_qry:end_qry]=d
                        d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
                        if nquery%200 == 0:
                            print('nquery = {}; Set value at {} from {} to {}'
                                         ' (original has {}) ->'
                                         ' new distance: {}; \npred label:{}; L0:{}'.format(nquery,
                                             index, old_value, best_adv_value,
                                             original_value, distance,
                                            self.model(torch.from_numpy(x.reshape(shape)).to(self.device)).argmax(dim=1) ,d))
                    else:
                        x[idx]=old_value

                if nquery > max_query:
                    terminate = True
                    break
            if not improved:
                # no improvement for any of the indices
                terminate = True
                #break

        d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
        D[end_qry:nquery]=d

        return x,nquery, D[:nquery]

# ==============================================================================================
    def pw_perturb_multiple_scheduling(self,oimg,timg,olabel,tlabel,npix=196,max_query=1000):

        shape = oimg.shape
        N = oimg.size
        start_qry = 0
        end_qry = 0

        # flatten an image
        original = oimg.copy().reshape(-1)
        x = timg.copy().reshape(-1)
        nquery = 0
        D = np.zeros(max_query+500).astype(int)
        d = l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))

        terminate = False
        #while True:
        while not terminate:
            # draw random shuffling of all indices
            ngroup = N//npix
            indices = list(range(N))
            random.shuffle(indices)

            for index in range(ngroup):
                # change multiple pixels (a group)
                idx = indices[index*npix:(index+1)*npix]
                old_value = x[idx]
                new_value = original[idx]
                tmp = np.abs(old_value - new_value)
                if tmp.sum()==0:
                    continue
                x[idx] = new_value

                nquery += 1
                is_adversarial = self.check_adv_status(x.reshape(shape),olabel,tlabel)

                # if adversarial, restart from there
                if is_adversarial:
                    distance = np.linalg.norm(original - x)
                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry]=d
                    d = l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
                    if nquery%200 == 0:
                        print('nqry = {}; Reset value to original -> new distance: {}; L0 = {}; pred label: {}' .format(nquery,distance,d,self.model(torch.from_numpy(x.reshape(shape)).to(self.device)).argmax(dim=1)))

                else:
                    x[idx] = old_value

                if nquery>max_query:
                    terminate = True
                    break
            else:
                # no index (group) was succesful
                terminate = True

            if npix>=2:
                npix //= 2

        if nquery>max_query:
            terminate = True
        else:
            terminate = False

        print('refine stage!')
        while not terminate:
            # draw random shuffling of all indices
            indices = list(range(N))
            random.shuffle(indices)

            # whether that run through all values made any improvement
            improved = False

            for index in range(ngroup):
                # change multiple pixels (a group)
                idx = indices[index*npix:(index+1)*npix]
                old_value = x[idx]
                original_value = original[idx]
                tmp = np.abs(old_value - original_value)
                if tmp.sum()==0:
                    continue
                x[idx] = original_value

                # check if still adversarial
                nquery += 1
                is_adversarial = self.check_adv_status(x.reshape(shape),olabel,tlabel)

                # if adversarial, no binary search needed
                if is_adversarial:  # pragma: no cover
                    distance = np.linalg.norm(original - x)

                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry]=d
                    d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))

                    improved = True
                else:
                    adv_value = old_value              # x
                    non_adv_value = original_value     # ori
                    #best_adv_value,nqry = self.binary_search(x, index, adv_value, non_adv_value,shape,olabel,tlabel)
                    best_adv_value,nqry = self.binary_search(x, idx, adv_value, non_adv_value,shape,olabel,tlabel)
                    nquery += nqry
                    tmp2 = old_value - best_adv_value

                    #if old_value != best_adv_value:
                    if tmp2.sum() != 0:
                        x[idx] = best_adv_value
                        improved = True

                        distance = np.linalg.norm(original - x)

                        start_qry = end_qry
                        end_qry = nquery
                        D[start_qry:end_qry]=d
                        d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
                        if nquery%200 == 0:
                            print('nquery = {}; Set value at {} from {} to {}'
                                         ' (original has {}) ->'
                                         ' new distance: {}; \npred label:{}; L0:{}'.format(nquery,
                                             index, old_value, best_adv_value,
                                             original_value, distance,
                                            self.model(torch.from_numpy(x.reshape(shape)).to(self.device)).argmax(dim=1) ,d))
                    else:
                        x[idx]=old_value

                if nquery > max_query:
                    terminate = True
                    break
            if not improved:
                # no improvement for any of the indices
                terminate = True
                #break

        d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
        D[end_qry:nquery]=d

        return x,nquery, D[:nquery]

# ==============================================================================================

    def masking(self,idx,w):
        print(w)
        c1 = idx
        c2 = idx +   w*w
        c3 = idx + 2*w*w
        out = c1#([c1,c2,c3])
        return out

    def pw_perturb_multiple_px(self,oimg,timg,olabel,tlabel,npix=196,max_query=1000):

        shape = oimg.shape
        N = oimg.size
        start_qry = 0
        end_qry = 0

        # flatten an image
        original = oimg.copy().reshape(-1)
        x = timg.copy().reshape(-1)
        nquery = 0
        D = np.zeros(max_query+500).astype(int)
        d = l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))

        #npix = 48#196
        ngroup = N//npix

        terminate = False
        #while True:
        while not terminate:
            # draw random shuffling of all indices
            indices = list(range(N))
            random.shuffle(indices)

            for index in range(ngroup):
                # change multiple pixels (a group)
                idx = masking(indices[index*npix:(index+1)*npix],shape[1]) # from n selected dimensions to 3n dimensions (n selected pixels)
                old_value = x[idx]
                new_value = original[idx]
                tmp = old_value - new_value
                if tmp.sum()==0:
                    continue
                x[index*npix:(index+1)*npix] = new_value

                nquery += 1
                is_adversarial = self.check_adv_status(x.reshape(shape),olabel,tlabel)

                # if adversarial, restart from there
                if is_adversarial:
                    distance = np.linalg.norm(original - x)
                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry]=d
                    d = l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
                    if nquery%100 == 0:
                        print('nqry = {}; Reset value to original -> new distance: {}; L0 = {}; pred label: {}' .format(nquery,distance,d,self.model(torch.from_numpy(x.reshape(shape)).to(self.device)).argmax(dim=1)))

                else:
                    x[index*npix:(index+1)*npix] = old_value

                if nquery>max_query:
                    terminate = True
                    break
            else:
                # no index (group) was succesful
                terminate = True

        if nquery>max_query:
            terminate = True
        else:
            terminate = False

        while not terminate:
            # draw random shuffling of all indices
            indices = list(range(N))
            random.shuffle(indices)

            # whether that run through all values made any improvement
            improved = False

            for index in range(ngroup):
                # change multiple pixels (a group)
                idx = masking(indices[index*npix:(index+1)*npix],shape[1])
                old_value = x[idx]
                original_value = original[idx]
                tmp = old_value - original_value
                if tmp.sum()==0:
                    continue
                x[index*npix:(index+1)*npix] = original_value

                # check if still adversarial
                nquery += 1
                is_adversarial = self.check_adv_status(x.reshape(shape),olabel,tlabel)

                # if adversarial, no binary search needed
                if is_adversarial:  # pragma: no cover
                    distance = np.linalg.norm(original - x)

                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry]=d
                    d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))

                    improved = True
                else:
                    adv_value = old_value              # x
                    non_adv_value = original_value     # ori
                    best_adv_value,nqry = self.binary_search(x, index, adv_value, non_adv_value,shape,olabel,tlabel)
                    nquery += nqry

                    if old_value != best_adv_value:
                        x[index] = best_adv_value
                        improved = True

                        distance = np.linalg.norm(original - x)

                        start_qry = end_qry
                        end_qry = nquery
                        D[start_qry:end_qry]=d
                        d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
                        print('nquery = {}; Set value at {} from {} to {}'
                                         ' (original has {}) ->'
                                         ' new distance: {}; \npred label:{}; L0:{}'.format(nquery,
                                             index, old_value, best_adv_value,
                                             original_value, distance,
                                            self.model(torch.from_numpy(x.reshape(shape)).to(self.device)).argmax(dim=1) ,d))
                    else:
                        x[index*npix:(index+1)*npix]=old_value

                if nquery > max_query:
                    terminate = True
                    break
            if not improved:
                # no improvement for any of the indices
                terminate = True
                #break

        d =l0(torch.from_numpy(oimg),torch.from_numpy(x.reshape(shape)))
        D[end_qry:nquery]=d

        return x,nquery, D[:nquery]