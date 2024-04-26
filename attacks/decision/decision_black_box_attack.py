from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''

This file is copied from the following source:
link: https://github.com/ash-aldujaili/blackbox-adv-examples-signhunter/blob/master/src/attacks/blackbox/black_box_attack.py

@inproceedings{
al-dujaili2020sign,
title={Sign Bits Are All You Need for Black-Box Attacks},
author={Abdullah Al-Dujaili and Una-May O'Reilly},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SygW0TEFwH}
}

The original license is placed at the end of this file.

basic structure for main:
    1. config args and prior setup
    2. define funtion that returns a summary of the attack results
    3. set the decision-based attacking
    4. return the logs
    
'''

"""
Implements the base class for decision-based black-box attacks
"""
import numpy as np
import torch
from torch import Tensor as t
from attacks.certified_attack.probabilistic_fingerprint import *

import sys

def get_tracker(query, window_size, hash_kept, roundto, step_size, workers):
    tracker = InputTracker(query, window_size, hash_kept, round=roundto, step_size=step_size, workers=workers)
    LOGGER.info("Blacklight detector created.")
    return tracker
class DecisionBlackBoxAttack(object):
    def __init__(self, max_queries=np.inf, epsilon=0.5, p='inf', lb=0., ub=1., batch_size=1,target=False,target_type="median",device="cuda",blacklight=True,sigma=0,post_sigma=0):
        """
        :param max_queries: max number of calls to model per data point
        :param epsilon: perturbation limit according to lp-ball
        :param p: norm for the lp-ball constraint
        :param lb: minimum value data point can take in any coordinate
        :param ub: maximum value data point can take in any coordinate
        """
        assert p in ['inf', '2'], "L-{} is not supported".format(p)

        self.p = p
        self.max_queries = max_queries
        self.total_queries = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_distance = 0
        self.sigma = sigma
        self.post_sigma=post_sigma
        self.post_noise=0
        self.EOT = 1
        self.lb = lb
        self.ub = ub
        self.epsilon = epsilon/ 255
        self.batch_size = batch_size
        self.list_loss_queries = torch.zeros(1, self.batch_size)
        self.target=target
        self.targeted = target
        self.target_type=target_type
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


    def result(self):
        """
        returns a summary of the attack results (to be tabulated)
        :return:
        """
        list_loss_queries = self.list_loss_queries[1:].view(-1)
        mask = list_loss_queries > 0
        list_loss_queries = list_loss_queries[mask]
        self.total_queries = int(self.total_queries)
        self.total_successes = int(self.total_successes)
        self.total_failures = int(self.total_failures)
        return {
            "total_queries": self.total_queries,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "average_num_queries": "NaN" if self.total_successes == 0 else self.total_queries / self.total_successes,
            "failure_rate": "NaN" if self.total_successes + self.total_failures == 0 else self.total_failures / (self.total_successes + self.total_failures),
            "median_num_loss_queries": "NaN" if self.total_successes == 0 else torch.median(list_loss_queries).item(), 
            "config": self._config(),
            "blacklight_detection_rate":self.blacklight_detection/self.blacklight_total_sample if self.blacklight_total_sample>0 else None,
            "blacklight_coverage":np.mean(self.blacklight_cover_list),
            "blacklight_query_to_detect":np.mean(self.blacklight_query_to_detect_list),
            "distance": np.mean(self.distances)

        }

    def _config(self):
        """
        return the attack's parameter configurations as a dict
        :return:
        """
        raise NotImplementedError

    def distance(self, x_adv, x = None):
        if x is None:
            diff = x_adv.reshape(x_adv.size()[0], -1)
        else:
            diff = (x_adv - x).reshape(x.size(0), -1)
        if self.p == '2':
            out = torch.sqrt(torch.sum(diff * diff)).item()
        elif self.p == 'inf':
            out = torch.sum(torch.max(torch.abs(diff), 1)[0]).item()
        return out
    
    def is_adversarial(self, x, y):
        '''
        check whether the adversarial constrain holds for x
        '''
        if self.targeted:
            return self.predict_label(x) == y
        else:
            return self.predict_label(x) != y

    def predict_label(self, xs):
        if type(xs) is torch.Tensor:
            x_eval = xs.permute(0,3,1,2)
        else:
            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2)).to(self.device)
        self.RAND_noise=self.sigma * torch.randn_like(x_eval)
        x_eval=x_eval+self.RAND_noise
        x_eval = torch.clamp(x_eval, 0, 1)
        if self.blacklight:
            self.blacklight_detect(x_eval)
            self.blacklight_count_total+=x_eval.shape[0]
            if self.blacklight_count==0:
                self.blacklight_query_to_detect+=1

        if self.ub == 255:
            out = self.model(x_eval)
        else:
            out = self.model(x_eval)
        self.post_noise=self.post_sigma*torch.randn_like(out)
        l = (out+self.post_noise).argmax(dim=1)
        return l.detach()

    def _perturb(self, xs_t, ys):
        raise NotImplementedError

    def get_label(self,target_type):
        logit = self.model(torch.FloatTensor(self.x_batch.transpose(0, 3, 1, 2) / 255.).to(self.device))
        if target_type == 'random':
            label = torch.randint(low=0, high=logit.shape[1], size=self.y_batch.shape).long().to(self.device)
        elif target_type == 'least_likely':
            label = logit.argmin(dim=1)
        elif target_type == 'most_likely':
            label = torch.argsort(logit, dim=1, descending=True)[:, 1]
        elif target_type == 'median':
            label = torch.argsort(logit, dim=1, descending=True)[:, 4]
        elif 'label' in target_type:
            label = torch.ones_like(self.y_batch) * int(target_type[5:])
        return label.detach()
    def run(self, xs, ys, model, dset=None):



        self.model=model
        xs = xs.permute(0, 2, 3, 1).cpu().numpy()*255.0
        y_batch = ys.cpu().numpy()
        self.x_batch=xs

        if self.target:
            ys_t = self.get_label(self.target_type)
        else:
            ys_t = torch.LongTensor(y_batch).to(self.device)

        self.y_batch=ys_t

        self.model = model
        self.train_dataset = dset

        self.logs = {
            'iteration': [0],
            'query_count': [0]
        }

        xs = xs / 255.0
        xs_t = t(xs).to(self.device)

        if self.blacklight:
            self.tracker = get_tracker(xs_t.permute(0,3,1,2), window_size= 20, hash_kept= 50, roundto= 50, step_size= 1, workers= 5)
            self.blacklight_count=0
            self.blacklight_query_to_detect=0
            self.blacklight_count_total=0


        # initialize
        if self.targeted:
            check = self.is_adversarial(xs_t, ys_t)
            if torch.any(check):
                print('Some original images already belong to the target class!')
                return xs_t.permute(0, 3, 1, 2)
        else:
            check = self.is_adversarial(xs_t, ys_t)
            if torch.any(check):
                print('Some original images do not belong to the original class!')
                return xs_t.permute(0, 3, 1, 2)

        if self.blacklight:
            self.blacklight_total_sample += 1

        adv, q = self._perturb(xs_t, ys_t)

        if self.blacklight:
            if self.blacklight_count>0:
                self.blacklight_detection+=1
                self.blacklight_cover_list.append(self.blacklight_count/self.blacklight_count_total)
                self.blacklight_query_to_detect_list.append(self.blacklight_query_to_detect)
            print("blacklight detection rate: {}, cover: {}, query to detect: {} ".format(self.blacklight_detection/self.blacklight_total_sample,np.mean(self.blacklight_cover_list),np.mean(self.blacklight_query_to_detect_list)))

        success = self.distance(adv,xs_t) < self.epsilon
        self.total_queries += np.sum(q * success)
        self.total_successes += np.sum(success)
        self.total_failures += ys_t.shape[0] - success
        self.list_loss_queries = torch.cat([self.list_loss_queries, torch.zeros(1, self.batch_size)], dim=0)
        if type(q) is np.ndarray:
            self.list_loss_queries[-1] = t(q * success)
        else:
            self.list_loss_queries[-1] = q * success
        # self.total_distance += self.distance(adv,xs_t)

        x_adv=adv if success else xs_t

        self.distances.append(torch.linalg.norm((x_adv-xs_t).reshape(x_adv.shape[0]*x_adv.shape[1]*x_adv.shape[2]*x_adv.shape[3]),ord=2).cpu().data)


        return x_adv.permute(0, 3, 1, 2)+self.RAND_noise

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
'''
    
MIT License
Copyright (c) 2019 Abdullah Al-Dujaili
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''
