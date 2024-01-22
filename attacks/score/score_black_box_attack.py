from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
This file is copied from the following source:
link: https://github.com/ash-aldujaili/blackbox-adv-examples-signhunter/blob/master/src/attacks/blackbox/black_box_attack.py

The original license is placed at the end of this file.

basic structure for main:
    1. config args and prior setup
    2. define funtion that returns a summary of the attack results
    3. set the score-based attacking
    4. return the logs
'''

"""
Implements the base class for black-box attacks
"""

import numpy as np
import torch 
from torch import Tensor as t

from attacks.utils.compute import l2_proj_maker, linf_proj_maker
import sys
class ScoreBlackBoxAttack(object):
    def __init__(self, max_loss_queries=np.inf,
                 max_extra_queries=np.inf,
                 epsilon=0.5, p='inf', lb=0., ub=255.,batch_size = 50, name = "nes",target=False,target_type="median",device='cuda'):
        """
        :param max_loss_queries: max number of calls to model per data point
        :param max_extra_queries: max number of calls to early stopping extraerion per data point
        :param epsilon: perturbation limit according to lp-ball
        :param p: norm for the lp-ball constraint
        :param lb: minimum value data point can take in any coordinate
        :param ub: maximum value data point can take in any coordinate
        """
        assert p in ['inf', '2'], "L-{} is not supported".format(p)

        self.epsilon = epsilon
        self.p = p
        self.batch_size = batch_size
        self.max_loss_queries = max_loss_queries
        self.max_extra_queries = max_extra_queries
        self.list_loss_queries = torch.zeros(1, self.batch_size)
        self.total_loss_queries = 0
        self.total_extra_queries = 0
        self.total_successes = 0
        self.total_failures = 0
        self.lb = lb
        self.ub = ub
        self.name = name
        # the _proj method takes pts and project them into the constraint set:
        # which are
        #  1. epsilon lp-ball around xs
        #  2. valid data pt range [lb, ub]
        # it is meant to be used within `self.run` and `self._perturb`
        self._proj = None
        # a handy flag for _perturb method to denote whether the provided xs is a
        # new batch (i.e. the first iteration within `self.run`)
        self.is_new_batch = False
        self.target=target
        self.target_type=target_type
        self.device=torch.device(device)

    def result(self):
        """
        returns a summary of the attack results (to be tabulated)
        :return:
        """
        list_loss_queries = self.list_loss_queries[1:].view(-1)
        mask = list_loss_queries > 0
        list_loss_queries = list_loss_queries[mask]
        self.total_loss_queries = int(self.total_loss_queries)
        self.total_extra_queries = int(self.total_extra_queries)
        self.total_successes = int(self.total_successes)
        self.total_failures = int(self.total_failures)
        return {
            "total_loss_queries": self.total_loss_queries,
            "total_extra_queries": self.total_extra_queries,
            "average_num_loss_queries": "NaN" if self.total_successes == 0 else self.total_loss_queries / self.total_successes,
            "average_num_extra_queries": "NaN" if self.total_successes == 0 else self.total_extra_queries / self.total_successes,
            "median_num_loss_queries": "NaN" if self.total_successes == 0 else torch.median(list_loss_queries).item(), 
            "total_queries": self.total_extra_queries + self.total_loss_queries,
            "average_num_queries": "NaN" if self.total_successes == 0 else (self.total_extra_queries + self.total_loss_queries) / self.total_successes,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "failure_rate": "NaN" if self.total_successes + self.total_failures == 0 else self.total_failures / (self.total_successes + self.total_failures),
            "config": self._config()
        }

    def _config(self):
        """
        return the attack's parameter configurations as a dict
        :return:
        """
        raise NotImplementedError

    def _perturb(self, xs_t, loss_fct):
        """
        :param xs_t: batch_size x dim x .. (torch tensor)
        :param loss_fct: function to query (the attacker would like to maximize) (batch_size data pts -> R^{batch_size}
        :return: suggested xs as a (torch tensor)and the used number of queries per data point
            i.e. a tuple of (batch_size x dim x .. tensor, batch_size array of number queries used)
        """
        raise NotImplementedError

    def proj_replace(self, xs_t, sugg_xs_t, dones_mask_t):
        sugg_xs_t = self._proj(sugg_xs_t)
        # replace xs only if not done
        xs_t = sugg_xs_t * (1. - dones_mask_t) + xs_t * dones_mask_t
        return xs_t


    def cw_loss(self,logit, label, target=False):
        if target:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(label)
            second_max_index = target_is_max.long() * argsort[:, 1] + (~ target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return target_logit - second_max_logit
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logit.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label)
            second_max_index = gt_is_max.long() * argsort[:, 1] + (~gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return second_max_logit - gt_logit

    def xent_loss(self,logit, label, target=False):
        if not target:
            return torch.nn.CrossEntropyLoss(reduction='none')(logit, label)
        else:
            return -torch.nn.CrossEntropyLoss(reduction='none')(logit, label)

    def loss_fct(self, xs, es=False):
        if type(xs) is torch.Tensor:
            x_eval = (xs.permute(0, 3, 1, 2) / 255.).to(self.device)
        else:
            x_eval = (torch.FloatTensor(xs.transpose(0, 3, 1, 2)) / 255.).to(self.device)

        if self.p== 'inf':
            x_eval = torch.clamp(x_eval - self.x_ori, - self.epsilon/255, self.epsilon/255) + self.x_ori
        else:
            # proj_2 = l2_proj_maker(x_ori, epsilon)
            x_eval = self.proj_2(x_eval)
        x_eval = torch.clamp(x_eval, 0, 1)
        # ---------------------#
        # sigma = config["sigma"]
        sigma = 0
        # ---------------------#
        x_eval = x_eval + sigma * torch.randn_like(x_eval)
        x_eval = torch.clamp(x_eval, 0, 1)
        # _, y_logit = model(x_eval.to(self.device))
        y_logit = self.model(x_eval.to(self.device))
        loss = self.cw_loss(y_logit, self.y_batch, self.target)
        if es:
            y_logit = y_logit.detach()
            correct = torch.argmax(y_logit, axis=1) == self.y_batch
            if self.target:
                return correct, loss.detach()
            else:
                return ~correct, loss.detach()
        else:
            return loss.detach()

    def early_stop_crit_fct(self,xs):

        if type(xs) is torch.Tensor:
            x_eval = xs.permute(0, 3, 1, 2) / 255.
        else:
            x_eval = torch.FloatTensor(xs.transpose(0, 3, 1, 2)) / 255.
        x_eval = torch.clamp(x_eval, 0, 1)
        # ---------------------#
        # sigma = config["sigma"]
        sigma = 0
        # ---------------------#
        x_eval = x_eval + sigma * torch.randn_like(x_eval)
        x_eval = torch.clamp(x_eval, 0, 1)
        # _, y_logit = model(x_eval.to(self.device))
        y_logit = self.model(x_eval.to(self.device))
        y_logit = y_logit.detach()
        correct = torch.argmax(y_logit, axis=1) == self.y_batch

        if self.target:
            return correct
        else:
            return ~correct

    def get_label(self,target_type):
        _, logit = self.model(torch.FloatTensor(self.x_batch.transpose(0, 3, 1, 2) / 255.).to(self.device))
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

    def run(self, xs, ys, model):
        """
        attack with `xs` as data points using the oracle `l` and
        the early stopping extraerion `early_stop_extra_fct`
        :param xs: data points to be perturbed adversarially (numpy array)
        :param loss_fct: loss function (m data pts -> R^m)
        :param early_stop_extra_fct: early stop function (m data pts -> {0,1}^m)
                ith entry is 1 if the ith data point is misclassified
        :return: a dict of logs whose length is the number of iterations
        """
        self.model=model
        x_batch = xs.permute(0, 2, 3, 1).cpu().numpy()*255.0
        y_batch = ys.cpu().numpy()

        if self.target:
            self.y_batch = self.get_label(self.target_type)
        else:
            self.y_batch = torch.LongTensor(y_batch).to(self.device)


        self.x_ori = torch.FloatTensor(x_batch.copy().transpose(0, 3, 1, 2) / 255.).to(self.device)

        if self.p == 'inf':
            pass
        else:
            self.proj_2 = l2_proj_maker(self.x_ori, self.epsilon/255)
        # convert to tensor
        xs_t = t(x_batch).to(self.device)
        
        batch_size = x_batch.shape[0]
        num_axes = len(x_batch.shape[1:])
        num_loss_queries = torch.zeros(batch_size).to(self.device)
        num_extra_queries = torch.zeros(batch_size).to(self.device)

        dones_mask = self.early_stop_crit_fct(xs_t)
        correct_classified_mask = ~dones_mask

        # list of logs to be returned
        logs_dict = {
            'total_loss': [],
            'total_successes': [],
            'total_failures': [],
            'iteration': [],
            'total_loss_queries': [],
            'total_extra_queries': [],
            'total_queries': [],
            'num_loss_queries_per_iteration': [],
            'num_extra_queries_per_iteration': []
        }

        # init losses for performance tracking
        losses = torch.zeros(batch_size).to(self.device)

        # make a projector into xs lp-ball and within valid pixel range
        if self.p == '2':
            _proj = l2_proj_maker(xs_t, self.epsilon)
            self._proj = lambda _: torch.clamp(_proj(_), self.lb, self.ub)
        elif self.p == 'inf':
            _proj = linf_proj_maker(xs_t, self.epsilon)
            self._proj = lambda _: torch.clamp(_proj(_), self.lb, self.ub)
        else:
            raise Exception('Undefined l-p!')

        # iterate till model evasion or budget exhaustion
        self.is_new_batch = True
        its = 0
        while True:
            # if np.any(num_loss_queries + num_extra_queries >= self.max_loss_queries):
            if torch.any(num_loss_queries >= self.max_loss_queries):
                print("#loss queries exceeded budget, exiting")
                break
            if torch.any(num_extra_queries >= self.max_extra_queries):
                print("#extra_queries exceeded budget, exiting")
                break
            if torch.all(dones_mask):
                print("all data pts are misclassified, exiting")
                break
            # propose new perturbations
            sugg_xs_t, num_loss_queries_per_step = self._perturb(xs_t, self.loss_fct)
            # project around xs and within pixel range and
            # replace xs only if not done
            ##updated x here
            xs_t = self.proj_replace(xs_t, sugg_xs_t, (dones_mask.reshape(-1, *[1] * num_axes).float()))

            # update number of queries (note this is done before updating dones_mask)
            num_loss_queries += num_loss_queries_per_step.to(self.device) * (~dones_mask)
            num_extra_queries += (~dones_mask)
            losses = self.loss_fct(xs_t) * (~dones_mask) + losses * dones_mask

            # update dones mask
            dones_mask = dones_mask | self.early_stop_crit_fct(xs_t)
            success_mask = dones_mask * correct_classified_mask
            its += 1

            self.is_new_batch = False


            if its % 1 == 0:
                success_mask = dones_mask * correct_classified_mask
                total_successes = float(success_mask.sum())
                print ("Iteration : ", its, 'ave_loss_queries : ', ((num_loss_queries * success_mask).sum() / total_successes).item(),\
                    "ave_extra_queries : ", ((num_extra_queries * success_mask).sum()  / total_successes).item(), \
                    "ave_queries : ", ((num_loss_queries * success_mask).sum() / total_successes + (num_extra_queries * success_mask).sum()  / total_successes).item(), \
                    "successes : ", success_mask.sum().item() / float(success_mask.shape[0]), \
                    "failures : ", ((~dones_mask) * correct_classified_mask).sum().item()  / float(success_mask.shape[0]))
                sys.stdout.flush()
            


        success_mask = dones_mask * correct_classified_mask
        self.total_loss_queries += (num_loss_queries * success_mask).sum()
        self.total_extra_queries += (num_extra_queries * success_mask).sum()

        current_batch_size = self.list_loss_queries.size(1)  # Assuming size 1 is the dimension that must match
        if batch_size != current_batch_size:
            # If batch_size is different, adjust the size of the zeros tensor
            additional_zeros = torch.zeros(1, current_batch_size)
        else:
            additional_zeros = torch.zeros(1, batch_size)

        self.list_loss_queries = torch.cat([self.list_loss_queries, additional_zeros], dim=0)
        self.list_loss_queries[-1][:batch_size] = num_loss_queries * success_mask

        self.total_successes += success_mask.sum()
        self.total_failures += ((~dones_mask) * correct_classified_mask).sum()

        # set self._proj to None to ensure it is intended use
        self._proj = None

        return xs_t.permute(0, 3, 1, 2) / 255
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
