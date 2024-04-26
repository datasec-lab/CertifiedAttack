'''
- rev 0.1.0:
- rev 0.2.0:  
- rev 0.3.0: change PretrainedModel class to work with pretrained model on 
             unnorm dataset and path to user defined path for pretrained models
- rev 0.4.0: add new functions to export data using pandas

'''
import torch
from torchvision import datasets, transforms, models
import torch.utils.data as data
import numpy as np
import random
import pandas as pd
import torch.backends.cudnn as cudnn

# ======================== Dataset ========================

def load_data(dataset, data_path=None, batch_size=1):

    if dataset == 'imagenet' and data_path==None:
        data_path = '../datasets/ImageNet-val'
    elif dataset == 'cifar10'and data_path==None:
        data_path = '../datasets/cifar10' 

    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset == 'cifar100':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset == 'imagenet':
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])

        testset = datasets.ImageNet(root=data_path, split='val', download=False, transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader, testset


# ======================== Model ========================

class PretrainedModel():
    def __init__(self,model,dataset='imagenet',arch='vit',norm=False):
        self.model = model
        self.dataset = dataset
        self.arch = arch
        self.norm = norm
        
        # ======= non-normalized =========       
        if self.norm:
            self.mu = torch.Tensor([0., 0., 0.]).float().view(1, 3, 1, 1).cuda()
            self.sigma = torch.Tensor([1., 1., 1.]).float().view(1, 3, 1, 1).cuda()
        
        # ======= ViT =========
        elif self.arch == 'vit':
            self.mu = torch.Tensor([0.5, 0.5, 0.5]).float().view(1, 3, 1, 1).cuda()
            self.sigma = torch.Tensor([0.5, 0.5, 0.5]).float().view(1, 3, 1, 1).cuda()
        
        else:
            # ======= CIFAR10 ==========
            if self.dataset == 'cifar10':
                self.mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(1, 3, 1, 1).cuda()
                self.sigma = torch.Tensor([0.2023, 0.1994, 0.2010]).float().view(1, 3, 1, 1).cuda()
            
            # ======= CIFAR100 =========
            elif self.dataset == 'cifar100':
                self.mu = torch.Tensor([0.5071, 0.4865, 0.4409]).float().view(1, 3, 1, 1).cuda()
                self.sigma = torch.Tensor([0.2673, 0.2564, 0.2762]).float().view(1, 3, 1, 1).cuda()
            
            # ======= ImageNet =========
            elif self.dataset == 'imagenet':
                self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
                self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()              
            
            # ======= MNIST =========
            elif self.dataset == 'mnist':
                self.mu = torch.Tensor([0., 0., 0.]).float().view(1, 1, 1, 1).cuda()
                self.sigma = torch.Tensor([1., 1., 1.]).float().view(1, 1, 1, 1).cuda()          

    def predict(self, x):
        
        # x: unnorm input
        # shape: [n,c,w,h]
        # already .cuda()
        img = (x - self.mu) / self.sigma
        if self.arch =='vit':
            #img = x.clone()
            out = self.model(img)[0] #self.model(img) => logit, att_mat; self.model(out)[0] => only logit result
        else:
            #img = (x - self.mu) / self.sigma
            out = self.model(img)
        return  out

    def predict_label(self, x):
        img = (x - self.mu) / self.sigma
        if self.arch =='vit':
            out = self.model(img)[0]
        else:
            out = self.model(img)
        out = torch.max(out,1)
        return out[1]

    def __call__(self, x):
        return self.predict(x)

# ======================== Generate CIFAR10 subset ========================

def gen_rand_dataset(model,evalset, n_class, N_sam, n, targeted=True, seed=None):
    if seed != None:
        np.random.seed(seed)
        random.seed(seed)

    idxs = np.zeros((n_class,N_sam),dtype=int)
    count = np.zeros(n_class,dtype=int)
    cnt = 0
    
    # generate subset data
    for i, (x, y) in enumerate(evalset):
        x = torch.unsqueeze(x, 0).cuda()
        y_pred = model.predict_label(x)
        if (y_pred == y) and (count[y] < N_sam):
            cnt += 1
            idxs[y,count[y]] = i
            count[y] += 1
            if cnt >= N_sam*n_class:
                break
    # generate random subset for evaluation
    cnt = 0
    if targeted:
        # out: [ocla, oID, tcla, tID] => targeted attack
        #out = np.zeros((n_class*n*(n_class-1),4)).astype(int)
        out = torch.zeros((n_class*n*(n_class-1),4),dtype=int).cuda()
        for i in range(n_class):
            tmp = random.sample(idxs[i].tolist(),n)
            #tmp = np.random.choice(idxs[i], n, replace = False)
            for j in range(n):
                for k in range(n_class):
                    if i != k:
                        idx = np.random.choice(idxs[k], 1, replace = False)
                        out[cnt] = torch.tensor([i,tmp[j],k,idx[0]],dtype=torch.long)
                        cnt += 1
    else:
        # out: [ocla, oID] => untargeted attack
        #out = np.zeros((n_class*n,2)).astype(int)
        out = np.zeros((n_class*n,2),dtype=int).cuda()
        for i in range(n_class):
            tmp = random.sample(idxs[i].tolist(),n)
            for j in range(n):
                out[cnt] = [i,tmp[j]]
                cnt += 1
    return out

# ======================== Load pre-defined ImageNet subset ========================

def load_predefined_set(filename,targeted):
    df = pd.read_csv(filename, index_col=0)
    if targeted:
        # out: [ocla, oID, tcla, tID]
        np_df = df.to_numpy()
    else:
        # out: [ocla, oID]
        np_df = df.drop(['tcla', 'tID'], axis=1)
        np_df = np_df.drop_duplicates(subset=['ocla','o_ID'],keep='first')
        np_df = np_df.to_numpy()

    return np_df

def get_evalset(model,dataset,net,input_set,seed,targeted):
    if dataset == 'imagenet':
        if net == 'vit':
            subset_path = './evaluation_set/vit_dataset_200_final.csv'
        elif net == 'resnet50':
            subset_path = './evaluation_set/ResNet_dataset_200_final.csv'
        output = load_predefined_set(subset_path,targeted)
    elif dataset == 'cifar10' :
        evalset = input_set
        n_class = 10
        N_sam = 500
        n = 100
        output = gen_rand_dataset(model,evalset, n_class, N_sam, n, targeted,seed)
    return output

# ======================== Generate starting img ========================

def salt_pepper_noise(img,rndtype,scale =8,seed=None): #(normal/uniform distribution)
    
    if seed != None:
        torch.manual_seed(seed)
    
    # img = [n,c,w,h]
    c = img.shape[1]
    wi = img.shape[2]
    he = img.shape[3]
    img_size = []
    img_size.append(img.shape[0])
    img_size.append(c)
    img_size.append(wi//scale)
    img_size.append(he//scale)
    
    if rndtype == 'normal':
        rnd = torch.randn(wi,he).type(torch.FloatTensor)
        timg = torch.clamp((rnd>0*1).type(torch.FloatTensor).cuda(),0,1)
    else:
        rnd = torch.rand(wi,he).type(torch.FloatTensor)
        timg = torch.clamp((rnd>0.5*1).type(torch.FloatTensor).cuda(),0,1)
    
    timg = timg.repeat(c,1,1)
    timg = np.transpose(timg.cpu().numpy(),(1,2,0))
    oimg = np.zeros((wi,he,c))
    
    for i in range(wi//scale):
        for j in range(he//scale):
            oimg[i*scale:(i+1)*scale,j*scale:(j+1)*scale,:] = timg[i,j,:]
    
    oimg = np.transpose(oimg,(2,0,1))
    oimg = torch.from_numpy(oimg).unsqueeze(0).cuda()
    oimg = oimg.to(torch.float)
    return oimg

def rand_img_upscale(img,rndtype,scale=8,seed=None): #(normal/uniform distribution)
    
    if seed != None:
        torch.manual_seed(seed)
    
    # img = [n,c,w,h]
    c = img.shape[1]
    wi = img.shape[2]
    he = img.shape[3]
    img_size = []
    img_size.append(img.shape[0])
    img_size.append(c)
    img_size.append(wi//scale)
    img_size.append(he//scale)
    
    if rndtype == 'normal':
        rnd = torch.randn(img_size).type(torch.FloatTensor)
    else:
        rnd = torch.rand(img_size).type(torch.FloatTensor)

    timg = torch.clamp(rnd.cuda(),0,1)
    timg = timg[0].cpu().numpy()
    timg = np.transpose(timg,(1,2,0))
    
    oimg = np.zeros((wi,he,c))
    
    for i in range(wi//scale):
        for j in range(he//scale):
            oimg[i*scale:(i+1)*scale,j*scale:(j+1)*scale,:] = timg[i,j,:]
    
    oimg = np.transpose(oimg,(2,0,1))
    oimg = torch.from_numpy(oimg).unsqueeze(0).cuda()
    oimg = oimg.to(torch.float)
    return oimg

def check_adv_status(model,img,olabel,tlabel,flag=True):
    is_adv = False
    pred_label = model(torch.from_numpy(img).cuda()).argmax(dim=1)
    if flag == True:
        if pred_label == tlabel:
            is_adv = True
    else:
        if pred_label != olabel:
            is_adv = True
    return is_adv

def salt_pepper_att(model, oimg,olabel,repetitions=10,eps=0.1):
    
    min_ = 0
    max_ = 1
    flag = False
    tlabel = None
    # axis = a.channel_axis(batch=False)
    # index of channel & color. Eg: [1,3,32,32] -> channels = 1; [3,32,32] -> channels = 0; 
    # number of channel of color = 3 if RGB or 1 if gray scale
    start_qry = 0
    end_qry = 0
    nquery = 0
    #D = np.zeros(5000).astype(int)
    D = torch.zeros(1000,dtype=int).cuda()
    axis = 1
    channels = oimg.shape[axis] 
    
    shape = list(oimg.shape)
    shape[axis] = 1
    r = max_ - min_
    
    epsilons = 100
    pixels = np.prod(shape)      
    
    epsilons = min(epsilons, pixels)
    
    max_epsilon = 1
    distance = np.inf
    adv = oimg.copy()
    d = 0
    
    for i in range(repetitions):
        for epsilon in np.linspace(0, max_epsilon, num=epsilons + 10)[1:]:
            p = epsilon #probability
            u = np.random.uniform(size=shape)
            u = u.repeat(channels, axis=axis)
            
            salt = (u >= 1 - p / 2).astype(oimg.dtype) * r
            pepper = -(u < p / 2).astype(oimg.dtype) * r 
            saltpepper = np.clip(salt + pepper,-eps,eps)
            perturbed = oimg + saltpepper
            perturbed = np.clip(perturbed, min_, max_)
            
            temp_dist = l0(torch.from_numpy(oimg),torch.from_numpy(perturbed))
            
            if temp_dist >= distance:
                continue
            nquery += 1
            is_adversarial = check_adv_status(model,perturbed,olabel,tlabel,flag)
            if is_adversarial:
                # higher epsilon usually means larger perturbation, but
                # this relationship is not strictly monotonic, so we set
                # the new limit a bit higher than the best one so far
                distance = temp_dist
                adv = perturbed
                max_epsilon = epsilon * 1.2
                
                start_qry = end_qry
                end_qry = nquery
                D[start_qry:end_qry] = d
                d = l0(torch.from_numpy(oimg),torch.from_numpy(adv))
                
                break
        print('i: %d; nqry: %d, pred_label: %d; temp_dist: %2.3f; L0: %2.3f' %(i,nquery, model(torch.from_numpy(perturbed).cuda()).argmax(dim=1),temp_dist, distance))
    d = l0(torch.from_numpy(oimg),torch.from_numpy(adv))
    D[end_qry:nquery] = d

    return adv,nquery,D[:nquery]

def gen_starting_point(model,oimg,olabel,seed,dataset_name,init_mode):

    nqry = 0
    i = 0
    rndtype = 'normal'
    if dataset_name == 'imagenet':
        scale = np.array([1,2,4,8,16,32])
    else:
        scale = np.array([1,2,4,8,16])

    if init_mode == 'salt_pepper_att':
        eps = 1
        repetitions = 2#10
        out, nqry, D = salt_pepper_att(model,oimg.cpu().numpy(),olabel,repetitions,eps)
        out = torch.from_numpy(out.reshape(oimg.shape)).cuda()
    elif init_mode == 'salt_pepper':
        while True:
            out = salt_pepper_noise(oimg,rndtype,scale[i],seed)
            label = model.predict_label(out)
            nqry += 1
            if label!= olabel:
                #D = np.ones(nqry).astype(int) * l0(oimg,out)
                D = torch.ones(nqry,dtype=int).cuda() * l0(oimg,out)
                break
            elif i<len(scale):
                i += 1

    elif init_mode == 'gauss_rand':
        while True:
            out = rand_img_upscale(oimg,rndtype,scale[i],seed)
            label = model.predict_label(out)
            nqry += 1
            if label!= olabel:
                #D = np.ones(nqry).astype(int) * l0(oimg,out)
                D = torch.ones(nqry,dtype=int).cuda() * l0(oimg,out)
                break
            elif i<len(scale):
                i += 1

    return out, nqry, D

# ======================== Measurement ========================

def l0(img1,img2):
    # exactly
    xo = torch.abs(img1-img2)
    d = torch.zeros(xo.shape[2],xo.shape[3]).bool().cuda()
    for i in range (xo.shape[1]):
        tmp = (xo[0,i]>0.).bool().cuda()
        d = tmp | d # "or" => + ; |
    return d.sum().item()

def l0a(img1,img2):
    # roughly
    xo = torch.sum(img1,1)
    xt = torch.sum(img2,1)
    d = torch.abs(xo-xt)>0.0
    return d.sum().item()

def l0b(img1,img2):
    # exactly
    xo = torch.abs(img1-img2)
    d = torch.sum(xo,1)>0.0
    return d.sum().item()

# ========================= Export to csv ======================

def export_pd_csv(D,head,key_info,output_path,n_point=None,query_limit=None):

    if n_point is not None:
        #1.
        key = pd.DataFrame(key_info)
        step_size = int(query_limit/n_point)

        #2.
        data = np.zeros(n_point).astype(int)  
        for k in range(n_point):
            q = k*step_size
            if q<(len(D)):
                data[k]= D[q]
            else:
                data[k]= D[len(D)-1]

        #3.
        out = pd.concat([key.transpose(), pd.DataFrame(data).transpose()], axis=1).to_numpy()

    else:
        key = key_info.copy()
        key.append(str(D))
        out = np.array(key).reshape(1,-1)

    #4
    df = pd.DataFrame(out,columns = head)
    with open(output_path, mode = 'a') as f:
        df.to_csv(f, header=f.tell()==0,index = False)
