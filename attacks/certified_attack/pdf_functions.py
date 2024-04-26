import numpy as np
from scipy.stats import norm as Norm
from scipy.stats import beta
import scipy.special as sc
import torch

def Gaussian_iid(x,args):
    """sigma=args[1]"""
    return np.exp(-np.abs(x)**2/(2*args[1]**2))

def Gaussian(x,args):
    """sigma=args[1]"""
    return torch.exp(-torch.pow(torch.abs(x),2)/(2*args[1]**2))
    # return np.exp(-np.abs(x)**2/(2*args[1]**2))

def Laplace_iid(x,args):
    """sigma=0.12, args[1]=0.08485281374
       sigma=0.25, args[1]=0.17677669529
       sigma=0.5,  args[1]=0.35355339059
       sigma=0.75, args[1]=0.53033008589
       sigma=1,    args[1]=0.70710678118"""
    # return np.exp(-np.abs(x/args[1]))
    return torch.exp(-torch.abs(x/args[1]))

def Laplace(x,args):
    return torch.exp(-torch.abs(x/args[1]))


def exp_inf(x,args):
    """sigma=1, args[1]=0.000651041701"""
    return torch.exp(-torch.abs(x/args[1]))

def exp_2norm(x,args):
    """"sigma=1, args[1]=0.018039260073589584"""
    return np.exp(-np.abs(x/args[1]))

def cauthy_iid(x,args):
    """sigma=1, args=[-1,0.3345]"""
    return args[1]**2/(x**2+args[1]**2)

def Pareto(x,args):

    """sigma=1,args=[-1,1,1]"""

    return 1/torch.pow((1+torch.abs(x/args[1])),args[2]+1)

def Pareto_R_L1(PA,args):
    F=sc.hyp2f1(1, args[2]/(args[2]+1),args[2]/(args[2]+1)+1 , (2*PA-1)**(1+1/args[2]))
    return args[1]*(2*PA-1)/args[2]*F
def Gen_normal(x,args):
    """sigma=1, args=[-1,1.1645,1.5]
       sigma=1, args=[-1,1.64,3]
       sigma=1, args=[-1,0.1122,0.5]
       sigma=0.12, args[1]=0.16970562748
       sigma=0.25, args[1]=0.35355339059
       sigma=0.5,  args[1]=0.70710678118
       sigma=1,    args[1]=1.41421356237"""
    # return np.exp(-np.abs(x/args[1])**args[2])
    return torch.exp(-torch.pow(torch.abs(x/args[1]),args[2]))

def H_secant(x,args):
    """sigma=1,args=[-1,0.642]"""
    return 1/torch.cosh(x/args[1])

def Gaussian_Lalace_mix(x,args):
    """"sigma=1, args=[-1,0.912]
    [-1,1.06,0.3]"""

    return args[2]*torch.exp(-torch.abs(x / args[1]))+(1-args[2])*torch.exp(-torch.abs(x / args[1])**2)

def Gaussian_R(PA,PB,sigma):
    return sigma * (Norm.ppf(PA) - Norm.ppf(PB)) / 2

def Gaussian_R_infnorm(PA,d,sigma):
    return sigma * Norm.ppf(PA)/np.sqrt(d)

def Gaussian_R_binary(PA,sigma):
    return sigma * Norm.ppf(PA)

def Laplace_R(PA,sigma):
    Lambda=1/np.sqrt(2)*sigma
    return -Lambda*np.log(2*(1-PA))

def Exp_inf_R_1(PA,d,args):

    return 2*d*args[1]*(PA-1/2)

def Exp_inf_R_inf(PA,d,sigma):

    Lambda=np.sqrt(1/((d+1)*(d-1)/(3+1)))*sigma
    return Lambda*np.log(1/2/(1-PA))

def Exp_2norm_R(PA,d,sigma):

    Lambda=np.sqrt(1/(d+1))*sigma
    return Lambda*(d-1)/np.sqrt(d)*np.arctanh(1-2*beta.ppf(1-PA,(d-1)/2,(d-1)/2))

def Exp_2norm_R_2norm(PA,d,sigma):

    Lambda=np.sqrt(1/(d+1))*sigma
    return Lambda*(d-1)*np.arctanh(1-2*beta.ppf(1-PA,(d-1)/2,(d-1)/2))

def compute_sigma(function,args):
    x=torch.linspace(-5,5,1000)
    c=torch.sum(function(x,args)*10/1000)
    sigma=torch.sqrt(torch.sum(x**2*(1/c)*function(x,args)*10/1000))
    return sigma
def compute_sigma_empirical(function,args,d):
    x=torch.rand(d)
    sigma=torch.sqrt(torch.sum(function(x,args)**2/d))
    return sigma

def mix_exp(x,args):
    return torch.exp(-args[2]*torch.abs(x/args[1])**1-(1-args[2])*torch.abs(x/args[1])**2)

def Gen_norm_R_L1(PA,args):
    pdf, discrete_scaler = discrete_pdf_function(Gen_normal,args)
    s_pdf = []
    s = 0
    for p in pdf:
        s = s + p
        s_pdf.append(s)
    left_bound = 0
    for i, v in enumerate(s_pdf):
        if PA < v and PA >= left_bound:
            return (i- pdf.shape[0] / 2) * discrete_scaler
    return -1

def mix_exp_R_L1(PA,args):
    pdf, discrete_scaler = discrete_pdf_function(mix_exp,args)
    s_pdf = []
    s = 0
    for p in pdf:
        s = s + p
        s_pdf.append(s)
    left_bound = 0
    for i, v in enumerate(s_pdf):
        if PA < v and PA >= left_bound:
            return (i- pdf.shape[0] / 2) * discrete_scaler
    return -1

def discrete_pdf_function(pdf,args, discrete_range_factor=10):
    """prepare the discrete version of pdf function"""
    # estimate the sigma
    s = torch.linspace(-10, 10, 500)
    t = pdf(s, args)
    t = t / torch.sum(t * 20 / 500)
    sigma = torch.sqrt(torch.sum(s ** 2 * t * 20 / 500))
    # prepare the discrete pdf
    s_ = torch.linspace(-sigma * discrete_range_factor, sigma * discrete_range_factor, 1000)
    discrete_pdf = pdf(s_, args)
    discrete_pdf = discrete_pdf / torch.sum(discrete_pdf * 1)

    discrete_scaler = 1 / 1000 * 2 * sigma * discrete_range_factor

    return discrete_pdf.data, discrete_scaler

def plot():
    import matplotlib.pyplot as plt
    import matplotlib

    colors=['#e53d00','#ffe900','#00a878','#7A306c','#8e8dbe']
    plt.figure(dpi=300)
    plt.rcParams["font.family"] = "serif"
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    x=torch.linspace(-5,5,1000)
    x1=torch.linspace(-0.75,0.75,1000)
    c1=torch.sum(Gaussian(x,[-1,0.25])*10/1000)
    c2=torch.sum(cauthy_iid(x,[-1,0.01969])*10/1000)
    c3 = torch.sum(H_secant(x, [-1, 0.1592]) * 10 / 1000)
    c4 = torch.sum(Gen_normal(x, [-1, 0.2909,1.5]) * 10 / 1000)
    c5 = torch.sum(Gen_normal(x, [-1, 0.4092, 3]) * 10 / 1000)
    plt.plot(x1,1/c1*Gaussian(x1,[-1,0.25]),label='Gaussian',c=colors[0])
    plt.plot(x1, 1 / c2 * cauthy_iid(x1,[-1,0.01969]),label='Cauthy',c=colors[1])
    plt.plot(x1, 1 / c3 * H_secant(x1, [-1, 0.1592]),label='Hyperbolic Secant',c=colors[2])
    plt.plot(x1, 1 / c4 * Gen_normal(x1, [-1, 0.2909,1.5]),label=r'Gen. Normal ($\beta=1.5$)',c=colors[3])
    plt.plot(x1, 1 / c5 * Gen_normal(x1, [-1, 0.4092, 3]),label=r'Gen. Normal ($\beta=3.0$)',c=colors[4])
    plt.grid()
    plt.ylim(0,4)
    plt.xlabel('$x$',fontsize=15)
    plt.ylabel(r'$\mu(x)$',fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.set_facecolor('#F1F4F4')
    plt.legend(fontsize=12)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./figures/distributions.pdf')


if __name__ == '__main__':
    plot()
    # print(compute_sigma(Gaussian,[-1,0.25]))
    # print(compute_sigma(cauthy_iid,[-1,0.01969]))
    # print(compute_sigma(H_secant, [-1, 0.1592]))
    # print(compute_sigma(Gen_normal, [-1, 0.4092, 3]))
    # print(compute_sigma(mix_exp,[-1,1.05,0.75]))
    # print(mix_exp_R_L1(0.51,[-1,0.90,0.5]))
    # print(Gen_norm_R_L1(0.51,[-1,1.73,4]))
    # print(compute_sigma_empirical(exp_inf,[-1,1],3072))

