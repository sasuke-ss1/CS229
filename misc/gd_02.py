import random
import numpy as np
from matplotlib import pyplot as plt

hyp=lambda x,t0=100,t1=1/2: t0+t1*x

def batch_grd_dsc(x,y,t0=0,t1=1,lr=1,tol=9):
    cnt=0; err=10; lerr=[err]; lt0=[t0]; lt1=[t1]
    while np.abs(err)>tol and cnt<500000:
        t0=t0+lr*sum((y[i]-hyp(x=x[i],t0=t0,t1=t1))*1 for i in range(len(x)))
        lt0.append(t0)
        t1=t1+lr*sum((y[i]-hyp(x=x[i],t0=t0,t1=t1))*x[i] for i in range(len(x)))
        lt1.append(t1)
        err=sum((y[i]-hyp(x=x[i],t0=t0,t1=t1))*1 for i in range(len(x)))
        lerr.append(err)
        cnt+=1
    return lt0,lt1,lerr,cnt

def stochastic_grd_dsc(x,y,t0=0,t1=1,lr=1,tol=9,ttol=1e-5,msicnt=10):
    cnt=0; err=10; lerr=[err]; lt0=[t0]; lt1=[t1]; sicnt=0;
    while sicnt<msicnt and np.abs(err)>tol and cnt<500000:
        for i in range(len(x)):
            t0=t0+lr*(y[i]-hyp(x=x[i],t0=t0,t1=t1))*1
            t1=t1+lr*(y[i]-hyp(x=x[i],t0=t0,t1=t1))*x[i]
        lt0.append(t0)
        lt1.append(t1)
        err=sum((y[i]-hyp(x=x[i],t0=t0,t1=t1))*1 for i in range(len(x)))
        lerr.append(err)
        pt0=lt0[-2];pt1=lt1[-2];
        if np.abs(pt0-t0)<=ttol and np.abs(pt1-t1)<=ttol: sicnt+=1
        cnt+=1
    return lt0,lt1,lerr,cnt

def graph_ineff(funct, x_range, cl='r--', label=None, show=False):
    y_range=[]
    for x in x_range:
        y_range.append(funct(x))
    plt.plot(x_range,y_range,cl,label=label)
    if label is not None:
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=2, mode="expand", borderaxespad=0.)
    if show: plt.show()

apply_lr_params=lambda x,t0=0,t1=1: t0+t1*x

def run(pltr=True):
    np.random.seed(3)
    x=[21.4,23.4,20.1,23.1];y=[402.6,428.6,373.6,421.8]
    n=100;x=np.random.normal(13,3,n);os=np.random.normal(0,39,n);y=30*x+os
    print("DD x={} y={}".format(x[:4],y[:4]))
    #lt0,lt1,lerr,cnt = batch_grd_dsc(x=x,y=y,t0=0,t1=1,lr=0.000001)
    lt0,lt1,lerr,cnt = stochastic_grd_dsc(x=x,y=y,t0=0,t1=1,lr=0.000001)
    print("lt0={} lt1={} lerr={} cnt={}".format(lt0[-1],lt1[-1],lerr[-1],cnt))
    if pltr:
        plt.scatter(x,y)
        graph_ineff(lambda x1: apply_lr_params(x=x1,t0=lt0[-1],t1=lt1[-1]),x,show=True)

if __name__ == '__main__':
    run()
