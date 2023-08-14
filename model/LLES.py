import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.functional import vjp, vhp, jacobian, hessian
from scipy.interpolate import RegularGridInterpolator


from torch.autograd import grad
from torch.autograd import Variable

from copy import copy

class kernel(nn.Module):
    def __init__(self, N,nfield,neighbor_train,device):
        super(kernel, self).__init__()
        self.N = N
        self.D = torch.tensor(3)
        self.device =device
        self.pi = torch.tensor(3.14159265358).to(self.device)

        self.h = nn.Parameter(torch.tensor((((np.pi*2)**3/N*neighbor_train)/np.pi/(4/3))**(1/3)),requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(100.0),requires_grad=True)

        self.nfield = nfield

        self.lneighbor = neighbor_train
        self.neighbor = torch.zeros([self.N,self.lneighbor],dtype=torch.long).to(self.device)
        self.fneighbor = torch.zeros([self.nfield,self.lneighbor],dtype=torch.long).to(self.device)

        self.qp = torch.linspace(0,1,101).to(self.device)
        self.l1 = nn.Linear(1,20)
        self.l2 = nn.Linear(20,100)
        self.l3 = nn.Linear(100,20)
        self.l4 = nn.Linear(20,1)
        self.act = nn.Tanh()
    def wnn_nn(self,r):
        out = self.l1(r.reshape([-1,1]))
        out = self.act(out)
        out = self.l2(out)
        out = self.act(out)
        out = self.l3(out)
        out = self.act(out)
        out = self.l4(out)
        out = out*torch.sigmoid(10*(1-r)).reshape([-1,1])*self.alpha
        return out.reshape([-1])

    def wnn_r(self,r):
        dr = 0.00001
        x1 = r.reshape([-1,1]) + dr
        x2 = r.reshape([-1,1]) - dr
        y1 = self.wnn_nn(x1)
        y2 = self.wnn_nn(x2)
        return (y1-y2)/(2*dr)/self.h

    def wnn_r_grad(self,r):
        out = vjp(self.wnn_nn, r.reshape([-1,1]), torch.ones(r.shape[0]).to(self.device),create_graph=True )[1]

        return out/self.h
    def wnn_drr(self):
        out = vjp(self.wnn_r_grad,torch.tensor(0.0).reshape([1]).to(self.device), torch.ones(1,1).to(self.device),create_graph=True )[1]

        return out.flatten()/self.h

    def cal_integral(self):
        dh = (self.qp[1]-self.qp[0])*self.h
        y = self.wnn_nn(self.qp).reshape([-1])
        surface = 4.0*self.pi*(self.qp*self.h).pow(2)
        y = y*surface
        return 0.5*dh*(y[0]+y[-1]+2.0*(y[1:-1].sum()))
    
    
    def cal_disv(self,X,Xfield,i,batch):
        temp1 = torch.abs(Xfield[batch]-X[self.fneighbor[batch,i]])
        temp1_1 = -torch.sign(Xfield[batch]-X[self.fneighbor[batch,i]])*torch.sign(Xfield[batch]-X[self.fneighbor[batch,i]]+torch.ones(temp1.shape).to(self.device)*\
self.pi)*torch.sign(Xfield[batch]-X[self.fneighbor[batch,i]]-torch.ones(temp1.shape).to(self.device)*self.pi)
        temp2 = torch.ones(temp1.shape).to(self.device)*self.pi*2.0-temp1
        out = temp1_1*torch.min(torch.stack([temp1,temp2],axis=2),axis=2)[0]
        out2 = torch.sum(out*out,axis=1).reshape([-1,1])
        return torch.sqrt(out2)/self.h, out/torch.sqrt(out2)

    def cal_dis(self,X,i,batch):
        temp1 = torch.unsqueeze(torch.abs(X[batch]-X[self.neighbor[batch,i]]),2)
        temp2 = torch.ones(temp1.shape).to(self.device)*self.pi*2.0-temp1
        out2 = torch.cat((temp1,temp2),axis=2)
        out2 = torch.min(torch.stack([temp1,temp2],axis=2),axis=2)[0]
        return torch.sqrt(torch.sum(out2*out2,axis=1))/self.h


    def cal_dis_field(self,X,Xfield,i,batch):
        temp1 = torch.unsqueeze(torch.abs(Xfield[batch]-X[self.fneighbor[batch,i]]),2)
        temp2 = torch.ones(temp1.shape).to(self.device)*self.pi*2.0-temp1
        out2 = torch.cat((temp1,temp2),axis=2)
        out2 = torch.min(torch.stack([temp1,temp2],axis=2),axis=2)[0]
        return torch.sqrt(torch.sum(out2*out2,axis=1))/self.h
    
    
    def cal_rho_nn(self,X,batch):
        rho = self.wnn_nn(torch.zeros([batch.shape[0]]).to(self.device))
        for i in range(self.lneighbor):
            dis = self.cal_dis(X,i,batch)
            rho = rho+self.wnn_nn(dis)
        return rho

    def cal_rho_nn_field(self,X,Xfield,batch):
        rho = self.wnn_nn(torch.zeros([batch.shape[0]]).to(self.device))
        for i in range(self.lneighbor):
            dis = self.cal_dis_field(X,Xfield,i,batch)
            rho = rho+self.wnn_nn(dis)
        return rho
    
    
    def cal_f_nn(self,X,Xfield,f,ffield,batch):
        rho_f = self.cal_rho_nn_field(X,Xfield,batch).reshape([-1,1])
        rho = torch.zeros([batch.shape[0],f.shape[-1] ]).to(self.device)
        drhodx = torch.zeros([batch.shape[0],self.D ]).to(self.device)

        for i in range(self.lneighbor):
            rho_p = self.cal_rho_nn(X,self.fneighbor[batch,i])
            dis,disv = self.cal_disv(X,Xfield,i,batch)
            rho = rho+f[self.fneighbor[batch,i]]*self.wnn_nn(dis).reshape([-1,1])/rho_p.reshape([-1,1])
            dwdr = self.wnn_r(dis).reshape([-1,1])
            drhodx = drhodx + (f[self.fneighbor[batch,i],1]-ffield[batch]).reshape([-1,1])*disv*dwdr
        return rho, drhodx/rho_f
    def update_neighborlist_sklearn(self,X,Xfield):
        traj_gt_period = []
        traj_copy = X.clone().cpu()
        traj_eu = Xfield.clone().cpu()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    traj_gt_period.append(traj_copy.cpu()+np.ones(traj_copy.shape)*np.array([2*np.pi*(i-1),2*np.pi*(j-1),2*np.pi*(k-1)]))
        traj_gt_period=np.stack(traj_gt_period)
        traj_gt_period = traj_gt_period.reshape([-1,self.D])
        nbrs = NearestNeighbors(n_neighbors=self.lneighbor, algorithm='ball_tree').fit(traj_gt_period)
        distances, neighbor_new = nbrs.kneighbors(traj_copy)
        neighbor_new= torch.from_numpy(neighbor_new).to(self.device)
        self.neighbor = (torch.remainder(neighbor_new,self.N)).clone().to(self.device)

        distances, neighbor_new = nbrs.kneighbors(traj_eu)
        neighbor_new= torch.from_numpy(neighbor_new).to(self.device)
        self.fneighbor = (torch.remainder(neighbor_new,self.N)).clone().to(self.device)


class LLES(nn.Module):
    def __init__(self,device, N,dt, vref, tref,rhoref, nfield, kernel_wnn ,neighbor_train):
        super(LLES, self).__init__()
        self.N = N  
        self.D = 3
        self.nfeat = 5
        self.nfeat_out_vel  = 2
        self.nfeat_out_rho  = 1

        self.nfield = nfield
        self.kernel = copy(kernel_wnn)
        self.device = device
        self.dt = dt

        self.pi = torch.tensor(3.14159265358).to(self.device)
        self.lneighbor = neighbor_train
        self.neighbor = torch.zeros([self.N,self.lneighbor],dtype=torch.long).to(self.device)
        self.fneighbor = torch.zeros([self.nfield,self.lneighbor],dtype=torch.long).to(self.device)
        self.h=torch.tensor((((np.pi*2)**3/N*neighbor_train)/np.pi/(4/3))**(1/3)).to(device)
        
        self.alpha1 = nn.Parameter(torch.tensor(0.1),requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(0.1),requires_grad=True)

        self.beta1 = nn.Parameter(torch.tensor(0.1),requires_grad=True)
        self.beta2 = nn.Parameter(torch.tensor(0.1),requires_grad=True)

        self.vref = vref
        self.tref = tref
        self.aref = self.vref/self.tref
        self.rhoref = rhoref
        self.drhoref = self.rhoref/self.tref
        
        self.l1 = nn.Linear(self.nfeat, 20)
        self.l2 = nn.Linear(20,100)
        self.l3 = nn.Linear(100,20)
        self.l4 = nn.Linear(20,self.nfeat_out_vel)

        self.l1_rho = nn.Linear(self.nfeat, 20)
        self.l2_rho = nn.Linear(20,100)
        self.l3_rho = nn.Linear(100,20)
        self.l4_rho = nn.Linear(20,self.nfeat_out_rho)

        self.act = nn.Tanh()

    def wnn(self,r):
        with torch.no_grad():
            out = self.kernel.wnn_nn(r.reshape([-1,1]))
        return out

    def knn_nn(self,r):
        out = self.l1(r.reshape([-1,self.nfeat]))
        out = self.act(out)
        out = self.l2(out)
        out = self.act(out)
        out = self.l3(out)
        out = self.act(out)
        out = self.l4(out)

        out_rho = self.l1_rho(r.reshape([-1,self.nfeat]))
        out_rho = self.act(out_rho)
        out_rho = self.l2_rho(out_rho)
        out_rho = self.act(out_rho)
        out_rho = self.l3_rho(out_rho)
        out_rho = self.act(out_rho)
        out_rho = self.l4_rho(out_rho)

        return out, out_rho
    
    def cal_dis(self,X,i,batch):
        temp1 = torch.unsqueeze(torch.abs(X[batch]-X[self.neighbor[batch,i+1]]),2)
        temp2 = torch.ones(temp1.shape).to(self.device)*self.pi*2.0-temp1
        out2 = torch.cat((temp1,temp2),axis=2)
        out2 = torch.min(torch.stack([temp1,temp2],axis=2),axis=2)[0]
        return torch.sqrt(torch.sum(out2*out2,axis=1))/self.kernel.h
    def cal_disv(self,X,V,rho,i,batch):
        temp1 = torch.abs(X[batch]-X[self.neighbor[batch,i+1]])
        temp1_1 = -torch.sign(X[batch]-X[self.neighbor[batch,i+1]])*torch.sign(X[batch]-X[self.neighbor[batch,i+1]]+torch.ones(temp1.shape).to(self.device)*self.pi)\
*torch.sign(X[batch]-X[self.neighbor[batch,i+1]]-torch.ones(temp1.shape).to(self.device)*self.pi)
        temp2 = torch.ones(temp1.shape).to(self.device)*self.pi*2.0-temp1
        out = temp1_1*torch.min(torch.stack([temp1,temp2],axis=2),axis=2)[0]
        outv = (V[batch]-V[self.neighbor[batch,i+1]]).reshape([-1,self.D])
        out = out/self.h
        outv = outv/self.vref
        outc = torch.cross(out,outv,axis=1)
        out2 = torch.sum(out*out,axis=1).reshape([-1,1])
        outv2 = torch.sum(outv*outv,axis=1).reshape([-1,1])
        out2v = torch.sum(out*outv,axis=1).reshape([-1,1])
        drho1 = (rho[batch]).reshape([-1,1])/self.rhoref
        drho2 = (rho[self.neighbor[batch,i+1]]).reshape([-1,1])/self.rhoref

        return torch.cat([drho1,drho2, torch.sqrt(out2),torch.sqrt(outv2),out2v],axis=1), torch.cat([(drho1-drho2)/(drho1-drho2).abs(), out/torch.sqrt(out2),outv/torch.sqrt(outv2)],axis=1)

    def cal_rho_nn(self,X,batch):
        prho = torch.zeros([batch.shape[0],1]).to(self.device)
        for i in range(self.lneighbor):
            dis = self.cal_dis(X,i-1,batch)
            w = self.wnn(dis)
            prho[:] = prho[:] + w
        return prho

    def cal_a_nn(self,X,V,rho,batch):
        drho = torch.zeros([batch.shape[0],1+self.D]).to(self.device)
        for i in range(1,self.lneighbor):
            feature,dis = self.cal_disv(X,V,rho,i-1,batch)
            knn_out, knn_out_rho = self.knn_nn(feature)
            drho[:,0] = drho[:,0] + (self.drhoref)*knn_out_rho[:,0]*dis[:,0]
            drho[:,1:] = drho[:,1:] + self.aref*(knn_out[:,:].reshape([-1,2,1])*dis[:,1:].reshape([-1,2,self.D])).sum(axis=1)
            av = self.cal_av(feature[:,0]-feature[:,1],feature[:,2], feature[:,4],dis[:,1:].reshape([-1,2,self.D])[:,0] )
            drho[:,0] = drho[:,0] + self.drhoref*av[:,0]
            drho[:,1:] = drho[:,1:]  + self.aref*av[:,1:]
        return drho
    def cal_field(self,X,V,rho,batch,w_field):
        field = torch.zeros([batch.shape[0],1+self.D]).to(self.device)
        for i in range(self.lneighbor):
            accl = self.cal_a_nn(X,V,rho,self.fneighbor[batch,i])
            field[:,:] = field[:,:] + w_field[batch,i].reshape([-1,1])*(V[self.fneighbor[batch,i]]+accl*self.dt*0.1)
        return field

    def cal_field_kl(self,X,V,rho,batch,w_field):
        field = torch.zeros([batch.shape[0],1+self.D]).to(self.device)
        accl_t = []
        accl_gt = []

        rho_t = []
        rho_gt = []

        for i in range(self.lneighbor):
            accl = self.cal_a_nn(X,V[0],rho[0],self.fneighbor[batch,i])
            field[:,1:] = field[:,1:] + w_field[batch,i].reshape([-1,1])*(V[0,self.fneighbor[batch,i]]+accl[:,1:]*self.dt)
            field[:,0] = field[:,0] + w_field[batch,i].reshape([-1])*(rho[0,self.fneighbor[batch,i],0]+accl[:,0]*self.dt)

            accl_t.append(accl[:,1:])
            accl_gt.append((V[1,self.fneighbor[batch,i]] - V[0,self.fneighbor[batch,i]])/self.dt)
            rho_t.append(accl[:,0])
            rho_gt.append((rho[1,self.fneighbor[batch,i]] - rho[0,self.fneighbor[batch,i]])/self.dt)
        return field,torch.stack(accl_t).flatten(),torch.stack(accl_gt).flatten(),torch.stack(rho_t).flatten(), torch.stack(rho_gt).flatten()
    def cal_av(self, drho,xx,xv, vec):
        out_rho = drho*self.h**2/(xx**2 + 0.1*self.h**2)
        out_rho = -(torch.abs(self.beta1)+torch.abs(self.beta2)*torch.abs(out_rho))*out_rho
        out_rho = out_rho.reshape([-1,1])

        out = -1.0*self.h*self.act(-1.0*xv)/((xx)**2+0.1*self.h**2)
        out = -1.0*torch.abs(self.alpha1)*out + torch.abs(self.alpha2)*out**2
        out = out.reshape([-1,1])*vec
        return torch.cat([out_rho,out],axis=1)

    def update_neighborlist_sklearn(self,X,fneighbor):
        traj_gt_period = []
        traj_copy = X.clone().cpu()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    traj_gt_period.append(traj_copy.cpu()+np.ones(traj_copy.shape)*np.array([2*np.pi*(i-1),2*np.pi*(j-1),2*np.pi*(k-1)]))
        traj_gt_period=np.stack(traj_gt_period)
        traj_gt_period = traj_gt_period.reshape([-1,self.D])
        nbrs = NearestNeighbors(n_neighbors=self.lneighbor, algorithm='ball_tree').fit(traj_gt_period)
        distances, neighbor_new = nbrs.kneighbors(traj_copy)
        neighbor_new= torch.from_numpy(neighbor_new).to(self.device)
        self.neighbor = (torch.remainder(neighbor_new,self.N)).clone().to(self.device)

        self.fneighbor = torch.from_numpy(fneighbor).to(self.device)
        return traj_gt_period
    def cal_forcing(self,X,V,batch):
        force = torch.zeros([batch.shape[0],self.D]).to(self.device)
        for i in range(self.lneighbor):
            dis = self.cal_dis(X,i-1,batch)
#            w = self.wnn(dis)                                                     
            force = force + V[self.neighbor[batch,i]]/self.lneighbor
        return force


