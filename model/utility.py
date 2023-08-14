import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import RegularGridInterpolator



def creat_mesh(n,l):
    x = np.linspace(0,2*np.pi,n+1)
    dx = x[1]-x[0]
    x_pre =x[:l] -x[l]-dx
    x_post = x[:l]+x[-1]+dx
    x = np.concatenate([x_pre,x,x_post],axis=0)
    y=x
    z=x
    return x,y,z
def create_periodicity(u,l):
    n = u.shape[0]
    u2 = np.zeros([n+2*l+1,n+2*l+1,n+2*l+1])
    u2[l:n+l,l:n+l,l:n+l] = u

    u2[l:n+l,l:n+l,-l-1:] = u2[l:n+l,l:n+l,l:l+l+1]
    u2[l:n+l,l:n+l,:l] = u2[l:n+l,l:n+l,-l-l-1:-l-1]

    u2[l:n+l,-l-1:,:] = u2[l:n+l,l:l+l+1,:]
    u2[l:n+l,:l,:] = u2[l:n+l,-l-l-1:-l-1,:]

    u2[-l-1:,:,:] = u2[l:l+l+1,:,:]
    u2[:l,:,:] = u2[-l-l-1:-l-1,:,:]
    return u2
def cal_w(n):
    W = np.zeros(n)
    for i in range(n):
        if i <= int(n/2):
            W[i] = i
        else:
            W[i] = i-n
    return W
def generate_data(device):
    ngridx = 64
    Wx = cal_w(ngridx)
    Wy = cal_w(ngridx)
    Wz = cal_w(ngridx)[:int(ngridx/2)+1]

    Wx2 = Wx*Wx
    Wy2 = Wy*Wy
    Wz2 = Wz*Wz
    kmax = ngridx/2*(2*np.sqrt(2)/3)
    tke = 1e4
    pos = np.random.rand(ngridx**3,3)*2*np.pi
    x,y,z = creat_mesh(ngridx,1)
    
    iu = np.zeros((ngridx, ngridx, int(ngridx/2)+1),dtype = np.complex128)
    iux = np.zeros((ngridx, ngridx, int(ngridx/2)+1),dtype = np.complex128)
    iuy = np.zeros((ngridx, ngridx, int(ngridx/2)+1),dtype = np.complex128)
    iuz = np.zeros((ngridx, ngridx, int(ngridx/2)+1),dtype = np.complex128)
    
    iv = np.zeros((ngridx, ngridx, int(ngridx/2)+1),dtype = np.complex128)
    iw = np.zeros((ngridx, ngridx, int(ngridx/2)+1),dtype = np.complex128)
    for i in range(ngridx):
        for j in range(ngridx):
            for k in range(int(ngridx/2)+1):
                W2 = (Wx2[i] + Wy2[j] + Wz2[k])**0.5
                if W2 < kmax and W2 != 0:
                    iu[i,j,k] = tke*complex(np.random.randn(1),np.random.randn(1))
                    iu[i,j,k] = iu[i,j,k]/W2**2
                    iv[i,j,k] = tke*complex(np.random.randn(1),np.random.randn(1))
                    iv[i,j,k] = iv[i,j,k]/W2**2
                    iw[i,j,k] = tke*complex(np.random.randn(1),np.random.randn(1))
                    iw[i,j,k] = iw[i,j,k]/W2**2
                    
                    iux[i,j,k] = iu[i,j,k]*Wx[i]*complex(0,1)
                    iuy[i,j,k] = iu[i,j,k]*Wy[j]*complex(0,1)
                    iuz[i,j,k] = iu[i,j,k]*Wz[k]*complex(0,1)
                    
    u = np.fft.irfftn(iu)
    v = np.fft.irfftn(iu)
    w = np.fft.irfftn(iw)
    train_value = np.stack([u,v,w], axis = 3).reshape([-1,3])
    
    ux = np.fft.irfftn(iux)
    uy = np.fft.irfftn(iuy)
    uz = np.fft.irfftn(iuz)
    train_gradient = np.stack([ux,uy,uz], axis = 3).reshape([-1,3])
    
    u = create_periodicity(u,1)
    v = create_periodicity(v,1)
    w = create_periodicity(w,1)
    
    ux = create_periodicity(ux,1)
    uy = create_periodicity(uy,1)
    uz = create_periodicity(uz,1)

    interp_u = RegularGridInterpolator((x, y, z), u)
    interp_v = RegularGridInterpolator((x, y, z), v)
    interp_w = RegularGridInterpolator((x, y, z), w)
    
    par_u = interp_u(pos)
    par_v = interp_v(pos)
    par_w = interp_w(pos)

    vel = np.stack([par_u, par_v, par_w], axis=1)
    
    
    return torch.from_numpy(train_value).to(device), torch.from_numpy(train_gradient).to(device), torch.from_numpy(pos).to(device), torch.from_numpy(vel).to(device)

class histogramcnn(nn.Module):
    def __init__(self,maxx):
        super(histogramcnn,self).__init__()
        self.outc = 30
        self.dx = maxx*1.05/self.outc
        self.cnst = torch.tensor(self.outc)
        self.l1 = nn.Conv1d(in_channels=1,out_channels=self.outc*2+1, kernel_size = 1)
        self.act = nn.ReLU()
    def set_param(self,device):
        with torch.no_grad():
            self.l1.weight.fill_(1.0)
            self.l1.bias.data=(-torch.tensor(np.linspace(-self.outc,self.outc,self.outc*2+1,dtype=np.float64)).to(device)*self.dx)
    def forward(self,X):
        out = self.act(X+self.outc*self.dx) - self.outc*self.dx
        out = -self.act(-out+self.outc*self.dx) + self.outc*self.dx
        out = self.l1(X)
        out = out.abs()
        out = self.act(out*(-1.0/self.dx)+1.0)
        out = out.sum(axis=2)
        return out
def histtopdf(hist, data1,data2):
    pdf1 = (hist(data1.reshape([1,1,-1]))+1.0e-10)/data1.shape[0]
    pdf2 = (hist(data2.reshape([1,1,-1]))+1.0e-10)/data2.shape[0]
    kl = (pdf1*(pdf1/pdf2).log()).sum() + (pdf2*(pdf2/pdf1).log()).sum()
    return kl

def make_grid(nfield):
    x = np.linspace(0,2*np.pi,nfield+1)
    y = x
    z = x
    xx,yy,zz = np.meshgrid(x[:nfield],y[:nfield],z[:nfield],indexing='ij')
    grid = np.stack([xx,yy,zz]).transpose([1,2,3,0])
    grid = grid.reshape([-1,3])
    return grid

def cal_grid_data(pos_traj,vel_traj,rho_traj,grid,kernel,neighbor_train):
    print('Generating GT field data')
    n_learn = pos_traj.shape[0]
    pos_tmp = pos_traj.cpu().detach().numpy()
    vel_tmp = vel_traj.cpu().detach().numpy()
    rho_tmp = rho_traj.cpu().detach().numpy()
    h = kernel.h.cpu().detach().numpy()

    traj_gt_period = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = np.ones(pos_tmp.shape)*np.array([2*np.pi*(i-1),2*np.pi*(j-1),2*np.pi*(k-1)])
                b = pos_tmp
                traj_gt_period.append(a+b)
    traj_gt_period=np.stack(traj_gt_period)
    traj_gt_period = traj_gt_period.reshape([-1,3])

    nbrs = NearestNeighbors(n_neighbors=neighbor_train, algorithm='ball_tree').fit(traj_gt_period)

    distances, neighbor_new = nbrs.kneighbors(grid)
    pdistances, pneighbor_new = nbrs.kneighbors(pos_tmp)

    neighbor = (np.remainder(neighbor_new,n_learn))

    rho = np.zeros([n_learn])

    for n in range(neighbor_train):
        dis = torch.tensor(pdistances[:,n]/h)
        with torch.no_grad():
            wxyz = kernel.wnn_nn(dis)
        rho = rho + wxyz.detach().numpy().reshape([-1])

    grid_field = np.zeros([grid.shape[0],4])
    w_field = np.zeros([grid.shape[0],neighbor_train])
    for n in range(neighbor_train):
        dis = torch.tensor(distances[:,n]/h)
        with torch.no_grad():
            wxyz = kernel.wnn_nn(dis)
        w_field[:,n] = wxyz.detach().numpy()/rho[neighbor[:,n]]
        grid_field[:,0] = grid_field[:,0] + rho_tmp[neighbor[:,n],0]*w_field[:,n]
        grid_field[:,1] = grid_field[:,1] + vel_tmp[neighbor[:,n],0]*w_field[:,n]
        grid_field[:,2] = grid_field[:,2] + vel_tmp[neighbor[:,n],1]*w_field[:,n]
        grid_field[:,3] = grid_field[:,3] + vel_tmp[neighbor[:,n],2]*w_field[:,n]


    return grid_field,w_field,rho, neighbor


def generate_data_Lag(device):
    ngridx = 64
    Wx = cal_w(ngridx)
    Wy = cal_w(ngridx)
    Wz = cal_w(ngridx)[:int(ngridx/2)+1]

    Wx2 = Wx*Wx
    Wy2 = Wy*Wy
    Wz2 = Wz*Wz
    kmax = ngridx/2*(2*np.sqrt(2)/3)
    tke = 1e4
    rhorms = 1e3
    pos = np.random.rand(ngridx**3,3)*2*np.pi
    x,y,z = creat_mesh(ngridx,1)
    
    iu = np.zeros((ngridx, ngridx, int(ngridx/2)+1),dtype = np.complex128)
    iv = np.zeros((ngridx, ngridx, int(ngridx/2)+1),dtype = np.complex128)
    iw = np.zeros((ngridx, ngridx, int(ngridx/2)+1),dtype = np.complex128)
    
    irho = np.zeros((ngridx, ngridx, int(ngridx/2)+1),dtype = np.complex128)

    for i in range(ngridx):
        for j in range(ngridx):
            for k in range(int(ngridx/2)+1):
                W2 = (Wx2[i] + Wy2[j] + Wz2[k])**0.5
                if W2 < kmax and W2 != 0:
                    iu[i,j,k] = tke*complex(np.random.randn(1),np.random.randn(1))
                    iu[i,j,k] = iu[i,j,k]/W2**2
                    iv[i,j,k] = tke*complex(np.random.randn(1),np.random.randn(1))
                    iv[i,j,k] = iv[i,j,k]/W2**2
                    iw[i,j,k] = tke*complex(np.random.randn(1),np.random.randn(1))
                    iw[i,j,k] = iw[i,j,k]/W2**2
                    irho[i,j,k] = rhorms*complex(np.random.randn(1),np.random.randn(1))
                    irho[i,j,k] = irho[i,j,k]/W2**2                    
                    
    u = np.fft.irfftn(iu)
    v = np.fft.irfftn(iu)
    w = np.fft.irfftn(iw)
    rho = np.fft.irfftn(irho)
    
    u = create_periodicity(u,1)
    v = create_periodicity(v,1)
    w = create_periodicity(w,1)
    rho = create_periodicity(rho,1)
    
    interp_u = RegularGridInterpolator((x, y, z), u)
    interp_v = RegularGridInterpolator((x, y, z), v)
    interp_w = RegularGridInterpolator((x, y, z), w)
    interp_rho = RegularGridInterpolator((x, y, z), rho)
    
    par_u = interp_u(pos)
    par_v = interp_v(pos)
    par_w = interp_w(pos)
    par_rho = interp_rho(pos)

    vel = np.stack([par_u, par_v, par_w], axis=1)
    
    pos_next = pos + vel*0.05
    
    par_unext = interp_u(pos_next)
    par_vnext = interp_v(pos_next)
    par_wnext = interp_w(pos_next)
    par_rhonext = interp_rho(pos_next)
    vel_next = np.stack([par_unext, par_vnext, par_wnext], axis=1)
    
    pos_traj = np.stack([pos, pos_next])
    vel_traj = np.stack([vel, vel_next])
    rho_traj = np.stack([par_rho, par_rhonext])
    return torch.from_numpy(pos_traj).to(device), torch.from_numpy(vel_traj).to(device), torch.from_numpy(rho_traj.reshape([2,-1,1])).to(device)
