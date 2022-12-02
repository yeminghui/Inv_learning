import torch.nn as nn
import torch
import argparse
import os
import numpy as np
import itertools
import time
from write import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR


parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="agent")
parser.add_argument("--exp-itr", type=str, default="1")

parser.add_argument("--data-file", type=str, default="48")
parser.add_argument("--gap", type=int, default=0)
parser.add_argument("--checkpoint", type=str, default='6x192debug.tar' )
parser.add_argument("--seeds", type=int, default=2)

parser.add_argument("--layer", type=int, default=6, )
parser.add_argument("--neuron", type=int, default=192)

parser.add_argument("--n-epoch", type=int, default=15000)
parser.add_argument("--batch-size", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--gxy-lr", type=float, default=0.0001)
parser.add_argument("--gyx-decrease", type=int, default=350)
parser.add_argument("--gxy-decrease", type=int, default=1000)
parser.add_argument("--device", type=str, default="7")
parser.add_argument("--decay_epoch", type=int, default=100)
parser.add_argument("--checkpoint_interval", type=int, default=-1)
opt = parser.parse_args()


Tensor = torch.cuda.FloatTensor


def para_rescale(decision):
    multiply = Tensor(np.array([1700,1500,14,1,7]).reshape([1,5]))
    add_to = Tensor(np.array([3000,2500,25,17.5,15]).reshape([1,5]))
    # add_to = Tensor(np.zeros([1,5]))
    decision = decision * multiply + add_to
    return decision


class agent_net(nn.Module):
    def __init__(self,input_shape=3,lay1_unit=32,lay2_unit= 32,lay3_unit=32,lay4_unit= 32,
                 lay5_unit= 32, output_shape=5):

        super(agent_net, self).__init__()

        self.layer6 = nn.Sequential(
        nn.Linear(1, output_shape,bias=None),
        nn.Sigmoid()
    )

    def forward(self,x):
        out = self.layer6(x)
        return out


class G_xy(nn.Module):
    def __init__(self, input_shape=6, output_shape=2):
        super(G_xy, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_shape, opt.neuron),
            # nn.BatchNorm1d(91),
            nn.Tanh()
        )
        self.module_list = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(opt.neuron, opt.neuron, bias=True),
                # nn.BatchNorm1d(91),
                nn.Tanh())
                for _ in range(opt.layer )
            ]
        )
        self.layer_end = nn.Sequential(
            nn.Linear(opt.neuron, output_shape)
        )

    def forward(self, x):
        out = self.layer1(x)
        for i, l in enumerate(self.module_list):
            out = l(out)
        out = self.layer_end(out)
        return out


class MyTestset(Dataset):
    def __init__(self, data_path="./exp_data/"+opt.data_file):
        self.data_path = data_path
        self.x = np.loadtxt(data_path+"/log_x.txt")
        self.first = np.loadtxt(data_path + "/log_first_stress.txt")
        self.sheer = np.loadtxt(data_path+"/log_sheer_stress.txt")

        print(self.first.shape)
        print(self.sheer.shape)
        opt.gap = self.sheer.shape[0] - self.first.shape[0]

        print('gap:',opt.gap)

    def __len__(self):
        # return self.para.shape[0]
        return 1

    def __getitem__(self, idx):
        return self.x,self.sheer,self.first

# Init all
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"]=opt.device

# Load Data
testloader = DataLoader(dataset=MyTestset(),batch_size=1,shuffle=False)
mseloss = torch.nn.MSELoss()
maeloss = torch.nn.MSELoss(reduce=None)

g_xy = G_xy().cuda()
checkpoint = torch.load('./gxymodel/'+opt.checkpoint ,map_location=lambda storage, loc: storage.cuda(0))
g_xy.load_state_dict(checkpoint['state_dict'])

# lr_scheduler_agent = MultiStepLR(optimizer_agent, milestones=[500, ], gamma=0.1)


# '''
####################### Start Training #######################
start_time = time.time()
mul = Variable(
    Tensor(np.array([[3.53, 0., 0., 0., 0.],
                     [0., 3.33, 0., 0., 0.],
                     [0., 0., 3.49, 0., 0.],
                     [0., 0., 0., 5., 0.],
                     [0., 0., 0., 0., 3.5]])), requires_grad=False)

add = Variable(
    Tensor(np.array([-1.77, -1.67, -1.72, -2.5, -2]).reshape(1, 5)), requires_grad=False)

agent_input=Tensor(np.ones((1,1)).reshape(1,1)).cuda()

para_list_all = []
loss_list_all = []
for times in range(opt.seeds):
    agent = agent_net().cuda()
    agent.train()
    for m in agent.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -7, 7)
    optimizer_agent = torch.optim.Adam(itertools.chain(agent.parameters()), lr=opt.lr)
    lr_scheduler_agent = ReduceLROnPlateau(optimizer_agent, mode='min', factor=0.1, min_lr=1e-2, patience=300,
                                           verbose=False)

    para_list = []
    loss_list = []
    for step in range(opt.n_epoch):
        for epoch, (x,sheer,first) in enumerate(testloader):
            agent.train()
            optimizer_agent.zero_grad()
            x = x.float().cuda()
            sheer=sheer.float().cuda()
            first=first.float().cuda()
            y_input = torch.cat([first, sheer], dim=1)

            action=agent(agent_input)

            action = torch.matmul(action,mul)+add
            x_single=torch.chunk(x,91, dim=1)
            x_para_input=torch.cat([x_single[0],action],dim=1)
            y_out=g_xy(x_para_input)
            first_list, sheer_list=y_out.chunk(2, dim=1)
            for i in range(1,x.shape[1]):
                x_para_input = torch.cat([x_single[i], action], dim=1)
                y_out = g_xy(x_para_input)
                y1, y2 = y_out.chunk(2, dim=1)
                first_list=torch.cat([first_list, y1],dim=1)
                sheer_list=torch.cat([sheer_list, y2],dim=1)
            first_list=first_list[:,opt.gap:]
            loss=mseloss(torch.cat([first_list,sheer_list],dim=1),y_input)

            loss.backward()
            optimizer_agent.step()
            lr_scheduler_agent.step(loss)

        if step%1000 ==0:
            para = para_rescale(action)
            para = para.cpu().detach().numpy()[0]
            loss = loss.cpu().detach().numpy()

            write_property("constitutiveProperties", order=str(opt.data_file)+'_'+str(times), etas=0.001, etap=para[0],
                            lamdaD=para[1] , lamdaR=para[2] ,
                            chimax=para[3], beta=para[4],
                            delta=-0.5, lamda=0.00001)

            print("Predicted Parameter:",para)

            print('Seed number:',times,"Step number:",step," time:",time.time()-start_time)
            start_time=time.time()
            print("Mse loss:",loss)
            print("Step size:", optimizer_agent.state_dict()['param_groups'][0]['lr'], "\n")

            para_list.append(para)
            loss_list.append(loss)

    para_list_all.append(np.vstack(para_list))
    loss_list_all.append(np.vstack(loss_list))

np.save("para_" + str(opt.data_file), np.stack(para_list_all))
np.save("loss_agent_" + str(opt.data_file), np.stack(loss_list_all))
