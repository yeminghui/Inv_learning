import torch.nn as nn
import torch
import argparse
import os
import numpy as np
import time
from write import change_property, write_property
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR


np.set_printoptions(precision=4)
parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="agent")
parser.add_argument("--exp-itr", type=str, default="1")
parser.add_argument("--testset", type=str, default="theory")

parser.add_argument("--order", type=int, default=1)
parser.add_argument("--layer", type=int, default=6, )
parser.add_argument("--neuron", type=int, default=192,)
parser.add_argument("--checkpoint", type=str, default='6x192debug.tar' )
parser.add_argument("--seeds", type=int, default=2)

parser.add_argument("--n-epoch", type=int, default=15000)
parser.add_argument("--batch-size", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.04)
parser.add_argument("--gxy-lr", type=float, default=0.0001)
parser.add_argument("--gyx-decrease", type=int, default=350)
parser.add_argument("--gxy-decrease", type=int, default=1000)
parser.add_argument("--decay_epoch", type=int, default=100)
opt = parser.parse_args()

Tensor = torch.cuda.FloatTensor


def para_rescale(decision):
    multiply = Tensor(np.array([1700,1500,14,1,7]).reshape([1,5]))
    add_to = Tensor(np.array([3000,2500,25,17.5,15]).reshape([1,5]))
    # add_to = Tensor(np.zeros([1,5]))
    decision = decision * multiply + add_to
    return decision


class agent(nn.Module):
    def __init__(self, input_shape=3, lay1_unit=32, lay2_unit=32, lay3_unit=32, lay4_unit=32,
                 lay5_unit=32, output_shape=5*opt.seeds):
        super(agent, self).__init__()
        self.layer6 = nn.Sequential(
            nn.Linear(1, output_shape, bias=None),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer6(x)
        return out.view(opt.seeds, 1, 5) #


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
    def __init__(self, data_path="./log_data/"+ opt.testset):
        self.data_path = data_path
        self.x = np.loadtxt(data_path + "/log_x.txt")
        self.para = np.loadtxt(data_path + "/log_para_metricx.txt")[opt.order, :]
        self.first = np.loadtxt(data_path + "/log_first_stress.txt")[opt.order, :]
        self.sheer = np.loadtxt(data_path + "/log_sheer_stress.txt")[opt.order, :]

    def __len__(self):
        # return self.para.shape[0]
        return 1

    def __getitem__(self, idx):
        return self.x, self.para, self.first, self.sheer


# Init all
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# os.environ["CUDA_VISIBLE_DEVICES"]=opt.device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
Tensor = torch.cuda.FloatTensor
# 数据
testloader = DataLoader(dataset=MyTestset(), batch_size=1, shuffle=False)
mseloss = torch.nn.MSELoss()
mseloss_not_reduce = torch.nn.MSELoss(reduce=False)
# l1loss=torch.nn.L1Loss(reduce=None)

# 网络
agent = agent().cuda()
agent.train()
for m in agent.modules():
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight,-7,7)


agent = Variable(Tensor(np.random.randn(opt.seeds,1,5)), requires_grad=True)

g_xy = G_xy().cuda()
checkpoint = torch.load('./gxymodel/'+opt.checkpoint ,map_location=lambda storage, loc: storage.cuda(0))
g_xy.load_state_dict(checkpoint['state_dict'])
# 优化器
# optimizer_agent = torch.optim.Adam(itertools.chain(agent.parameters()), lr=opt.lr)
optimizer_agent = torch.optim.Adam([agent], lr=opt.lr)

# optimizer_agent = torch.optim.Adam(itertools.chain(agent.parameters()), lr=opt.lr)
# optimizer_agent = torch.optim.SGD(itertools.chain(agent.parameters()), lr=opt.lr)
# optimizer_agent = torch.optim.LBFGS(itertools.chain(agent.parameters()) )
# 学习率调整
lr_scheduler_agent = ReduceLROnPlateau(optimizer_agent, mode='min', factor=0.1, min_lr=0.00001, patience=200, verbose=False)
# lr_scheduler_agent = MultiStepLR(optimizer_agent, milestones=[7500, ], gamma=0.1)


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


etap = []
lamdaD = []
lamdaR = []
chiMax = []
beta = []
loss_agent_list = []
agent_input = Tensor(np.array([1]).reshape(1, 1)).cuda()
g_xy.eval()

para_list = []
loss_list = []

for step in range(opt.n_epoch):
    # 模型训练
    for epoch, (x, para, first, sheer) in enumerate(testloader):

        optimizer_agent.zero_grad()
        para = para.float().cuda()
        x = x.float().cuda()
        first = first.float().cuda()
        sheer = sheer.float().cuda()

        x_single = torch.chunk(x, 91, dim=1)
        y_input = torch.cat([first, sheer], dim=1)

        # action_out = agent(agent_input) # (opt.seeds, 1, 5)
        action_out = 1/(1 + torch.exp(-agent))
        action_rescale = torch.matmul(action_out, mul) + add

        x = x.unsqueeze(-1).expand(opt.seeds, 91, 1) # [opt.seeds x91x1]
        action = action_rescale.expand(opt.seeds, 91, 5) # [opt.seeds x91x5]
        x_para_input = torch.cat([x, action], dim=-1)
        y_target = torch.cat([first.unsqueeze(-1), sheer.unsqueeze(-1)], dim=-1).expand(opt.seeds, 91, 2) # [1x91x2]
        net_input = x_para_input
        net_target = y_target
        net_output = g_xy(net_input)
        # loss = abs(maeloss(net_output, y_target) / y_target)
        loss_not_reduce = mseloss_not_reduce(net_output, net_target)
        loss = torch.mean(loss_not_reduce)
        loss.backward()
    optimizer_agent.step()
    lr_scheduler_agent.step(loss)
    # lr_scheduler_agent.step()

    if step == 0:
        para_1 = para[:,1:6]
        x_para_input = torch.cat([x_single[0], para_1], dim=1)
        y_out = g_xy(x_para_input)
        first_list, sheer_list = y_out.chunk(2, dim=1)
        for i in range(1, x.shape[1]):
            x_para_input = torch.cat([x_single[i], para_1], dim=1)
            y_out = g_xy(x_para_input)
            y1, y2 = y_out.chunk(2, dim=1)
            first_list = torch.cat([first_list, y1], dim=1)
            sheer_list = torch.cat([sheer_list, y2], dim=1)
        # loss1 = (abs((torch.cat([ffirst(first_list), fsheer(sheer_list)],dim=1)-y_input)/y_input)).sum()/182
        loss1 = mseloss(torch.cat([first_list, sheer_list], dim=1), y_input)
        print("DNN predict Error for this set of parameter：", loss1.cpu().detach().numpy())

    if step % 1000 == 0:
        loss = loss.cpu().detach().numpy()
        loss_reduced = loss_not_reduce.cpu().detach().numpy().mean(axis=(1,2))
        min_index = np.argmin( loss_reduced )
        action_rescaled = action_rescale[min_index]

        action = para_rescale(action_rescaled).cpu().detach().numpy()
        para = para_rescale(para[:, 1:6]).cpu().detach().numpy()

        minus = abs(((action - para) / para)[0])
        sum_loss = np.sum(np.abs(minus[1:6])) / 5

        print("Correct Parameter:", para)
        print("Predicted Parameter:",action)
        print("Relative Error:", minus, "Sum of Relative Error：", sum_loss)
        print("Step number:",step," time:",time.time()-start_time)
        start_time=time.time()
        print("Optimal_seed_index:{} ,Corresponding loss:{}".format(min_index, loss_reduced[min_index]) )
        print("Step size:", optimizer_agent.state_dict()['param_groups'][0]['lr'], "\n")

        para_list.append(para_rescale(action_rescale).cpu().detach().numpy())
        loss_list.append(loss_reduced)

np.save("para_" + str(opt.order), np.stack(para_list).squeeze(-2))
np.save("loss_agent_" + str(opt.order), np.stack(loss_list))
# '''
