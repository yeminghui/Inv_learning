import torch.nn as nn
import torch
import argparse
import os
import numpy as np
import itertools
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
# from gpu_select import get_gpu

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="minghui")
parser.add_argument("--para", type=int, default=5)
parser.add_argument("--exp-itr", type=str, default="1")
parser.add_argument("--dataset", type=str, default="30000_originalProcess")
parser.add_argument("--n-epoch", type=int, default=100000, )
parser.add_argument("--hidden-layer", type=int, default=5, )
parser.add_argument("--neuron", type=int, default=128)
parser.add_argument("--data", type=int, default=25000)
parser.add_argument("--testdata", type=int, default=5000)
parser.add_argument("--batch-size", type=int, default=2000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--device", type=str, default="0")
opt = parser.parse_args()
print(opt)

# opt = parser.parse_known_args()[0]

# os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu()
# capeble_gpu = get_gpu()
# os.environ["CUDA_VISIBLE_DEVICES"] = capeble_gpu
# torch.cuda.set_device(capeble_gpu)

class G_xy(nn.Module):
    def __init__(self, input_shape=1 + opt.para, output_shape=2):
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
                for _ in range(opt.hidden_layer )
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


# Tensor = torch.cuda.FloatTensor
Tensor = torch.cuda.FloatTensor


def read_data(data_path):
    para = np.loadtxt("./log_data/" +data_path + "/log_para_metricx.txt")[:opt.data, 1:6]
    first = np.loadtxt("./log_data/" +data_path + "/log_first_stress.txt")[:opt.data, :]
    sheer = np.loadtxt("./log_data/" +data_path + "/log_sheer_stress.txt")[:opt.data, :]
    return para,first,sheer

class MyDataset(Dataset):
    def __init__(self, data_path="./log_data/" +opt.dataset ):
        self.data_path = data_path
        self.x = Tensor(np.loadtxt(data_path + "/log_x.txt"))

        para, first, sheer = read_data('10000_targetRange_logRandom')
        para1, first1, sheer1 = read_data('10000_targetRange_Random')
        self.para = Tensor(np.vstack([para,para1]))
        self.first = Tensor(np.vstack([first,first1]))
        self.sheer = Tensor(np.vstack([sheer,sheer1]))

    def __len__(self):
        return self.para.shape[0]
        # return 2000

    def __getitem__(self, idx):
        return self.x, self.para[idx, :], self.first[idx, :], self.sheer[idx, :]


class MyTestset(Dataset):
    def __init__(self, data_path="./log_data/" + opt.dataset ):
        self.data_path = data_path
        self.x = Tensor(np.loadtxt(data_path + "/log_x.txt"))
        # self.para = Tensor(np.loadtxt(data_path+"/log_para_metricx.txt"))[25000:26900,1:6]
        # self.first = Tensor(np.loadtxt(data_path + "/log_first_stress.txt"))[25000:26900,:]
        # self.sheer = Tensor(np.loadtxt(data_path+"/log_sheer_stress.txt"))[25000:26900,:]

        self.para = Tensor(np.loadtxt(data_path + "/log_para_metricx.txt"))[opt.testdata:, 1:6]
        self.first = Tensor(np.loadtxt(data_path + "/log_first_stress.txt"))[opt.testdata:, :]
        self.sheer = Tensor(np.loadtxt(data_path + "/log_sheer_stress.txt"))[opt.testdata:, :]

    def __len__(self):
        return self.para.shape[0]
        # return 2000

    def __getitem__(self, idx):
        return self.x, self.para[idx, :], self.first[idx, :], self.sheer[idx, :]


# Init all
# '''

if not os.path.isdir("./exp/" + opt.exp_name):
    os.mkdir("./exp/" + opt.exp_name)
if not os.path.isdir("./exp/" + opt.exp_name + "/" + opt.exp_itr + "/"):
    os.mkdir("./exp/" + opt.exp_name + "/" + opt.exp_itr + "/")

argsDict = opt.__dict__
with open("./exp/" + opt.exp_name + "/" + opt.exp_itr + "/" + "setting.txt", 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')
# '''

dataloader = DataLoader(dataset=MyDataset(), batch_size=opt.batch_size, shuffle=True,num_workers=0)
testloader = DataLoader(dataset=MyTestset(), batch_size=len(MyTestset()), shuffle=False)
mseloss = torch.nn.MSELoss()
l1loss = torch.nn.L1Loss()

forword_Dnn = G_xy().cuda()
for m in forword_Dnn.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

weight_p, bias_p = [], []
for name, p in forword_Dnn.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

optimizer_Gxy = torch.optim.Adam(
    [
        {'params': weight_p, 'weight_decay': 1e-5},
        {'params': bias_p, 'weight_decay': 0}
    ], lr=opt.lr
)

Tensor = torch.cuda.FloatTensor
# lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_G, lr_lambda=LambdaLR(opt.n_epoch, opt.decay_epoch).step
# )
# lr_scheduler_Gxy = torch.optim.lr_scheduler.ExponentialLR(optimizer_Gxy, opt.gamma, last_epoch=-1)
lr_scheduler_Gxy = ReduceLROnPlateau(optimizer_Gxy, mode='min', factor=0.1, patience=200,
                                     cooldown= 500, verbose=False,min_lr=1e-6)
# lr_scheduler_Gxy = StepLR(optimizer_Gxy, step_size=opt.gxy_decrease, gamma=0.5)


# checkpoint = torch.load('./gxymodel/fast_G_xy7.798238e-05')
# forword_Dnn.load_state_dict(checkpoint['state_dict'])

# '''
####################### Start Training #######################
if __name__ =='__main__':
    lossmin = 0.01
    start_time = time.time()
    # '''
    for num_epoch in range(opt.n_epoch):
        # 模型训练
        forword_Dnn.train()
        for step, (x, para, first, sheer) in enumerate(dataloader):
            optimizer_Gxy.zero_grad()
            x = x.unsqueeze(-1)
            para = para.unsqueeze(1).expand(para.shape[0],91,para.shape[-1])
            x_para_input = torch.cat([x, para], dim=-1)
            y_target = torch.cat([first.unsqueeze(-1), sheer.unsqueeze(-1)], dim=-1)

            net_input = Variable(x_para_input)
            net_target = Variable(y_target)

            net_output = forword_Dnn(net_input)

            loss_train = mseloss(net_output,net_target)
            loss_train.backward()
            optimizer_Gxy.step()

        lr_scheduler_Gxy.step(loss_train)
        # lr_scheduler_Gxy.step()

        if num_epoch % 100 == 0:
            forword_Dnn.eval()
            print("Testdata MSE Lost:", loss_train.cpu().detach().numpy())
            for step, (x, para, first, sheer) in enumerate(testloader):
                x = x.unsqueeze(-1)
                para = para.unsqueeze(1).expand(para.shape[0], 91, para.shape[-1])
                x_para_input = torch.cat([x, para], dim=-1)
                y_target = torch.cat([first.unsqueeze(-1), sheer.unsqueeze(-1)], dim=-1)

                net_input = Variable(x_para_input)
                net_target = Variable(y_target)

                net_output = forword_Dnn(net_input)

                loss_first = mseloss(net_output[:, :, 0], net_target[:, :, 0])
                loss_sheer = mseloss(net_output[:, :, 1], net_target[:, :, 1])

                # 相对误差补充
                num_size = net_output[:, :, 0].shape[0] * net_output[:, :, 0].shape[1]
                loss_first_rela = torch.sum(torch.sum(torch.abs(10**net_output[:, :, 0] - 10**net_target[:, :, 0])/10**net_target[:, :, 0]))/num_size
                loss_sheer_rela = torch.sum(torch.sum(torch.abs(10**net_output[:, :, 1] - 10**net_target[:, :, 1])/10**net_target[:, :, 1]))/num_size
                loss_rela = (loss_first_rela+loss_sheer_rela).cpu().detach().numpy()/2

                loss_test = (loss_first+loss_sheer).cpu().detach().numpy()/2
                print("epoch:", num_epoch, " time:", time.time() - start_time)
                start_time = time.time()
                print("Testdata MSE Loss:", loss_test, "First:", loss_first.cpu().detach().numpy(),
                      "Sheer:", loss_sheer.cpu().detach().numpy())

                print("Testdata Relative Loss:", loss_rela, "First:", loss_first_rela.cpu().detach().numpy(),
                      "Sheer:", loss_sheer_rela.cpu().detach().numpy())
                print("lr:", optimizer_Gxy.state_dict()['param_groups'][0]['lr'], "\n")

            if num_epoch % 1000 == 0:
                if loss_test < lossmin:
                    lossmin = loss_test
                    # torch.save(forword_Dnn, "./exp/" + opt.exp_name + "/" + opt.exp_itr + "/" + "fast_G_xy"+ str(lossmin))
                    torch.save({'state_dict': forword_Dnn.state_dict()},
                               "./exp/" + opt.exp_name + "/" + opt.exp_itr + "/" + "fast_G_xy"+ str(lossmin))

    # '''