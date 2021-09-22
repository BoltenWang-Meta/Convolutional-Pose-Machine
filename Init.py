import torch
from    torch import nn
from    torch import optim
from    torch.utils.data import DataLoader
from    Model.CPM import CPM
from    Data.LSPDataSet import LSPSet

class DataInit(object):
    def __init__(self, root_path, bz):
        super(DataInit, self).__init__()
        trainset = LSPSet(root_path, 'train')
        testset = LSPSet(root_path, 'test')
        self.train_loader = DataLoader(trainset, batch_size=bz,
                                       shuffle=True, num_workers=4)
        self.test_loader = DataLoader(testset, batch_size=bz,
                                      shuffle=False, num_workers=2)
    def Reality(self):
        return self.train_loader, self.test_loader

def ModelInit(num_kp):
    hyper_dict = {'lr': 1e-4, 'bz': 10, 'dv': torch.device('cuda'
                                                           if torch.cuda.is_available() else 'cpu'),
                  'ep': 400
                  }
    model = CPM(num_kp).to(hyper_dict['dv'])
    optimizer = optim.SGD(model.parameters(), lr=hyper_dict['lr'], momentum=0.7)
    lossFn = nn.MSELoss().to(hyper_dict['dv'])
    return hyper_dict, model, optimizer, lossFn

