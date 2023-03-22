from My_resnet50 import resnet50

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import transforms 
from torchsummary import summary
from torch import optim

from tqdm import tqdm
import numpy as np

# Crossentropy 대신 imbalanced class에 적합한 Focal loss 사용
# FocalLoss class
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(777)
if device == 'cuda':
      torch.cuda.manual_seed_all(777)
print(device)

# transforms
trans = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                            transforms.RandomHorizontalFlip(p=1),
                            transforms.ColorJitter(brightness=(0.1,0.6), contrast=1,saturation=0, hue=0.4),
                            transforms.RandomRotation(degrees=66)
                            
])

# ImageFolder로 class folder로 분리되어 있는 dataset load
dataset = torchvision.datasets.ImageFolder(root='D:\\resnet101\\abcd\\project\\data\\20211217_153350\\src\\image_data\\REAL', transform=trans)

# train_size: 0.8, val_size: 0.2
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# split train, val
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=4)

# models.resnet50에서 구현하였던 pretrained model load
# fc layer class 개수만 수정
model = resnet50(pretrained=True).to(device)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1006).to(device)

print(summary(model, (3, 128, 128)))

lr = 0.0001
num_epochs = 100
optimizer = optim.Adam(model.parameters(), lr=lr)

# loss function으로 Retinanet에서 사용되는 Focal loss 사용
loss_function = FocalLoss().to(device)

params = {
    'num_epochs':num_epochs,
    'optimizer':optimizer,
    'loss_function':loss_function,
    'train_dataloader':train_dl,
    'val_dataloader': val_dl,
    'device':device
}

def train(model, params):
    loss_function=params["loss_function"]
    train_dataloader=params["train_dataloader"]
    valid_dataloader=params["val_dataloader"]
    device=params["device"]
    early_stopping = EarlyStopping(patience = 5, verbose = True, path='weights\\resnet50_model.pt')
    
    for epoch in range(0, num_epochs):
        for i, data in tqdm(enumerate(train_dataloader, 0), unit='batch', total=len(train_dataloader)):
            # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 이전 batch에서 계산된 가중치를 초기화
            optimizer.zero_grad() 

            # forward + back propagation 연산
            outputs = model(inputs)
            train_loss = loss_function(outputs, labels)
            train_loss.backward()
            optimizer.step()

    # val accuracy 계산
        total = 0
        correct = 0
        accuracy = []
        for i, data in enumerate(valid_dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 결과값 연산
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss = loss_function(outputs, labels).item()
            accuracy.append(100 * correct/total)
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop: # 조건 만족 시 조기 종료
            break
        # 학습 결과 출력
        print('Epoch: %d/%d, Train loss: %.6f, Val loss: %.6f, Accuracy: %.2f' %(epoch+1, num_epochs, train_loss.item(), val_loss, correct/total))

train(model, params)