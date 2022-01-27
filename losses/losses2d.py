import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        eps=1e-7
        real2 = real.clone()
        fake2 = fake.clone()
        real2[real2==0] = eps
        fake2[fake2==0] = eps
        loss = torch.sqrt(torch.mean(torch.abs(torch.log(real2) - torch.log(fake2)) ** 2))
        return loss

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        loss = torch.mean( torch.abs(10.*real-10.*fake) )
        return loss

class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        loss = torch.mean( torch.abs(torch.log(real)-torch.log(fake)) )
        return loss
    
class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold
    
    def forward(real, fake):
        mask = real>0
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        fake = fake * mask
        diff = torch.abs(real-fake)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()[0]

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        part2 = part2 / (2.*delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss
    
class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        loss = torch.sqrt( torch.mean( torch.abs(10.*real-10.*fake) ** 2 ) )
        return loss

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    
    # L1 norm
    def forward(self, grad_fake, grad_real):
        
        return torch.sum( torch.mean( torch.abs(grad_real-grad_fake) ) )

    
class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
    
    def forward(self, grad_fake, grad_real):
        prod = ( grad_fake[:,:,None,:] @ grad_real[:,:,:,None] ).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt( torch.sum( grad_fake**2, dim=-1 ) )
        real_norm = torch.sqrt( torch.sum( grad_real**2, dim=-1 ) )
        
        return 1 - torch.mean( prod/(fake_norm*real_norm) )

def imgrad(img):
    # img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False).cuda()
    # weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    # weight=weight.expand(-1, 3, -1, -1)
    # if img.is_cuda:
    #     weight = weight.cuda()
    # conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False).cuda()
    # weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    # weight=weight.expand(-1, 3, -1, -1)
    # if img.is_cuda:
    #     weight = weight.cuda()
    # conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

#     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
    
    return grad_y, grad_x

def imgrad_yx(img):
    N,C,H,W = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,H*W), grad_x.view(N,C,H*W)), dim=1)
