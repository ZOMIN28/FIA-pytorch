import torch
import torch.nn as nn
from utils.FIAloss import FIAloss
import numpy as np
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FIAAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=10, alpha=0.01, prob=0.7, mask_num=100, mu= 0.5):
        # set Parameters
        self.model = model.to(device)
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
        self.prob = prob
        self.mask_num = mask_num
        self.mu = mu
        self.device = device


    def perturb(self, X_nat):
        # get grads
        _,temp = self.model.features_grad(X_nat)
        batch_size = X_nat.shape[0]
        image_size = X_nat.shape[-1]
        grad_sum = torch.zeros((temp.shape)).to(device)
        for i in range(self.mask_num):
            self.model.zero_grad()
            img_temp_i = X_nat.clone()
            # get mask
            mask = torch.tensor(np.random.binomial(1, self.prob, size=(batch_size,3,image_size,image_size))).to(device)
            img_temp_i = img_temp_i * mask
            out,y = self.model.features_grad(img_temp_i)
            # out.backward(torch.ones_like(out))
            # grad_temp = y.grad
            grad_temp = torch.autograd.grad(out, y, grad_outputs=torch.ones_like(out))[0]
            grad_sum += grad_temp
                # avr
        grad_sum = grad_sum / self.mask_num

        g = 0
        eta = 0
        x_cle = X_nat.detach()
        x_adv = X_nat.clone().requires_grad_()
        for epoch in range(self.k):
            x_adv.requires_grad = True
            self.model.zero_grad()
            mid_feature = self.model.layer2_features(x_adv)
            loss = FIAloss(grad_sum, mid_feature)  # FIA loss
            loss.backward()
            g = self.mu*g + x_adv.grad
            x_adv = x_adv - self.alpha*g.sign()
            with torch.no_grad():
                eta = torch.clamp(x_adv - x_cle, min=-self.epsilon, max=self.epsilon)
                X = torch.clamp(x_cle + eta, min=-1, max=1).detach_()
            x_adv = torch.clamp(x_cle+eta, min=-1, max=1).detach_()
        return X
        
