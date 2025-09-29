import torch
import torch.nn.functional as F

tau=0.1
k=5
alpha = torch.rand((14,k)).requires_grad_(True)
alpha = torch.clip(alpha,0.1,1)
W = F.gumbel_softmax(alpha, tau=tau, dim=-1)
fir=-k*torch.log(torch.sum(alpha/((W+1e-8)**tau), dim=-1))
sec= torch.sum(torch.log(alpha/((W+1e-8)**(tau+1))), dim=-1)
loss=torch.sum(fir+sec)
grad = torch.autograd.grad(loss,alpha)
pass