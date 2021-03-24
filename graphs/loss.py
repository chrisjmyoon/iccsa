import torch

"""
Computes the contrastive loss between two feature embeddings
"""
def weighted_ca_loss(z1, z2, yc, y1=None, y2=None, weights=None, margin=1):
    yc = yc.float()
    dist = ((z1 - z2)**2).sum(dim=1).sqrt()
    loss_ca = yc*(dist**2) + (1 - yc)*((margin - dist).max(other=torch.zeros(len(dist)).cuda())**2)

    # Weigh the ca loss based on weights
    if weights is not None:
        wv = torch.zeros(y1.size(0))
        for i in range(y1.size(0)):
            wv[i] = weights[y1[i]][y2[i]] / y1.size(0)
        wv = wv.cuda()
        return torch.matmul(wv, loss_ca)
    
    return loss_ca.mean()
