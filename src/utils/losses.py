import torch
import torch.nn as nn
import torch.nn.functional as F


class XAQC(nn.Module):
    """
    Based on the MoCo Loss.
    GitHub: https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    Paper: https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=768, K=1024, m=0.999, T=0.07):
        super(XAQC, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        if self.K % batch_size != 0:
            return

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, query, key):
        with torch.no_grad():  # no gradient to keys
            key = nn.functional.normalize(key, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [query, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(key)

        return logits, labels


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class XMRDAngle(nn.Module):
    """
    Based on Relational Knowledge Distillation.
    Paper: https://arxiv.org/pdf/1904.05068.pdf
    Code: https://github.com/lenscloth/RKD/blob/master/metric/loss.py
    """

    def forward(self, photo, sketch_q, sketch_k, teacher, student, modality):
        # N x C
        # N x N x C
        with torch.no_grad():
            teacher = torch.cat([teacher.forward_features(photo, sketch_q)[0], teacher.forward_features(photo, sketch_k)[0]])
        if modality == 'photo':
            student = torch.cat([student.forward_features(photo), student.forward_features(photo)])
        else:
            student = torch.cat([student.forward_features(sketch_q), student.forward_features(sketch_k)])

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class XMRDDistance(nn.Module):
    """
    Based on Relational Knowledge Distillation.
    Paper: https://arxiv.org/pdf/1904.05068.pdf
    Code: https://github.com/lenscloth/RKD/blob/master/metric/loss.py
    """

    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss
