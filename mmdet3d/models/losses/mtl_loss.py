from LibMTL.weighting import MGDA


class MGDA_S(MGDA):
    r"""Multiple Gradient Descent Algorithm (MGDA).

    This method is proposed in `Multi-Task Learning as Multi-Objective Optimization (NeurIPS 2018) <https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/isl-org/MultiObjectiveOptimization>`_.

    Args:
        mgda_gn ({'none', 'l2', 'loss', 'loss+'}, default='none'): The type of gradient normalization.

    """

    def __init__(self, model, task_num, device, rep_grad=False):
        super(MGDA_S, self).__init__()
        self.model = model
        self.task_num = task_num
        self.device = device
        self.to(device)
        self.rep_grad = rep_grad

    def get_share_params(self):
        return self.model.parameters()
    def zero_grad_share_params(self):
        self.model.zero_grad()
