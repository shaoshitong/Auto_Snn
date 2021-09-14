import torch


def get_optimizer(params, conf, model):
    optimizer_conf = conf['optimizer']
    optimizer_choice = optimizer_conf['optimizer_choice']

    if optimizer_choice == 'Adam':
        lr = optimizer_conf['Adam']['lr']
        weight_decay=optimizer_conf['Adam']['weight_decay']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.Adam(params, lr)
    elif optimizer_choice == 'AdamW':
        lr = optimizer_conf['AdamW']['lr']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.AdamW(params=params, lr=lr,weight_decay=optimizer_conf['AdamW']['weight_decay'])
    elif optimizer_choice == 'SGD':
        lr = optimizer_conf['SGD']['lr']
        momentum = optimizer_conf['SGD']['momentum']
        weight_decay = optimizer_conf['SGD']['weight_decay']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr, 'momentum:', momentum)
        return torch.optim.SGD(params, lr, momentum=momentum)
    elif optimizer_choice == 'ASCD':
        lr = optimizer_conf['ASGD']['lr']
        weight_decay = optimizer_conf['ASGD']['weight_decay']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.ASGD(params, lr)
    elif optimizer_choice == 'Rprop':
        lr = optimizer_conf['Rprop']['lr']
        etas = optimizer_conf['Rprop']['etas']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.Rprop(params, lr=lr, etas=etas)
    elif optimizer_choice == 'SAM':
        lr = optimizer_conf['SGD']['lr']
        weight_decay=optimizer_conf['SGD']['weight_decay']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        adam=torch.optim.SGD
        optimizer=SAM(params,adam,lr=lr,momentum=0.9,weight_decay=weight_decay)
        return optimizer
    elif optimizer_choice == 'RMSprop':
        pass
    elif optimizer_choice == 'Adadelta':
        pass
    elif optimizer_choice == 'Adagrad':
        pass
    elif optimizer_choice == 'LBFGS':
        pass

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
