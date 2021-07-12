import torch


def get_optimizer(params, conf, model):
    optimizer_conf = conf['optimizer']
    optimizer_choice = optimizer_conf['optimizer_choice']

    if optimizer_choice == 'Adam':
        lr = optimizer_conf['Adam']['lr']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        print(params)
        return torch.optim.Adam(params, lr)
    elif optimizer_choice == 'AdamW':
        lr = optimizer_conf['AdamW']['lr']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.AdamW(params, lr)
    elif optimizer_choice == 'SGD':
        lr = optimizer_conf['SGD']['lr']
        momentum = optimizer_conf['SGD']['momentum']
        weight_decay = optimizer_conf['SGD']['weight_decay']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr, 'momentum:', momentum)
        return torch.optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_choice == 'ASCD':
        lr = optimizer_conf['ASGD']['lr']
        weight_decay = optimizer_conf['ASGD']['weight_decay']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.ASGD(params, lr, weight_decay=weight_decay)
    elif optimizer_choice == 'Rprop':
        lr = optimizer_conf['Rprop']['lr']
        etas = optimizer_conf['Rprop']['etas']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.Rprop(params, lr=lr, etas=etas)
    elif optimizer_choice == 'RMSprop':
        pass
    elif optimizer_choice == 'Adadelta':
        pass
    elif optimizer_choice == 'Adagrad':
        pass
    elif optimizer_choice == 'LBFGS':
        pass
