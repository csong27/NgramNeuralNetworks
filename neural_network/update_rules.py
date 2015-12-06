from lasagne.updates import *


def get_grad_updates(update_rule, cost, params, lr_rate, momentum_ratio):
    if update_rule == 'adadelta':
        grad_updates = adadelta(loss_or_grads=cost, params=params, learning_rate=lr_rate)
    elif update_rule == 'adagrad':
        grad_updates = adagrad(loss_or_grads=cost, params=params, learning_rate=lr_rate, epsilon=1e-6)
    elif update_rule == 'adam':
        grad_updates = adam(loss_or_grads=cost, params=params, learning_rate=lr_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif update_rule == 'momentum':
        grad_updates = momentum(loss_or_grads=cost, params=params, momentum=momentum_ratio, learning_rate=lr_rate)
    elif update_rule == 'rmsprop':
        grad_updates = rmsprop(loss_or_grads=cost, params=params, learning_rate=lr_rate)
    elif update_rule == 'nesterov':
        grad_updates = nesterov_momentum(loss_or_grads=cost, params=params, learning_rate=lr_rate, momentum=momentum_ratio)
    elif update_rule == 'sgd':
        grad_updates = sgd(loss_or_grads=cost, params=params, learning_rate=lr_rate)
    else:
        raise NotImplementedError("This optimization method is not implemented %s" % update_rule)
    return grad_updates
