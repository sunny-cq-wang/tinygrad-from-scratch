from basic_loss_function import loss
from regularization_penalty import reg_pen_L2
from cross_entropy_loss import cross_entropy_loss


def SVM_Loss(F, Y, W, delta, lambdaa):
    data_loss = loss(F, Y, delta)
    reg_loss = reg_pen_L2(W, lambdaa)
    return data_loss + reg_loss


def SoftMax_Loss(F, Y, W, lambdaa):
    data_loss = cross_entropy_loss(F, Y)
    reg_loss = reg_pen_L2(W, lambdaa)
    return data_loss + reg_loss

