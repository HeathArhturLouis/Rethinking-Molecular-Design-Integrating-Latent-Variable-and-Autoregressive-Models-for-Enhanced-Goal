import torch
from torch.autograd import Variable
# from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim


import sys
sys.path.append('../')

from config import CONFIG

class PerpCalculator(nn.Module):
    def __init__(self):
        super(PerpCalculator, self).__init__()
    '''
    input:
        target (int 65), loss (float32), output (float32)    [50, 99, 80]  seq_len x batch_size x decision_dim

        true_binary: one-hot, with size=time_steps x bsize x DECISION_DIM
        rule_masks: binary tensor, with size=time_steps x bsize x DECISION_DIM
        raw_logits: real tensor, with size=time_steps x bsize x DECISION_DIM
    '''
    def forward(self, true_binary, rule_masks, raw_logits):
        true_binary = true_binary.float()
        if CONFIG.loss_type == 'binary':
            exp_pred = torch.exp(raw_logits) * rule_masks

            norm = F.torch.sum(exp_pred, 2, keepdim=True)
            prob = F.torch.div(exp_pred, norm)

            return [F.binary_cross_entropy(prob, true_binary) * CONFIG.max_decode_steps]

        if CONFIG.loss_type == 'perplexity':
            return my_perp_loss(true_binary, rule_masks, raw_logits)

        if CONFIG.loss_type == 'vanilla':
            exp_pred = torch.exp(raw_logits) * rule_masks + 1e-30
            norm = torch.sum(exp_pred, 2, keepdim=True)
            prob = torch.div(exp_pred, norm)

            ll = F.torch.abs(F.torch.sum( true_binary * prob, 2))
            mask = 1 - rule_masks[:, :, -1]
            logll = mask * F.torch.log(ll)

            loss = -torch.sum(logll) / true_binary.size()[1]
            
            return loss
        print('unknown loss type %s' % CONFIG.loss_type)
        raise NotImplementedError 



class MyPerpLoss(torch.autograd.Function):
    '''
    Calculates perplexity respecting Syntax Directed constraints
    '''
    @staticmethod
    def forward(ctx, true_binary, rule_masks, input_logits):
        ctx.save_for_backward(true_binary, rule_masks, input_logits)

        b = F.torch.max(input_logits, 2, keepdim=True)[0]
        raw_logits = input_logits - b
        exp_pred = torch.exp(raw_logits) * rule_masks + 1e-30

        norm = torch.sum(exp_pred, 2, keepdim=True)
        prob = torch.div(exp_pred, norm)

        ll = F.torch.abs(F.torch.sum( true_binary * prob, 2))
        
        mask = 1 - rule_masks[:, :, -1]

        logll = mask * F.torch.log(ll)

        loss = -torch.sum(logll) / true_binary.size()[1]
        
        if input_logits.is_cuda:
            return torch.Tensor([loss]).cuda()
        else:
            return torch.Tensor([loss])

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        true_binary, rule_masks, input_logits  = ctx.saved_tensors

        b = F.torch.max(input_logits, 2, keepdim=True)[0]
        raw_logits = input_logits - b
        exp_pred = torch.exp(raw_logits) * rule_masks + 1e-30

        norm = torch.sum(exp_pred, 2, keepdim=True)
        prob = torch.div(exp_pred, norm)

        grad_matrix1 = grad_matrix2 = None
        
        grad_matrix3 = prob - true_binary
        bp_mask = rule_masks.clone()
        bp_mask[:, :, -1] = 0

        rescale = 1.0 / true_binary.size()[1]
        grad_matrix3 = grad_matrix3 * bp_mask * grad_output.data * rescale        

        return grad_matrix1, grad_matrix2, Variable(grad_matrix3)



## TODO: This wrapper looks like a bit of a hack, I can probably remove it later
def my_perp_loss(true_binary, rule_masks, raw_logits):    
    return MyPerpLoss.apply(true_binary, rule_masks, raw_logits)


## TODO: Doesn't appear to be fully implemented to me
class MyBinaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, true_binary, rule_masks, input_logits):
        ctx.save_for_backward(true_binary, rule_masks, input_logits)

        b = F.torch.max(input_logits, 2, keepdim=True)[0]
        raw_logits = input_logits - b
        exp_pred = torch.exp(raw_logits) * rule_masks

        norm = torch.sum(exp_pred, 2, keepdim=True)
        prob = torch.div(exp_pred, norm)
                
        loss = F.binary_cross_entropy(prob, true_binary)
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        raise NotImplementedError
        true_binary, rule_masks, input_logits  = ctx.saved_tensors

        b = F.torch.max(input_logits, 2, keepdim=True)[0]
        raw_logits = input_logits - b
        exp_pred = torch.exp(raw_logits) * rule_masks
        # exp_pred = torch.exp(input_logits) * rule_masks
        norm = F.torch.sum(exp_pred, 2, keepdim=True)
        prob = F.torch.div(exp_pred, norm)

        grad_matrix1 = grad_matrix2 = None

        grad_matrix3 = prob - true_binary
        
        grad_matrix3 = grad_matrix3 * rule_masks

        return grad_matrix1, grad_matrix2, Variable(grad_matrix3)

def my_binary_loss(true_binary, rule_masks, raw_logits):
    return MyBinaryLoss.apply(true_binary, rule_masks, raw_logits)

if __name__ == '__main__':
    pass
