import torch
from torch.autograd import Variable
# from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim


import sys
sys.path.append('../')

from config import CONFIG

CE_CAL = torch.nn.CrossEntropyLoss(ignore_index=79)

def masked_ce(true_one_hots, mask, logits, use_mask=True):
    # Convert targets from onehots to numbers
    '''
    true_one_hots is b_size x seq_len x decision_dim
    mask is b_size x seq_len x decision_dim
    logits is b_size x seq_len x decision_dim
    '''
    true_tokens = torch.argmax(true_one_hots, dim=-1).long()

    # Mask logits
    if use_mask:
        logits = logits + (mask - 1) * 1e9

    # Collapse dimensions
    logits = logits.view(-1, logits.size(-1))  # Reshape to [64*99, 80]
    true_tokens = true_tokens.view(-1)
    
    return [CE_CAL(logits, true_tokens)]




class PerpCalculator(nn.Module):
    def __init__(self, loss_type = None):
        super(PerpCalculator, self).__init__()

        if loss_type is None:
            self.loss_type = CONFIG.loss_type
        else:
            self.loss_type = loss_type
    '''
    input:
        target (int 65), loss (float32), output (float32)    [50, 99, 80]  seq_len x batch_size x decision_dim

        true_binary: one-hot, with size=time_steps x bsize x DECISION_DIM
        rule_masks: binary tensor, with size=time_steps x bsize x DECISION_DIM
        raw_logits: real tensor, with size=time_steps x bsize x DECISION_DIM
    '''
    def forward(self, true_binary, rule_masks, raw_logits):
        true_binary = true_binary.float()
        if self.loss_type == 'binary':
            exp_pred = torch.exp(raw_logits) * rule_masks

            # Clamp for numerical stability
            exp_pred = torch.clamp(exp_pred, min=1e-9, max=1e+9)
            norm = torch.clamp(norm, min=1e-9, max=1e+9)

            norm = F.torch.sum(exp_pred, 2, keepdim=True)
            prob = F.torch.div(exp_pred, norm)


            return [F.binary_cross_entropy(prob, true_binary) * CONFIG.max_decode_steps]

        if self.loss_type == 'perplexity':
            return my_perp_loss(true_binary, rule_masks, raw_logits)

        if self.loss_type == 'vanilla':
            exp_pred = torch.exp(raw_logits) * rule_masks + 1e-30
            norm = torch.sum(exp_pred, 2, keepdim=True)
            prob = torch.div(exp_pred, norm)

            ll = F.torch.abs(F.torch.sum( true_binary * prob, 2))
            mask = 1 - rule_masks[:, :, -1]
            logll = mask * F.torch.log(ll)

            loss = -torch.sum(logll) / true_binary.size()[1]
            
            return loss

        print('unknown loss type %s' % self.loss_type)
        raise NotImplementedError 



class MyPerpLoss(torch.autograd.Function):
    '''
    Calculates perplexity respecting Syntax Directed constraints
    '''
    @staticmethod
    def forward(ctx, true_binary, rule_masks, input_logits):

        ctx.save_for_backward(true_binary, rule_masks, input_logits)

        # Subtract largest logit --> Input dim 2 is decision dim
        b = F.torch.max(input_logits, 2, keepdim=True)[0]
        raw_logits = input_logits - b # What does subtracting a scalar from a 3d tensor do?


        # Why do we do it again ignoring the masks
        exp_pred = torch.exp(raw_logits) * rule_masks + 1e-30 # Mul w mask, add small constant for stability
        # exp_pred = torch.exp(raw_logits) + 1e-30

        norm = torch.sum(exp_pred, 2, keepdim=True)
        prob = torch.div(exp_pred, norm)
        
        # This leaves the loss at the true point, since it's the only nonzero true binary value
        # Likelyhood of true vale
        ll = F.torch.abs(F.torch.sum( true_binary * prob, 2))
        
        # Subtract the final element of each mask, if it's a final token mask it will yield zero
        # Final tokens don't contribute to loss
        mask = 1 - rule_masks[:, :, -1]

        logll = mask * F.torch.log(ll)
        # Why we loggin twice?
        # logll = F.torch.log(ll)

        # The loss is already notmalized by the batch size
        # The loss is the negative log likelyhood of each batch element
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

        # exp_pred = torch.exp(raw_logits) * rule_masks + 1e-30
        exp_pred = torch.exp(raw_logits) + 1e30

        norm = torch.sum(exp_pred, 2, keepdim=True)
        prob = torch.div(exp_pred, norm)

        grad_matrix1 = grad_matrix2 = None
        
        grad_matrix3 = prob - true_binary
        
        # Removed
        #bp_mask = rule_masks.clone()
        #bp_mask[:, :, -1] = 0

        rescale = 1.0 / true_binary.size()[1]
        # grad_matrix3 = grad_matrix3 * bp_mask * grad_output.data * rescale
        grad_matrix3 = grad_matrix3 * grad_output.data * rescale

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
        #raise NotImplementedError
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
