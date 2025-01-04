import torch

CE_CAL = torch.nn.CrossEntropyLoss(ignore_index=0)

def cross_entropy_calc(true_tokens, raw_logits):
    '''
    true_tokens: long64 of seq_len x batch_size
    raw_logits: float32 of seq_len x batch_size x decision_dim
    '''
    # CE expects batch first 
    raw_logits = raw_logits.permute(1, 0, 2)
    true_tokens = true_tokens.permute(1, 0)


    # raw_logits = raw_logits.view(-1, raw_logits.size(-1))  # Reshape to [b_size*seq_len, decision_dim]
    raw_logits = raw_logits.reshape(-1, raw_logits.size(-1))
    # print(raw_logits.shape) --> ([6464, 47])
    true_tokens = true_tokens.view(-1)
    
    return CE_CAL(raw_logits, true_tokens)