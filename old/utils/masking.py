
def generate_square_subsequent_mask(sz):
    # Creates an upper-triangular matrix of -inf, with zeros on the diagonal
    import torch
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)