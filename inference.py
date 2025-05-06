import torch
from utils import extract_patches
from PIL import Image

@torch.no_grad()
def generate_caption(model, image, sp, device, max_len=50):
    model.eval()
    patches = extract_patches(image).unsqueeze(0).to(device)
    tgt_input = torch.tensor([[sp.bos_id()]], dtype=torch.long).to(device)

    for _ in range(max_len):
        output = model(patches, tgt_input)
        next_token = output[:, -1, :].argmax(-1, keepdim=True)
        tgt_input = torch.cat([tgt_input, next_token], dim=1)
        if next_token.item() == sp.eos_id():
            break

    return sp.decode(tgt_input[0].tolist())
