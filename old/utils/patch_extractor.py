from torch import device
from torchvision import transforms

def image_to_patches(img, patch_size=16):
    img = transforms.ToTensor()(img)
    img_flat = img.view(28, 28)  # [sequence=28, dim=28]
    patches = img_flat.unsqueeze(0).to(device)  # [1, 28, 28]
    img = transforms.Resize((224, 224))(img)
    img_tensor = transforms.ToTensor()(img)  # shape: [3, H, W]
    C, H, W = img_tensor.shape
    patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(C, -1, patch_size, patch_size)  # [3, num_patches, p, p]
    patches = patches.permute(1, 0, 2, 3).flatten(1)  # [num_patches, C*p*p]
    return patches  # [num_patches, flattened_dim]