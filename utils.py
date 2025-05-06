import torchvision.transforms as T

def extract_patches(image, patch_size=16):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    image = transform(image)
    C, H, W = image.shape
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    patches = patches.view(-1, C * patch_size * patch_size)
    return patches
