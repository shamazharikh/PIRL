from torchvision.transforms import ToTensor
import torch
import torchvision.transforms.functional as F
from PIL import Image

class JigSaw(object):
    def __init__(self, n_patches, return_labels=True):
        self.n_patches = n_patches
    
    def __call__(self, img):
        assert isinstance(img, torch.Tensor) 
        image = img
        self.patch_size_1 = image.size[1] // self.n_patches[0] 
        self.patch_size_2 = image.size[0] // self.n_patches[1] 

        image = image.unfold(1, self.patch_size_1, self.patch_size_1).unfold(2, self.patch_size_2, self.patch_size_2)
        image = image.permute([1, 2, 0, 3, 4]).contiguous()
        image = image.view([-1, image.shape[2], image.shape[3], image.shape[4]])
        rand_perm = torch.randperm(image.shape[0])
        image = image[rand_perm]
        return img, image, rand_perm

class Rotate(object):
    def __init__(self, num_positions):
        self.degrees = torch.arange(num_positions) * (360.0 / num_positions) 

    def __call__(self, img):
        image = img
        ind = torch.randint(self.degrees.shape[0], (1,))
        angle = self.degrees[ind]
        if angle == 0. :
            return img, image, ind
        return img, F.rotate(image, angle), ind



