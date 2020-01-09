from torchvision.transforms import ToTensor
import torch
import torchvision.transforms.functional as F
from PIL import Image

class JigSaw(object):
    def __init__(self, n_patches):
        self.n_patches = n_patches
    
    def __call__(self, img):
        assert isinstance(img, torch.Tensor) 
        transformed_image = img
        self.patch_size_1 = transformed_image.size[1] // self.n_patches[0] 
        self.patch_size_2 = transformed_image.size[0] // self.n_patches[1] 

        transformed_image = transformed_image.unfold(1, self.patch_size_1, self.patch_size_1).unfold(2, self.patch_size_2, self.patch_size_2)
        transformed_image = transformed_image.permute([1, 2, 0, 3, 4]).contiguous()
        transformed_image = transformed_image.view([-1, transformed_image.shape[2], transformed_image.shape[3], transformed_image.shape[4]])
        rand_perm = torch.randperm(transformed_image.shape[0])
        transformed_image = transformed_image[rand_perm]
        return img, transformed_image
        
class Rotate(object):
    def __init__(self, num_positions=4, return_image=False):
        self.degrees = torch.arange(num_positions) * (360.0 / num_positions) 
        self.return_image = return_image

    def __call__(self, img):
        image = img
        ind = torch.randint(self.degrees.shape[0], (1,))
        angle = self.degrees[ind]
        if self.return_image:
            return img, F.rotate(image, angle)
        return F.rotate(image, angle)



