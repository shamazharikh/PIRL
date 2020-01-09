from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data

class MNISTInstance(datasets.MNIST):
    """MNIST Instance Dataset.
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img = self.train_data[index]
        else:
            img = self.test_data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img, transformed_image, _ = self.transform(img)
        return img, transformed_image, index 
