from __future__ import print_function
from PIL import Image
from os.path import join
import os
import torchvision.datasets as datasets
import torch.utils.data as data

class OmniglotInstance(datasets.Omniglot):
    """Omniglot Instance Dataset.
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self._flat_character_images[index]
        image_path = join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image, transformed_image = self.transform(image)
        return image, transformed_image, index 