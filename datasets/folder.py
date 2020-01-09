import torchvision.datasets as datasets

class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, _ = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img, transformed_image, _ = self.transform(img)
        return img, transformed_image, index 


