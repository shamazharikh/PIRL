from transforms import JigSaw, Rotate
from PIL import Image
import torchvision

def test_jigsaw():
    image = Image.open("cat.jpg").convert("RGB")
    image = torchvision.transforms.ToTensor()(image)
    transform = JigSaw((2, 2))
    image, transformed_image = transform(image)
    print(image.size(), transformed_image.size())
    for  i in range(transformed_image.size(0)):
        torchvision.utils.save_image(transformed_image[i], "cat_jigsaw_{}.png".format(i))

def test_rotate():
    image = Image.open("cat.jpg").convert("RGB")
    rotate = Rotate(4)
    image_t, position = rotate(image)
    transform = torchvision.transforms.ToTensor()
    image_t = transform(image_t)
    torchvision.utils.save_image(image_t, "rotate_test.png")
    print (position)

if __name__ == "__main__":
    test_jigsaw()