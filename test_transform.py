from transforms import * 
from PIL import Image
import torchvision

def test_jigsaw():
    image = Image.open("cat.jpg").convert("RGB")
    transform = JigSaw((4, 3))
    image_t, order = transform(image)
    print(image_t.shape)
    for  i in range(image_t.shape[0]):
        torchvision.utils.save_image(image_t[i], "cat_jigsaw_{}.png".format(i))

def test_rotate():
    image = Image.open("cat.jpg").convert("RGB")
    rotate = Rotate(4)
    image_t, position = rotate(image)
    transform = torchvision.transforms.ToTensor()
    image_t = transform(image_t)
    torchvision.utils.save_image(image_t, "rotate_test.png")
    print (position)

if __name__ == "__main__":
    test_rotate()