from torchvision import transforms
from PIL import Image


class ResizeWithPadding:

    def __init__(self, target_size=224, fill=0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:

        w, h = img.size

        scale = self.target_size / max(w, h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        img = img.resize((new_w, new_h), Image.BILINEAR)

        new_img = Image.new(
            "RGB",
            (self.target_size, self.target_size),
            (self.fill, self.fill, self.fill)
        )

        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2

        new_img.paste(img, (paste_x, paste_y))

        return new_img


class Transform:


    @staticmethod
    def get_basic_transform(img_size=224):
        return transforms.Compose([
            ResizeWithPadding(img_size),
            transforms.ToTensor()
        ])

    @staticmethod
    def get_calibrated_transform(mean, std, img_size=224, train=False):

        transform_list = [
            ResizeWithPadding(img_size)
        ]

        if train:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=0.5)
            )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return transforms.Compose(transform_list)