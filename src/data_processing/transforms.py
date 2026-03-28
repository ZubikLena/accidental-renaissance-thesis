from torchvision import transforms
from PIL import Image


class ResizeWithPadding:
    """
    Resize image while preserving aspect ratio,
    then pad to a square (no distortion).
    """

    def __init__(self, target_size=224, fill=0):
        self.target_size = target_size
        self.fill = fill  # padding color (0 = black)

    def __call__(self, img: Image.Image) -> Image.Image:

        # Original size
        w, h = img.size

        # Compute scaling factor
        scale = self.target_size / max(w, h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image (keep aspect ratio)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Create new square image
        new_img = Image.new(
            "RGB",
            (self.target_size, self.target_size),
            (self.fill, self.fill, self.fill)
        )

        # Center the image
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2

        new_img.paste(img, (paste_x, paste_y))

        return new_img


class Transform:
    """
    Centralized transformation manager for the project.
    Ensures consistency across training and evaluation.
    """

    @staticmethod
    def get_basic_transform(img_size=224):
        """
        Used for computing dataset statistics (mean/std).
        NO normalization here.
        """
        return transforms.Compose([
            ResizeWithPadding(img_size),
            transforms.ToTensor()
        ])

    @staticmethod
    def get_calibrated_transform(mean, std, img_size=224, train=False):
        """
        Final transformation pipeline with normalization.
        
        Args:
            mean (list): dataset mean
            std (list): dataset std
            train (bool): apply augmentation if True
        """

        transform_list = [
            ResizeWithPadding(img_size)
        ]

        # Data augmentation (only for training)
        if train:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=0.5)
            )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return transforms.Compose(transform_list)