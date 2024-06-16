import random  # Importing the random module for generating random numbers
import numbers  # Importing the numbers module for numeric operations
import numpy as np  # Importing the numpy library for numerical computations
import torch  # Importing the torch library for deep learning
from PIL import Image  # Importing the Image module from the PIL library for image processing
try:
    import accimage  # Trying to import the accimage module for accelerated image loading
except ImportError:
    accimage = None  # If the accimage module is not available, set it to None

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms  # Initializing the Compose class with a list of transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)  # Applying each transform to the image
        return img  # Returning the transformed image

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()  # Randomizing the parameters of each transform

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, norm_value=255):
        self.norm_value = norm_value  # Setting the normalization value for converting image to tensor

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))  # Converting numpy array to tensor
            return img.float().div(self.norm_value)  # Normalizing the tensor

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass

class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # Setting the desired output size for cropping

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))  # Cropping the image at the center

    def randomize_parameters(self):
        pass

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""
    def __call__(self, img):
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)  # Flipping the image horizontally
        return img

    def randomize_parameters(self):
        self.p = random.random()  # Generating a random probability for flipping

class RandomRotate(object):
    def __init__(self):
        self.interpolation = Image.BILINEAR  # Setting the interpolation method for rotation

    def __call__(self, img):
        im_size = img.size
        ret_img = img.rotate(self.rotate_angle, resample=self.interpolation)  # Rotating the image

        return ret_img

    def randomize_parameters(self):
        self.rotate_angle = random.randint(-10, 10)  # Generating a random rotation angle
