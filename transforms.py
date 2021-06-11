import numbers
import random
import numpy as np
import torchvision
import torch

class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h!=th else 0
        j = random.randint(0, w - tw) if w!=tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        
        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i+h, j:j+w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)





#needed transforms for each video
class Transform:
  """
  class to make the needed transforms for each video.
  params:
    size: an int with the value of h and w (assumed the they are equal)
    mean: the normalization means
    std: the normalizarion stds
  on object call:
    returns the transformed frames tensor.
  """
  def __init__(self, size = 256, means = [0.43216, 0.394666, 0.37645], stds = [0.22803, 0.22145, 0.216989]):
    self.size = size
    self.means = means
    self.stds = stds
    self.resize = torchvision.transforms.Resize((self.size, self.size), interpolation= torchvision.transforms.InterpolationMode.BICUBIC)
    self.normalize = torchvision.transforms.Normalize(self.means, self.stds)

  def __call__(self, video):
    frames = video / 255.0   #shape (T, H, W, C)
    t, h, w, c = frames.shape  #get the shape
    frames = torch.from_numpy(frames)  #transform the frames to torch tensors

    frames_ = torch.zeros([t, c, self.size, self.size])
    frames = frames.permute((0, 3, 1, 2)) #reshape the frames to be resized

    #resize and normalize
    for i in range(t):
      frames_[i]= self.resize(frames[i])
      frames_[i]= self.normalize(frames_[i]) #normalize
    
    #transpose the tensor to the shape (C, T, H, W)
    frames_ = frames_.permute((1, 0, 2, 3))
    return frames_