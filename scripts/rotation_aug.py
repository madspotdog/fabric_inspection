"""Picklable, lossless right-angle training augmentation."""
import random
from PIL import Image


class OrthogonalRotation:
    def __call__(self, image):
        angle = random.randrange(4)
        operations = (None, Image.Transpose.ROTATE_90,
                      Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)
        return image if angle == 0 else image.transpose(operations[angle])
