from PIL import Image
from PIL import ImageFilter
import numpy as np
img = Image.open('test.jpg')
def jcr():
    x, y = img.size
    img_wb = Image.open('test.jpg').convert("L")
    img_blur = img_wb.filter(ImageFilter.GaussianBlur(-10))
#def

