import string
import os, random
from glob import glob
from random import randint, choice, uniform
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from imgaug import augmenters as iaa

from .image import ImageUtil

default_vocabulary = list(string.ascii_lowercase) + list(string.digits) + [' ', '-', '.', ':', '?', '!', '<', '>', '#', '@', '(', ')', '$', '%', '&']

seq = iaa.SomeOf((0, 2), [
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
    iaa.Invert(1.0),
    iaa.MotionBlur(k=10)
])

def get_font(file_font, size_text):
    fonts = f'{file_font}{random.choice(os.listdir(file_font))}'
    font = ImageFont.truetype(fonts, size=randint(size_text[0], size_text[1]))
    return font

def rand_pad():
    return randint(2, 4), randint(2, 4), randint(2, 4), randint(2, 4)

# def random_string(length: Optional[int] = None):
#     if length is None:
#         length = randint(4, 20)

#     if randint(0, 1) == 0:
#         random_file = choice(list(glob('synthetic/texts/*.txt')))
#         with open(random_file, 'r') as f:
#             random_txt = f.readlines()
#         random_txt = choice(random_txt)
#         end = len(random_txt) - length
#         if end > 0:
#             start = randint(0, end)
#             random_txt = random_txt[start:start+length].strip()
#             if len(random_txt) > 1:
#                 return random_txt

#     letters = list(string.ascii_uppercase) + default_vocabulary
#     return (''.join(choice(letters) for _ in range(length))).strip()

def merging_backgroud(img):
    new_image = Image.new('RGB',(2*img.size[0], img.size[1]), (250,250,250))
    new_image.paste(img,(0,0))
    new_image.paste(img,(img.size[0],0))
    return new_image

def random_background(dir_bg, height, width):
    background_image = choice(list(glob(f'{dir_bg}*.jpg'))) 
    original = Image.open(background_image)
    background = original.copy()
    
    L = original.convert('L')
    original = Image.merge('RGB', (L, L, L))
    while True:
        if (original.size[0] - height) < 0:
            original = merging_backgroud(original)
            background = merging_backgroud(background)
        else: 
            break
    left = randint(0, original.size[0] - height)
    top = randint(0, original.size[1] - width)
    right = left + height
    bottom = top + width
    return original.crop((left, top, right, bottom)), background

def generate_image(text: str, dir_bg: str, file_font: str, size_text: tuple, blur_text:tuple, augment: bool) -> Tuple[np.array, str]:
    font = get_font(file_font, size_text)
    txt_width, txt_height = font.getsize(text)
    left_pad, right_pad, top_pad, bottom_pad = rand_pad()
    height = left_pad + txt_width + right_pad
    width = top_pad + txt_height + bottom_pad
    image, background = random_background(dir_bg, height, width)
    
    fgr_img = Image.new('RGBA', (height, width), color=(0,0,0,0))
    
    mask = Image.new('L', background.size, color=255)
    canvas = ImageDraw.Draw(mask)
    canvas.text((left_pad, top_pad), text, fill=0, font=font)
    
    mask = mask.filter(ImageFilter.GaussianBlur(uniform(blur_text[0], blur_text[1])))
    # mask = mask.transform(mask.size, Image.AFFINE, (1, 0.4, 0, 0, 1, 0))

    fgr_img.paste(background, (0, 0), mask)

    image = np.array(fgr_img)
    if augment:
        image = seq.augment_image(image)
    image_util = ImageUtil(width, height)
    image = image_util.preprocess(image)
    
    image_full = fgr_img.convert('RGB')
    return image_full, image, text
