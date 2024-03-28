import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from core.data_generator import generate_image

def generate_set(file: str, size: int, fonts_text: str, size_text: tuple, blur_text: tuple, directory_backgroud: str, directory_image_output: str, data_type: str, augment: bool):
    fout  = open(f"{directory_image_output}{data_type}_out.txt", 'a')
    with open(f"{file}", 'r') as fp:
        count_item = 1
        while True:
            text = fp.readline().rstrip('\n')
            if not text:
                break
            for i in range(size):
                image_full, img, txt = generate_image(text, directory_backgroud, fonts_text, size_text, blur_text, augment=augment)
                img = np.squeeze(img, axis=-1)
                img = Image.fromarray(np.uint8((img + 1.0) * 127.5))
                img.save(f'{directory_image_output}{data_type}_{count_item}_{i}.jpg')
                fout.write(f'{directory_image_output[2:-1]}{data_type}_{count_item}_{i}.jpg, "{txt}"\n')
                
                image_full.save(f'{directory_image_output}{data_type}_{count_item}_full_{i}.jpg')
                fout.write(f'{directory_image_output[2:-1]}{data_type}_{count_item}_full_{i}.jpg, "{txt}"\n')
            count_item += 1
            
        fout.close()
        fp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process create data images ocr.')
    parser.add_argument('--fonts_text', type=str, default='./fonts/CCCD_MOI/', required=False)
    parser.add_argument('--size_text', type=tuple, default=(18, 28), required=False)
    parser.add_argument('--directory_backgroud', type=str, default='./image_backgroud/test/', required=False)
    parser.add_argument('--file_text', type=str, default='./data/test.txt', required=False)
    parser.add_argument('--directory_image_output', type=str, default='./data/data_output/', required=False)
    parser.add_argument('--size', type=int, default=1, required=False)
    parser.add_argument('--blur_text', type=tuple, default=(1, 2), required=False)
    parser.add_argument('--data_type', type=str, default='train', required=False)
    
    args = parser.parse_args()
    
    generate_set(args.file_text, args.size, args.fonts_text, args.size_text, args.blur_text, args.directory_backgroud, args.directory_image_output, args.data_type, False)

