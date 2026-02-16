import os
import cv2
import numpy as np
from PIL import Image
import random
def create_copy_move_forgery(image_path, output_path):
    """
    Create a copy-move forgery by copying a random region and pasting it elsewhere.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    h, w = image.shape[:2]
    region_h = random.randint(int(h * 0.1), int(h * 0.3))
    region_w = random.randint(int(w * 0.1), int(w * 0.3))
    src_x = random.randint(0, w - region_w)
    src_y = random.randint(0, h - region_h)
    dest_x = random.randint(0, w - region_w)
    dest_y = random.randint(0, h - region_h)
    region = image[src_y:src_y+region_h, src_x:src_x+region_w].copy()
    image[dest_y:dest_y+region_h, dest_x:dest_x+region_w] = region
    cv2.imwrite(output_path, image)
    return output_path
def create_splicing_forgery(image1_path, image2_path, output_path):
    """
    Create a splicing forgery by combining parts from two different images.
    """
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    if image1 is None or image2 is None:
        return None
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    if (h1, w1) != (h2, w2):
        image2 = cv2.resize(image2, (w1, h1))
    split_x = random.randint(int(w1 * 0.3), int(w1 * 0.7))
    forged_image = image1.copy()
    forged_image[:, split_x:] = image2[:, split_x:]
    cv2.imwrite(output_path, forged_image)
    return output_path
def apply_jpeg_compression(image_path, output_path, quality=75):
    """
    Apply JPEG compression to an image.
    """
    image = Image.open(image_path)
    image.save(output_path, 'JPEG', quality=quality)
    return output_path
def generate_dataset(original_images_dir, output_dir, num_samples=100):
    """
    Generate a synthetic dataset with forged and compressed images.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'authentic'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'forged'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'jpeg_compressed'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ai_compressed'), exist_ok=True)
    image_files = [f for f in os.listdir(original_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) < 2:
        print("Need at least 2 images for dataset generation")
        return
    for i in range(num_samples):
        img1 = random.choice(image_files)
        img2 = random.choice([f for f in image_files if f != img1])
        img1_path = os.path.join(original_images_dir, img1)
        img2_path = os.path.join(original_images_dir, img2)
        authentic_path = os.path.join(output_dir, 'authentic', f'authentic_{i}.jpg')
        Image.open(img1_path).save(authentic_path)
        forged_path = os.path.join(output_dir, 'forged', f'forged_cm_{i}.jpg')
        create_copy_move_forgery(img1_path, forged_path)
        splicing_path = os.path.join(output_dir, 'forged', f'forged_sp_{i}.jpg')
        create_splicing_forgery(img1_path, img2_path, splicing_path)
        jpeg_path = os.path.join(output_dir, 'jpeg_compressed', f'jpeg_{i}.jpg')
        apply_jpeg_compression(img1_path, jpeg_path, quality=random.randint(50, 80))
        ai_path = os.path.join(output_dir, 'ai_compressed', f'ai_{i}.jpg')
        Image.open(img1_path).save(ai_path)
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")
    print("Dataset generation completed!")
if __name__ == "__main__":
    original_images_dir = "path/to/your/original/images"
    output_dir = "dataset/generated"
    if os.path.exists(original_images_dir):
        generate_dataset(original_images_dir, output_dir, num_samples=50)
    else:
        print(f"Original images directory '{original_images_dir}' not found.")
        print("Please update the path in the script.")