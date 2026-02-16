import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
def create_ai_like_image_1():
    """Create an image with grid artifacts (common in AI models)"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 32):
        cv2.line(img, (i, 0), (i, 255), (200, 200, 200), 1)
        cv2.line(img, (0, i), (255, i), (200, 200, 200), 1)
    cv2.rectangle(img, (50, 50), (150, 150), (100, 150, 200), -1)
    cv2.rectangle(img, (100, 100), (200, 200), (150, 100, 250), -1)
    return img
def create_ai_like_image_2():
    """Create an image with uniform texture patterns"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:128, :128] = [120, 80, 200]
    img[:128, 128:] = [80, 150, 100]
    img[128:, :128] = [200, 120, 80]
    img[128:, 128:] = [150, 200, 100]
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    return img
def create_ai_like_image_3():
    """Create an image with periodic patterns (GAN artifacts)"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            r = int(128 + 64 * np.sin(i * 0.1) * np.cos(j * 0.1))
            g = int(128 + 64 * np.sin(i * 0.15) * np.sin(j * 0.15))
            b = int(128 + 64 * np.cos(i * 0.12) * np.cos(j * 0.12))
            img[i, j] = [r, g, b]
    return img
def create_ai_like_image_4():
    """Create an image with uniform saturation (AI characteristic)"""
    base_color = np.array([150, 100, 200])
    img = np.full((256, 256, 3), base_color, dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            variation = np.random.normal(0, 2, 3).astype(np.int8)
            img[i, j] = np.clip(base_color + variation, 0, 255)
    return img
def create_ai_like_image_5():
    """Create an image with blocky artifacts (compression-like)"""
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 8):
        for j in range(0, 256, 8):
            block_color = np.random.randint(50, 200, 3)
            img[i:i+8, j:j+8] = block_color
    return img
def save_sample_images():
    """Generate and save sample AI-like images"""
    samples_dir = 'static/samples'
    os.makedirs(samples_dir, exist_ok=True)
    samples = [
        ('ai_generated_grid.jpg', create_ai_like_image_1()),
        ('ai_generated_uniform.jpg', create_ai_like_image_2()),
        ('ai_generated_periodic.jpg', create_ai_like_image_3()),
        ('ai_generated_saturated.jpg', create_ai_like_image_4()),
        ('ai_generated_blocky.jpg', create_ai_like_image_5())
    ]
    for filename, img_array in samples:
        filepath = os.path.join(samples_dir, filename)
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)
        pil_img.save(filepath)
        print(f"Created sample: {filepath}")
    print("All AI sample images created successfully!")
if __name__ == "__main__":
    save_sample_images()