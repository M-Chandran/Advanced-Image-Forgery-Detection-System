from PIL import Image, ImageDraw
import numpy as np
def create_copy_move_sample():
    """Create a sample image with copy-move forgery for testing"""
    img = Image.new('RGB', (400, 400), 'white')
    draw = ImageDraw.Draw(img)
    for i in range(0, 400, 20):
        draw.line([(0, i), (400, i)], fill=(200, 200, 200), width=1)
        draw.line([(i, 0), (i, 400)], fill=(200, 200, 200), width=1)
    draw.rectangle([50, 50, 150, 150], fill=(0, 255, 0))
    draw.ellipse([80, 80, 120, 120], fill=(255, 0, 0))
    draw.rectangle([250, 50, 350, 150], fill=(255, 165, 0))
    draw.ellipse([270, 80, 330, 120], fill=(0, 0, 255))
    img_array = np.array(img)
    source_region = img_array[50:150, 50:150].copy()
    img_array[200:300, 200:300] = source_region
    source_region2 = img_array[50:150, 250:350].copy()
    img_array[50:150, 200:300] = source_region2
    img = Image.fromarray(img_array)
    img.save('compression_aware_forensics/static/samples/copy_move_test.jpg')
    print("Sample copy-move image created: copy_move_test.jpg")
    highlighted = img_array.copy()
    highlighted[200:203, 200:300] = [255, 0, 0]
    highlighted[297:300, 200:300] = [255, 0, 0]
    highlighted[200:300, 200:203] = [255, 0, 0]
    highlighted[200:300, 297:300] = [255, 0, 0]
    highlighted[50:53, 200:300] = [255, 0, 0]
    highlighted[147:150, 200:300] = [255, 0, 0]
    highlighted[50:150, 200:203] = [255, 0, 0]
    highlighted[50:150, 297:300] = [255, 0, 0]
    highlighted_img = Image.fromarray(highlighted)
    highlighted_img.save('compression_aware_forensics/static/samples/copy_move_test_highlighted.jpg')
    print("Highlighted reference image created: copy_move_test_highlighted.jpg")
if __name__ == "__main__":
    create_copy_move_sample()