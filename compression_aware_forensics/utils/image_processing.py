import cv2
import numpy as np
import base64
from PIL import Image
from scipy import ndimage
import hashlib
import pickle
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
PREPROCESS_CACHE = {}
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)
@lru_cache(maxsize=50)
def preprocess_image_cached(image_path):
    """
    Cached version of preprocess_image for better performance.
    """
    try:
        import os
        stat = os.stat(image_path)
        cache_key = f"{image_path}_{stat.st_mtime}"
    except:
        cache_key = image_path
    if cache_key in PREPROCESS_CACHE:
        return PREPROCESS_CACHE[cache_key]
    result = preprocess_image(image_path)
    PREPROCESS_CACHE[cache_key] = result
    if len(PREPROCESS_CACHE) > 50:
        oldest_key = next(iter(PREPROCESS_CACHE))
        del PREPROCESS_CACHE[oldest_key]
    return result
def preprocess_image(image_path):
    """
    Preprocess the uploaded image for model input.
    """
    if not TORCH_AVAILABLE:
        return None
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed_image = transform(image).unsqueeze(0)
    return processed_image
def calculate_psnr(original, compressed):
    """
    Calculate PSNR between original and compressed image.
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
def calculate_ssim(original, compressed):
    """
    Calculate SSIM between original and compressed image.
    """
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(original, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(compressed, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(original**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(compressed**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(original * compressed, -1, window)[5:-5, 5:-5] - mu1_mu2
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    return ssim_map.mean()
def detect_copy_move(image_path, block_size=16, threshold=0.9, max_blocks=1000, fast_mode=False):
    """
    Enhanced copy-move forgery detection with improved region filling and highlighting.
    Returns a mask highlighting the complete copied regions.
    """
    if fast_mode:
        return None, []
    image = cv2.imread(image_path)
    if image is None:
        return None, []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    step_size = max(1, int(np.sqrt((height * width) / max_blocks)))
    step_size = max(step_size, block_size // 4)
    blocks = []
    positions = []
    for y in range(0, height - block_size + 1, step_size):
        for x in range(0, width - block_size + 1, step_size):
            block = gray[y:y+block_size, x:x+block_size]
            blocks.append(block)
            positions.append((x, y, x+block_size, y+block_size))
    if len(blocks) == 0:
        return None, []
    blocks = np.array(blocks)
    mask = np.zeros((height, width), dtype=np.uint8)
    copied_regions = []
    min_distance = block_size * 2
    for i, (block1, pos1) in enumerate(zip(blocks, positions)):
        x1, y1 = pos1[0], pos1[1]
        for j, (block2, pos2) in enumerate(zip(blocks[i+1:], positions[i+1:])):
            x2, y2 = pos2[0], pos2[1]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < min_distance:
                continue
            corr = cv2.matchTemplate(block1, block2, cv2.TM_CCOEFF_NORMED)[0][0]
            if corr > threshold:
                x1_end, y1_end = pos1[2], pos1[3]
                x2_end, y2_end = pos2[2], pos2[3]
                mask[y1:y1_end, x1:x1_end] = 255
                mask[y2:y2_end, x2:x2_end] = 255
                copied_regions.append((x1, y1, x1_end, y1_end))
                copied_regions.append((x2, y2, x2_end, y2_end))
                if len(copied_regions) > 50:
                    break
        if len(copied_regions) > 50:
            break
    if np.sum(mask) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > block_size * block_size:
                cv2.drawContours(mask, [contour], 0, 255, -1)
    copied_regions = merge_overlapping_regions(copied_regions)
    return mask, copied_regions
def merge_overlapping_regions(regions):
    """
    Merge overlapping bounding boxes.
    """
    if not regions:
        return []
    regions = sorted(regions, key=lambda x: x[0])
    merged = [regions[0]]
    for current in regions[1:]:
        last = merged[-1]
        if current[0] < last[2] and current[1] < last[3]:
            merged[-1] = (min(last[0], current[0]), min(last[1], current[1]),
                          max(last[2], current[2]), max(last[3], current[3]))
        else:
            merged.append(current)
    return merged
def extract_copied_regions_to_base64(image_path, copied_regions):
    """
    Extract copied regions and return as base64 encoded images.
    """
    image = cv2.imread(image_path)
    if image is None:
        return []
    extracted_images = []
    for i, (x1, y1, x2, y2) in enumerate(copied_regions):
        region = image[y1:y2, x1:x2]
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(region_rgb)
        from io import BytesIO
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        extracted_images.append(f"data:image/png;base64,{img_base64}")
    return extracted_images
def create_highlighted_image(image_path, mask):
    """
    Create a highlighted version of the image with copied regions marked.
    """
    image = cv2.imread(image_path)
    if image is None or mask is None:
        return None
    image_height, image_width = image.shape[:2]
    mask_height, mask_width = mask.shape[:2]
    if (mask_height, mask_width) != (image_height, image_width):
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy()
    overlay[mask > 0] = [0, 0, 255]
    highlighted = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(highlighted_rgb)
    from io import BytesIO
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"
def standardize_image(image_path, target_size=(256, 256)):
    """
    STEP 1: Image Standardization
    Resize to fixed resolution, convert RGB to YCrCb, extract Y channel
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]
    return y_channel.astype(np.float32)
def extract_noise_residual(y_channel, kernel_size=5, sigma=1.0):
    """
    STEP 2.1: Noise Residual Extraction (Most Important)
    Compute residual = Y - GaussianBlur(Y)
    """
    blurred = cv2.GaussianBlur(y_channel, (kernel_size, kernel_size), sigma)
    residual = y_channel - blurred
    residual = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
    return residual.astype(np.uint8)
def apply_high_pass_filter(residual):
    """
    STEP 2.2: High-Pass Filtering (Optional but Recommended)
    Emphasize high-frequency inconsistencies and tampering boundaries
    """
    kernel = np.array([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]], dtype=np.float32)
    high_pass = cv2.filter2D(residual.astype(np.float32), -1, kernel)
    high_pass = np.abs(high_pass)
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)
    return high_pass.astype(np.uint8)
def extract_patches(processed_image, patch_size=64, stride=32):
    """
    STEP 3: Patch Extraction (Critical)
    Divide into 64×64 overlapping patches for localized analysis
    """
    height, width = processed_image.shape
    patches = []
    positions = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = processed_image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((x, y, x+patch_size, y+patch_size))
    return np.array(patches), positions
def create_forgery_heatmap(patch_predictions, positions, image_shape, threshold=0.5):
    """
    STEP 5: Region-Level Localization
    Combine patch predictions to generate forgery heatmap
    """
    height, width = image_shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    for (pred, prob), (x1, y1, x2, y2) in zip(patch_predictions, positions):
        if prob > threshold:
            heatmap[y1:y2, x1:x2] += prob
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    return heatmap.astype(np.uint8)
def post_process_mask(mask, min_area=100):
    """
    STEP 6: Post-Processing
    Apply morphological operations to remove false positives
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask_region = (labels == i)
            if mask_region.shape == cleaned_mask.shape:
                cleaned_mask[mask_region] = 255
            else:
                cleaned_mask[labels == i] = 255
    return cleaned_mask
def evaluate_forgery_detection(true_mask, pred_mask):
    """
    STEP 7: Evaluation (Mandatory)
    Calculate Accuracy, Precision, Recall, F1-Score
    """
    true_flat = true_mask.flatten() / 255.0
    pred_flat = pred_mask.flatten() / 255.0
    true_binary = (true_flat > 0.5).astype(int)
    pred_binary = (pred_flat > 0.5).astype(int)
    tp = np.sum((true_binary == 1) & (pred_binary == 1))
    tn = np.sum((true_binary == 0) & (pred_binary == 0))
    fp = np.sum((true_binary == 0) & (pred_binary == 1))
    fn = np.sum((true_binary == 1) & (pred_binary == 0))
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
def detect_image_forgery(image_path, fast_mode=False):
    """
    Optimized image forgery detection with simplified analysis pipeline
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to load image"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if fast_mode:
            copy_move_result = detect_copy_move_advanced(image_path)
            compression_result = detect_compression_artifacts(gray)
            forgery_score = 0.0
            if copy_move_result['copy_move_detected']:
                forgery_score += 0.7
            if compression_result['suspicious']:
                forgery_score += 0.3
            confidence_level = 'High' if forgery_score > 0.5 else 'Low'
            forgery_type = 'Copy-Move Forgery' if copy_move_result['copy_move_detected'] else 'Unknown'
            return {
                'is_forged': forgery_score > 0.5,
                'forgery_score': forgery_score,
                'confidence': confidence_level,
                'forgery_type': forgery_type,
                'detailed_analysis': {'overall_score': forgery_score},
                'suspicious_segments': {'segments': [], 'mask': None},
                'forgery_highlighted_image': None,
                'detection_methods': {
                    'copy_move_analysis': copy_move_result,
                    'compression_artifacts': compression_result
                }
            }, None
        else:
            def run_copy_move():
                return detect_copy_move_advanced(image_path)
            def run_compression():
                return detect_compression_artifacts(gray)
            def run_noise():
                return analyze_multi_scale_noise(gray)
            def run_edges():
                return check_edge_consistency(image)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    'copy_move': executor.submit(run_copy_move),
                    'compression': executor.submit(run_compression),
                    'noise': executor.submit(run_noise),
                    'edges': executor.submit(run_edges)
                }
                results = {key: future.result() for key, future in futures.items()}
            forgery_score = calculate_optimized_forgery_score(results)
            confidence_level = assess_optimized_confidence(forgery_score)
            forgery_type = classify_optimized_forgery_type(results)
            suspicious_segments = detect_optimized_segments(image, results)
            highlighted_image = create_forgery_highlighted_image(image_path, suspicious_segments)
            return {
                'is_forged': forgery_score['overall_score'] > 0.6,
                'forgery_score': forgery_score['overall_score'],
                'confidence': confidence_level,
                'forgery_type': forgery_type,
                'detailed_analysis': forgery_score,
                'suspicious_segments': suspicious_segments,
                'forgery_highlighted_image': highlighted_image,
                'detection_methods': results
            }, None
    except Exception as e:
        return None, str(e)
def analyze_multi_scale_noise(gray_image):
    """Analyze noise patterns at multiple scales like a human expert"""
    scales = [1, 2, 4, 8]
    noise_scores = []
    for scale in scales:
        if scale > 1:
            scaled = cv2.resize(gray_image, (gray_image.shape[1]//scale, gray_image.shape[0]//scale))
            scaled = cv2.resize(scaled, (gray_image.shape[1], gray_image.shape[0]))
        else:
            scaled = gray_image
        noise = extract_noise_residual(scaled)
        noise_std = np.std(noise)
        noise_scores.append(noise_std)
    noise_variation = np.std(noise_scores) / np.mean(noise_scores)
    return {
        'noise_consistency': 1.0 - min(1.0, noise_variation * 2),
        'scale_variations': noise_scores,
        'suspicious': noise_variation > 0.3
    }
def check_edge_consistency(image):
    """Check edge continuity and consistency"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(edges, kernel, iterations=1)
    edge_discontinuities = np.sum(np.abs(dilated.astype(float) - eroded.astype(float)))
    return {
        'edge_continuity': 1.0 - min(1.0, edge_discontinuities / (gray.size * 0.01)),
        'discontinuities': edge_discontinuities,
        'suspicious': edge_discontinuities > gray.size * 0.005
    }
def analyze_lighting_consistency(image):
    """Analyze lighting and shadow consistency"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2]
    brightness_std = np.std(brightness)
    grad_x = cv2.Sobel(brightness, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(brightness, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    lighting_gradient_score = np.mean(gradient_magnitude) / brightness_std
    return {
        'lighting_consistency': 1.0 - min(1.0, lighting_gradient_score * 0.1),
        'brightness_variation': brightness_std,
        'gradient_score': lighting_gradient_score,
        'suspicious': lighting_gradient_score > 5.0
    }
def analyze_color_correlations(image):
    """Analyze color channel correlations for tampering detection"""
    channels = cv2.split(image)
    correlations = []
    for i in range(3):
        for j in range(i+1, 3):
            corr = np.corrcoef(channels[i].flatten(), channels[j].flatten())[0, 1]
            correlations.append(abs(corr))
    avg_correlation = np.mean(correlations)
    correlation_consistency = np.std(correlations)
    return {
        'correlation_consistency': 1.0 - min(1.0, correlation_consistency * 2),
        'avg_correlation': avg_correlation,
        'correlation_variation': correlation_consistency,
        'suspicious': correlation_consistency > 0.2
    }
def detect_compression_artifacts(gray_image):
    """Detect JPEG and other compression artifacts"""
    height, width = gray_image.shape
    artifacts_score = 0
    for i in range(0, height-8, 8):
        for j in range(0, width-8, 8):
            block = gray_image[i:i+8, j:j+8].astype(float)
            dct_block = cv2.dct(block)
            high_freq_energy = np.sum(np.abs(dct_block[5:, 5:]))
            low_freq_energy = np.sum(np.abs(dct_block[:3, :3]))
            if low_freq_energy > 0:
                artifact_ratio = high_freq_energy / low_freq_energy
                artifacts_score += min(1.0, artifact_ratio * 0.01)
    artifacts_score /= ((height//8) * (width//8))
    return {
        'compression_artifacts': 1.0 - min(1.0, artifacts_score),
        'artifact_score': artifacts_score,
        'suspicious': artifacts_score > 0.3
    }
def detect_statistical_anomalies(gray_image):
    """Detect statistical anomalies in pixel distributions"""
    flattened = gray_image.flatten()
    first_digits = []
    for pixel in flattened[flattened > 0]:
        first_digit = int(str(pixel)[0])
        first_digits.append(first_digit)
    if first_digits:
        digit_counts = np.bincount(first_digits, minlength=10)[1:]
        expected = np.log10(1 + 1/np.arange(1, 10)) * len(first_digits)
        chi_square = np.sum((digit_counts - expected)**2 / expected) if np.all(expected > 0) else 0
        benford_score = min(1.0, chi_square / 100)
    else:
        benford_score = 0
    entropy_map = np.zeros_like(gray_image, dtype=float)
    for i in range(1, gray_image.shape[0]-1):
        for j in range(1, gray_image.shape[1]-1):
            patch = gray_image[i-1:i+2, j-1:j+2]
            hist = np.histogram(patch, bins=256, range=(0, 256))[0]
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist))
            entropy_map[i, j] = entropy
    entropy_variation = np.std(entropy_map) / np.mean(entropy_map)
    return {
        'statistical_consistency': 1.0 - min(1.0, (benford_score + entropy_variation) / 2),
        'benford_score': benford_score,
        'entropy_variation': entropy_variation,
        'suspicious': (benford_score > 0.5) or (entropy_variation > 0.8)
    }
def detect_splicing_artifacts(image):
    """Detect splicing artifacts using advanced techniques"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    channels = cv2.split(image)
    splicing_score = 0
    for i in range(3):
        high_pass = apply_high_pass_filter(channels[i].astype(float))
        boundary_energy = np.sum(np.abs(high_pass))
        splicing_score += boundary_energy / (channels[i].size * np.std(channels[i]))
    splicing_score /= 3
    return {
        'splicing_artifacts': 1.0 - min(1.0, splicing_score * 0.1),
        'boundary_energy': splicing_score,
        'suspicious': splicing_score > 2.0
    }
def detect_ai_compression_artifacts(image_path):
    """
    Detect AI-generated compression artifacts in images.
    Returns True if AI compression artifacts are detected.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        smoothness_score = detect_smoothness_anomalies(gray)
        statistical_score = detect_ai_statistical_anomalies(gray)
        noise_score = detect_ai_noise_patterns(gray)
        color_score = detect_ai_color_artifacts(image)
        frequency_score = detect_frequency_anomalies(gray)
        weights = {
            'smoothness': 0.25,
            'statistics': 0.25,
            'noise': 0.20,
            'color': 0.15,
            'frequency': 0.15
        }
        overall_score = (
            smoothness_score * weights['smoothness'] +
            statistical_score * weights['statistics'] +
            noise_score * weights['noise'] +
            color_score * weights['color'] +
            frequency_score * weights['frequency']
        )
        ai_threshold = 0.95
        return overall_score > ai_threshold
    except Exception as e:
        print(f"Error in AI detection: {e}")
        return False
def enhanced_forgery_detection(image_path, ai_compression_detected=False):
    """
    Enhanced forgery detection that adapts based on compression type.
    """
    try:
        result, error = detect_image_forgery(image_path)
        if error:
            return {'forgery_detected': 'Error', 'error': error}
        if result:
            forgery_detected = 'Forged' if result['is_forged'] else 'Authentic'
            if ai_compression_detected:
                if result['forgery_score'] > 0.6:
                    forgery_detected = 'Forged (AI-Compressed)'
            return {
                'forgery_detected': forgery_detected,
                'forgery_score': result['forgery_score'],
                'confidence': result['confidence'],
                'forgery_type': result['forgery_type'],
                'ai_compression_detected': ai_compression_detected
            }
        else:
            return {'forgery_detected': 'Error', 'error': 'Detection failed'}
    except Exception as e:
        return {'forgery_detected': 'Error', 'error': str(e)}
def detect_copy_move_advanced(image_path):
    """Advanced copy-move detection"""
    mask, copied_regions = detect_copy_move(image_path)
    if mask is not None:
        copy_move_area = np.sum(mask > 0) / mask.size
    else:
        copy_move_area = 0
    return {
        'copy_move_detected': len(copied_regions) > 0 if copied_regions else False,
        'copy_move_area': copy_move_area,
        'regions_count': len(copied_regions) if copied_regions else 0,
        'suspicious': copy_move_area > 0.15
    }
def calculate_fast_forgery_score(analyses):
    """Calculate simplified forgery score for fast mode"""
    weights = {
        'compression': 0.4,
        'statistics': 0.4,
        'copy_move': 0.2
    }
    overall_score = 0
    suspicious_indicators = 0
    for method, analysis in analyses.items():
        if 'consistency' in analysis:
            score = analysis['consistency']
        elif 'artifacts' in analysis:
            score = analysis['artifacts']
        else:
            score = 0.5
        overall_score += score * weights[method]
        if analysis.get('suspicious', False):
            suspicious_indicators += 1
    if suspicious_indicators >= 2:
        overall_score = min(1.0, overall_score * 1.2)
    return overall_score
def calculate_comprehensive_forgery_score(analyses):
    """Calculate overall forgery score from all analyses"""
    weights = {
        'noise': 0.15,
        'edges': 0.15,
        'lighting': 0.15,
        'color': 0.10,
        'compression': 0.15,
        'statistics': 0.15,
        'splicing': 0.10,
        'copy_move': 0.05
    }
    overall_score = 0
    suspicious_indicators = 0
    for method, analysis in analyses.items():
        if 'consistency' in analysis:
            score = analysis['consistency']
        elif 'artifacts' in analysis:
            score = analysis['artifacts']
        else:
            score = 0.5
        overall_score += score * weights[method]
        if analysis.get('suspicious', False):
            suspicious_indicators += 1
    if suspicious_indicators >= 3:
        overall_score = min(1.0, overall_score * 1.3)
    elif suspicious_indicators >= 5:
        overall_score = min(1.0, overall_score * 1.5)
    return {
        'overall_score': overall_score,
        'suspicious_indicators': suspicious_indicators,
        'method_scores': {method: analysis.get('consistency', analysis.get('artifacts', 0.5))
                         for method, analysis in analyses.items()},
        'confidence_factors': suspicious_indicators
    }
def assess_detection_confidence(forgery_score):
    """Assess confidence level of detection"""
    score = forgery_score['overall_score']
    indicators = forgery_score['suspicious_indicators']
    if score > 0.9 and indicators >= 5:
        return 'Very High'
    elif score > 0.8 and indicators >= 4:
        return 'High'
    elif score > 0.7 and indicators >= 3:
        return 'Moderate'
    elif score > 0.6 and indicators >= 2:
        return 'Low'
    else:
        return 'Very Low'
def classify_forgery_type(forgery_score):
    """Classify the type of forgery detected"""
    method_scores = forgery_score['method_scores']
    max_method = max(method_scores.items(), key=lambda x: x[1])
    type_mapping = {
        'copy_move': 'Copy-Move Forgery',
        'splicing': 'Image Splicing',
        'compression': 'Re-compression Forgery',
        'noise': 'Noise Inconsistency',
        'edges': 'Edge Discontinuity',
        'lighting': 'Lighting Manipulation',
        'color': 'Color Channel Tampering',
        'statistics': 'Statistical Anomaly'
    }
    if forgery_score['overall_score'] > 0.7:
        return type_mapping.get(max_method[0], 'Unknown Forgery Type')
    else:
        return 'No Forgery Detected'
def detect_suspicious_segments(image, analyses):
    """
    Detect specific suspicious segments in the image based on analysis results
    """
    height, width = image.shape[:2]
    suspicious_mask = np.zeros((height, width), dtype=np.uint8)
    segments = []
    if analyses["edges"]["suspicious"]:
        edge_mask = detect_edge_anomalies(image)
        if edge_mask is not None:
            suspicious_mask = cv2.bitwise_or(suspicious_mask, edge_mask)
            segments.extend(extract_regions_from_mask(edge_mask, "edge_anomaly"))
    if analyses["lighting"]["suspicious"]:
        lighting_mask = detect_lighting_anomalies(image)
        if lighting_mask is not None:
            suspicious_mask = cv2.bitwise_or(suspicious_mask, lighting_mask)
            segments.extend(extract_regions_from_mask(lighting_mask, "lighting_anomaly"))
    if analyses["noise"]["suspicious"]:
        noise_mask = detect_noise_anomalies(image)
        if noise_mask is not None:
            suspicious_mask = cv2.bitwise_or(suspicious_mask, noise_mask)
            segments.extend(extract_regions_from_mask(noise_mask, "noise_anomaly"))
    if analyses["color"]["suspicious"]:
        color_mask = detect_color_anomalies(image)
        if color_mask is not None:
            suspicious_mask = cv2.bitwise_or(suspicious_mask, color_mask)
            segments.extend(extract_regions_from_mask(color_mask, "color_anomaly"))
    if analyses["compression"]["suspicious"]:
        compression_mask = detect_compression_anomaly_regions(image)
        if compression_mask is not None:
            suspicious_mask = cv2.bitwise_or(suspicious_mask, compression_mask)
            segments.extend(extract_regions_from_mask(compression_mask, "compression_anomaly"))
    if analyses["statistics"]["suspicious"]:
        statistical_mask = detect_statistical_anomaly_regions(image)
        if statistical_mask is not None:
            suspicious_mask = cv2.bitwise_or(suspicious_mask, statistical_mask)
            segments.extend(extract_regions_from_mask(statistical_mask, "statistical_anomaly"))
    if analyses["splicing"]["suspicious"]:
        splicing_mask = detect_splicing_boundaries(image)
        if splicing_mask is not None:
            suspicious_mask = cv2.bitwise_or(suspicious_mask, splicing_mask)
            segments.extend(extract_regions_from_mask(splicing_mask, "splicing_boundary"))
    segments = merge_overlapping_regions(segments)
    segments = [seg for seg in segments if (seg[2] - seg[0]) * (seg[3] - seg[1]) > 100]
    return {
        "segments": segments,
        "mask": suspicious_mask,
        "segment_count": len(segments),
        "total_suspicious_area": np.sum(suspicious_mask > 0) / suspicious_mask.size
    }
def detect_edge_anomalies(image):
    """Detect regions with edge discontinuities"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel)
    eroded = cv2.erode(edges, kernel)
    edge_gaps = cv2.absdiff(dilated, eroded)
    edge_gaps = cv2.threshold(edge_gaps, 30, 255, cv2.THRESH_BINARY)[1]
    return edge_gaps
def detect_lighting_anomalies(image):
    """Detect regions with lighting inconsistencies"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2]
    kernel_size = 15
    local_mean = cv2.blur(brightness, (kernel_size, kernel_size))
    local_std = cv2.blur((brightness - local_mean) ** 2, (kernel_size, kernel_size))
    local_std = np.sqrt(local_std)
    threshold = np.mean(local_std) + 2 * np.std(local_std)
    lighting_mask = (local_std > threshold).astype(np.uint8) * 255
    return lighting_mask
def detect_noise_anomalies(image):
    """Detect regions with noise inconsistencies"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_residual = extract_noise_residual(gray.astype(np.float32))
    mean_noise = np.mean(noise_residual)
    std_noise = np.std(noise_residual)
    high_noise = (noise_residual > mean_noise + 2 * std_noise).astype(np.uint8) * 255
    low_noise = (noise_residual < mean_noise - 2 * std_noise).astype(np.uint8) * 255
    noise_mask = cv2.bitwise_or(high_noise, low_noise)
    return noise_mask
def detect_color_anomalies(image):
    """Detect regions with color correlation anomalies"""
    channels = cv2.split(image)
    correlation_map = np.zeros_like(channels[0], dtype=np.float32)
    window_size = 32
    for i in range(0, correlation_map.shape[0] - window_size, window_size // 2):
        for j in range(0, correlation_map.shape[1] - window_size, window_size // 2):
            r_patch = channels[2][i:i+window_size, j:j+window_size].flatten()
            g_patch = channels[1][i:i+window_size, j:j+window_size].flatten()
            b_patch = channels[0][i:i+window_size, j:j+window_size].flatten()
            rg_corr = abs(np.corrcoef(r_patch, g_patch)[0, 1]) if len(r_patch) > 1 else 0
            rb_corr = abs(np.corrcoef(r_patch, b_patch)[0, 1]) if len(r_patch) > 1 else 0
            gb_corr = abs(np.corrcoef(g_patch, b_patch)[0, 1]) if len(g_patch) > 1 else 0
            avg_local_corr = (rg_corr + rb_corr + gb_corr) / 3
            correlation_map[i:i+window_size, j:j+window_size] = avg_local_corr
    mean_corr = np.mean(correlation_map)
    std_corr = np.std(correlation_map)
    high_corr_mask = (correlation_map > mean_corr + 2 * std_corr).astype(np.uint8) * 255
    low_corr_mask = (correlation_map < mean_corr - 2 * std_corr).astype(np.uint8) * 255
    color_mask = cv2.bitwise_or(high_corr_mask, low_corr_mask)
    return color_mask
def extract_regions_from_mask(mask, region_type):
    """Extract bounding boxes from a binary mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, x + w, y + h))
    return regions
def detect_compression_anomaly_regions(image):
    """Detect regions with compression artifacts"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    anomaly_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height-8, 8):
        for j in range(0, width-8, 8):
            block = gray[i:i+8, j:j+8].astype(float)
            dct_block = cv2.dct(block)
            high_freq_energy = np.sum(np.abs(dct_block[5:, 5:]))
            low_freq_energy = np.sum(np.abs(dct_block[:3, :3]))
            if low_freq_energy > 0:
                ratio = high_freq_energy / low_freq_energy
                if ratio > 0.5:
                    anomaly_mask[i:i+8, j:j+8] = 255
    return anomaly_mask
def detect_statistical_anomaly_regions(image):
    """Detect regions with statistical anomalies"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    anomaly_mask = np.zeros((height, width), dtype=np.uint8)
    entropy_map = np.zeros_like(gray, dtype=float)
    for i in range(1, height-1):
        for j in range(1, width-1):
            patch = gray[i-1:i+2, j-1:j+2]
            hist = np.histogram(patch, bins=256, range=(0, 256))[0]
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist))
            entropy_map[i, j] = entropy
    mean_entropy = np.mean(entropy_map)
    std_entropy = np.std(entropy_map)
    high_entropy = (entropy_map > mean_entropy + 2 * std_entropy).astype(np.uint8) * 255
    low_entropy = (entropy_map < mean_entropy - 2 * std_entropy).astype(np.uint8) * 255
    anomaly_mask = cv2.bitwise_or(high_entropy, low_entropy)
    return anomaly_mask
def detect_splicing_boundaries(image):
    """Detect splicing boundaries in the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_mask = np.zeros_like(gray)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(boundary_mask, [contour], -1, 255, 2)
    return boundary_mask
def create_ai_highlighted_image(image_path):
    """
    Create a fully red-highlighted version of an AI-generated image.
    Returns a base64 encoded image string.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    overlay = np.full_like(image, (0, 0, 255), dtype=np.uint8)
    highlighted = cv2.addWeighted(image, 0.3, overlay, 0.7, 0)
    highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(highlighted_rgb)
    from io import BytesIO
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"
def detect_smoothness_anomalies(gray_image):
    """
    Detect unnatural smoothness in AI-generated images.
    Returns a score from 0-1 where higher values indicate more smoothness (more likely AI-generated).
    Now more conservative to reduce false positives.
    """
    kernel_size = 5
    local_var = cv2.blur((gray_image - cv2.blur(gray_image, (kernel_size, kernel_size)))**2, (kernel_size, kernel_size))
    mean_variance = np.mean(local_var)
    smoothness_score = 1.0 - min(1.0, mean_variance / 1500.0)
    return smoothness_score
def detect_ai_statistical_anomalies(gray_image):
    """
    Detect statistical anomalies typical of AI-generated images.
    Returns a score from 0-1 where higher values indicate more anomalies.
    """
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    hist_std = np.std(hist)
    hist_entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    max_entropy = np.log2(256)
    uniformity_score = 1.0 - (hist_entropy / max_entropy)
    return uniformity_score
def detect_ai_noise_patterns(gray_image):
    """
    Detect noise patterns that are atypical for natural images but common in AI generation.
    Returns a score from 0-1 where higher values indicate AI-like noise.
    """
    noise = extract_noise_residual(gray_image.astype(np.float32))
    noise_std = np.std(noise)
    noise_entropy = -np.sum(noise[noise != 0] * np.log2(np.abs(noise[noise != 0])))
    noise_pattern_score = min(1.0, noise_entropy / 1000.0)
    return noise_pattern_score
def detect_ai_color_artifacts(image):
    """
    Detect color artifacts typical of AI-generated images.
    Returns a score from 0-1 where higher values indicate AI artifacts.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channels = cv2.split(lab)
    correlations = []
    for i in range(3):
        for j in range(i+1, 3):
            corr = np.corrcoef(channels[i].flatten(), channels[j].flatten())[0, 1]
            correlations.append(abs(corr))
    correlation_variation = np.std(correlations)
    ai_color_score = 1.0 - min(1.0, correlation_variation * 5)
    return ai_color_score
def detect_frequency_anomalies(gray_image):
    """
    Detect frequency domain anomalies typical of AI-generated images.
    Returns a score from 0-1 where higher values indicate AI anomalies.
    """
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    height, width = magnitude_spectrum.shape
    center_h, center_w = height // 2, width // 2
    low_freq = magnitude_spectrum[center_h-10:center_h+10, center_w-10:center_w+10]
    low_energy = np.sum(low_freq)
    high_freq_regions = [
        magnitude_spectrum[:height//4, :width//4],
        magnitude_spectrum[:height//4, -width//4:],
        magnitude_spectrum[-height//4:, :width//4],
        magnitude_spectrum[-height//4:, -width//4:]
    ]
    high_energy = sum(np.sum(region) for region in high_freq_regions)
    if low_energy > 0:
        freq_ratio = high_energy / low_energy
        ai_freq_score = max(0, 1.0 - freq_ratio / 10.0)
    else:
        ai_freq_score = 0.5
    return ai_freq_score
def calculate_optimized_forgery_score(results):
    """Calculate optimized forgery score from 4 key analyses"""
    weights = {
        'copy_move': 0.4,
        'compression': 0.3,
        'noise': 0.15,
        'edges': 0.15
    }
    overall_score = 0
    suspicious_indicators = 0
    for method, analysis in results.items():
        if 'consistency' in analysis:
            score = analysis['consistency']
        elif 'artifacts' in analysis:
            score = analysis['artifacts']
        else:
            score = 0.5
        overall_score += score * weights[method]
        if analysis.get('suspicious', False):
            suspicious_indicators += 1
    if suspicious_indicators >= 2:
        overall_score = min(1.0, overall_score * 1.2)
    elif suspicious_indicators >= 3:
        overall_score = min(1.0, overall_score * 1.4)
    return {
        'overall_score': overall_score,
        'suspicious_indicators': suspicious_indicators,
        'method_scores': {method: analysis.get('consistency', analysis.get('artifacts', 0.5))
                         for method, analysis in results.items()},
        'confidence_factors': suspicious_indicators
    }
def assess_optimized_confidence(forgery_score):
    """Assess confidence level for optimized detection"""
    score = forgery_score['overall_score']
    indicators = forgery_score['suspicious_indicators']
    if score > 0.8 and indicators >= 3:
        return 'Very High'
    elif score > 0.7 and indicators >= 2:
        return 'High'
    elif score > 0.6 and indicators >= 1:
        return 'Moderate'
    elif score > 0.5:
        return 'Low'
    else:
        return 'Very Low'
def classify_optimized_forgery_type(results):
    """Classify forgery type based on optimized analysis"""
    suspicious_methods = [method for method, analysis in results.items()
                         if analysis.get('suspicious', False)]
    if 'copy_move' in suspicious_methods:
        return 'Copy-Move Forgery'
    elif 'compression' in suspicious_methods:
        return 'Compression Artifacts'
    elif 'noise' in suspicious_methods:
        return 'Noise Inconsistency'
    elif 'edges' in suspicious_methods:
        return 'Edge Discontinuity'
    elif len(suspicious_methods) > 1:
        return 'Multiple Forgery Types'
    else:
        return 'No Forgery Detected'
def detect_optimized_segments(image, results):
    """Detect suspicious segments using optimized approach"""
    height, width = image.shape[:2]
    suspicious_mask = np.zeros((height, width), dtype=np.uint8)
    segments = []
    if results["copy_move"]["suspicious"]:
        mask, regions = detect_copy_move(None, fast_mode=True)
        if mask is not None:
            suspicious_mask = cv2.bitwise_or(suspicious_mask, mask)
            segments.extend(regions[:5] if regions else [])
    if results["compression"]["suspicious"]:
        comp_mask = detect_compression_anomaly_regions(image)
        if comp_mask is not None:
            suspicious_mask = cv2.bitwise_or(suspicious_mask, comp_mask)
            segments.extend(extract_regions_from_mask(comp_mask, "compression")[:3])
    if results["edges"]["suspicious"]:
        edge_mask = detect_edge_anomalies(image)
        if edge_mask is not None:
            suspicious_mask = cv2.bitwise_or(suspicious_mask, edge_mask)
            segments.extend(extract_regions_from_mask(edge_mask, "edges")[:3])
    segments = merge_overlapping_regions(segments)
    segments = [seg for seg in segments if (seg[2] - seg[0]) * (seg[3] - seg[1]) > 200]
    return {
        "segments": segments,
        "mask": suspicious_mask,
        "segment_count": len(segments),
        "total_suspicious_area": np.sum(suspicious_mask > 0) / suspicious_mask.size
    }
def create_forgery_highlighted_image(image_path, suspicious_segments):
    """Create a highlighted image showing suspicious regions"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    highlighted = image.copy()
    for segment in suspicious_segments.get("segments", []):
        x1, y1, x2, y2 = segment
        cv2.rectangle(highlighted, (x1, y1), (x2, y2), (0, 0, 255), 3)
    highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(highlighted_rgb)
    from io import BytesIO
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"