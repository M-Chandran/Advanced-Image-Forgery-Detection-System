import cv2
import numpy as np
import base64
from PIL import Image
from scipy import ndimage
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
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
def detect_copy_move(image_path, block_size=16, threshold=0.8, max_blocks=1000):
    """
    Detect copy-move forgery in the image using optimized block matching.
    Returns a mask highlighting the copied regions.
    """
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
def detect_image_forgery(image_path):
    """
    Advanced image forgery detection using multiple techniques like a human expert
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to load image"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        noise_analysis = analyze_multi_scale_noise(gray)
        edge_consistency = check_edge_consistency(image)
        lighting_analysis = analyze_lighting_consistency(image)
        color_analysis = analyze_color_correlations(image)
        compression_artifacts = detect_compression_artifacts(gray)
        statistical_anomalies = detect_statistical_anomalies(gray)
        splicing_indicators = detect_splicing_artifacts(image)
        copy_move_analysis = detect_copy_move_advanced(image_path)
        forgery_score = calculate_comprehensive_forgery_score({
            'noise': noise_analysis,
            'edges': edge_consistency,
            'lighting': lighting_analysis,
            'color': color_analysis,
            'compression': compression_artifacts,
            'statistics': statistical_anomalies,
            'splicing': splicing_indicators,
            'copy_move': copy_move_analysis
        })
        confidence_level = assess_detection_confidence(forgery_score)
        forgery_type = classify_forgery_type(forgery_score)
        suspicious_segments = detect_suspicious_segments(image, {
            'noise': noise_analysis,
            'edges': edge_consistency,
            'lighting': lighting_analysis,
            'color': color_analysis,
            'compression': compression_artifacts,
            'statistics': statistical_anomalies,
            'splicing': splicing_indicators
        })
        forgery_highlighted_image = create_forgery_highlighted_image(image_path, suspicious_segments)
        return {
            'is_forged': forgery_score['overall_score'] > 0.7,
            'forgery_score': forgery_score['overall_score'],
            'confidence': confidence_level,
            'forgery_type': forgery_type,
            'detailed_analysis': forgery_score,
            'suspicious_segments': suspicious_segments,
            'forgery_highlighted_image': forgery_highlighted_image,
            'detection_methods': {
                'noise_analysis': noise_analysis,
                'edge_consistency': edge_consistency,
                'lighting_analysis': lighting_analysis,
                'color_analysis': color_analysis,
                'compression_artifacts': compression_artifacts,
                'statistical_anomalies': statistical_anomalies,
                'splicing_indicators': splicing_indicators,
                'copy_move_analysis': copy_move_analysis
            }
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
        'suspicious': copy_move_area > 0.05
    }
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