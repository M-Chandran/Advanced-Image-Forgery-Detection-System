#!/usr/bin/env python3
"""
Simple test script to verify forgery detection functions
"""
import os
import sys
sys.path.append('.')
from utils.image_processing import detect_copy_move, detect_image_forgery, enhanced_forgery_detection, detect_ai_compression_artifacts
def test_detection():
    sample_dir = os.path.join(os.path.dirname(__file__), 'static', 'samples')
    if not os.path.exists(sample_dir):
        print("Sample directory not found")
        return
    images = ['original.jpg', 'copy_move_forgery.jpg', 'splicing_forgery.jpg', 'jpeg_compressed.jpg']
    for img_name in images:
        img_path = os.path.join(sample_dir, img_name)
        if os.path.exists(img_path):
            print(f"\nTesting {img_name}:")
            try:
                mask, regions = detect_copy_move(img_path)
                copy_move_detected = len(regions) > 0 if regions else False
                print(f"  Copy-move detected: {copy_move_detected}")
                if regions:
                    print(f"  Regions found: {len(regions)}")
            except Exception as e:
                print(f"  Copy-move detection error: {e}")
            try:
                ai_compressed = detect_ai_compression_artifacts(img_path)
                print(f"  AI compression detected: {ai_compressed}")
            except Exception as e:
                print(f"  AI compression detection error: {e}")
            try:
                result = enhanced_forgery_detection(img_path, False)
                print(f"  Forgery result: {result.get('forgery_detected', 'Unknown')}")
                print(f"  Forgery score: {result.get('forgery_score', 'N/A')}")
            except Exception as e:
                print(f"  Enhanced detection error: {e}")
        else:
            print(f"Image {img_name} not found")
if __name__ == '__main__':
    test_detection()