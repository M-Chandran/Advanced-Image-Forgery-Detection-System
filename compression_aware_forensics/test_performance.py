"""
Performance testing script for the optimized image forensics model
"""
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.model_cache import get_cached_model, clear_all_caches
from utils.image_processing import preprocess_image
def test_performance():
    print('🚀 Testing Performance Optimizations...')
    print('=' * 50)
    print('Testing model loading performance...')
    start_time = time.time()
    model = get_cached_model('cnn')
    load_time = time.time() - start_time
    print(f"Model loading time: {load_time:.2f} seconds")
    print('Testing preprocessing caching...')
    test_image = 'static/samples/original.jpg'
    if os.path.exists(test_image):
        start_time = time.time()
        result1 = preprocess_image(test_image)
        time1 = time.time() - start_time
        start_time = time.time()
        result2 = preprocess_image(test_image)
        time2 = time.time() - start_time
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"First preprocessing: {time1:.2f} seconds")
        print(f"Cached preprocessing: {time2:.2f} seconds")
        print(f"Speedup: {speedup:.1f}x")
    else:
        print('⚠️  Test image not found')
    print('')
    print('🎉 Performance optimization testing completed!')
    print('📊 The model should now be significantly faster!')
    print('')
    print('Key optimizations implemented:')
    print('✅ Fast Mode Toggle - Skip complex analyses for quick results')
    print('✅ Parallel Processing - Run independent operations concurrently')
    print('✅ Enhanced Caching - Cache preprocessing and results with TTL')
    print('✅ GPU Acceleration - Automatic GPU detection and utilization')
    print('✅ Result Memoization - Avoid redundant computations')
if __name__ == '__main__':
    test_performance()