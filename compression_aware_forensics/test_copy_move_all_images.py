import requests
import time
import os
def test_copy_move_detection():
    """Test that copy-move detection runs on all images for user clarity"""
    print("Starting Flask app...")
    os.system("start /B python compression_aware_forensics/app.py")
    time.sleep(3)
    base_url = "http://localhost:5000"
    test_images = [
        ("compression_aware_forensics/static/samples/original.jpg", "Authentic image"),
        ("compression_aware_forensics/static/samples/copy_move_forgery.jpg", "Forged image with copy-move"),
        ("compression_aware_forensics/static/samples/jpeg_compressed.jpg", "JPEG compressed image")
    ]
    print("\n🧪 Testing Copy-Move Detection on ALL Images")
    print("=" * 50)
    for image_path, description in test_images:
        if os.path.exists(image_path):
            print(f"\n📸 Testing: {description}")
            print(f"File: {os.path.basename(image_path)}")
            try:
                with open(image_path, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(f"{base_url}/upload", files=files)
                if response.status_code == 200:
                    result = response.json()
                    forgery_result = result.get('forgery_result', 'Unknown')
                    copy_move_detected = result.get('copy_move_detected', False)
                    has_highlighted = bool(result.get('highlighted_image'))
                    has_extracted = bool(result.get('extracted_regions'))
                    print(f"  🔍 Forgery Result: {forgery_result}")
                    print(f"  🎯 Copy-Move Detected: {copy_move_detected}")
                    print(f"  🖼️  Highlighted Image: {'Yes' if has_highlighted else 'No'}")
                    print(f"  📦 Extracted Regions: {len(result.get('extracted_regions', []))} regions")
                    if 'copy_move_detected' in result:
                        print("  ✅ Copy-move detection executed")
                    else:
                        print("  ❌ Copy-move detection missing from response")
                else:
                    print(f"  ❌ HTTP Error: {response.status_code}")
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        else:
            print(f"  ⚠️  Image not found: {image_path}")
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")
    print("\nExpected behavior:")
    print("- Copy-move detection should run on ALL images")
    print("- Suspicious regions highlighted when detected")
    print("- Extracted regions shown in separate section")
    print("- Works regardless of forgery classification")
if __name__ == "__main__":
    test_copy_move_detection()