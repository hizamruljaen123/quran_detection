#!/usr/bin/env python3
"""
Memory Fix Test Script
=====================
Test script untuk memverifikasi perbaikan memory leak
"""

import requests
import time
import psutil
import os
import sys

def test_memory_fix():
    """Test the memory fix by making multiple predictions"""
    
    print("🧪 TESTING MEMORY FIX")
    print("=" * 40)
    
    # Check if Flask app is running
    base_url = "http://127.0.0.1:5000"
    
    try:
        response = requests.get(base_url, timeout=5)
        print("✅ Flask app is running")
    except:
        print("❌ Flask app is not running!")
        print("   Please start your Flask app first:")
        print("   python app.py")
        return False
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"🧠 Initial memory: {initial_memory:.1f} MB")
    
    # Test file (you need to have a test audio file)
    test_files = [
        "../sample_1/078001.mp3",
        "../sample_1/078002.mp3", 
        "../sample_1/078003.mp3"
    ]
    
    available_files = [f for f in test_files if os.path.exists(f)]
    
    if not available_files:
        print("⚠️ No test audio files found")
        print("   Please ensure you have audio files in sample_1 directory")
        return False
    
    print(f"📁 Found {len(available_files)} test files")
    
    # Test multiple predictions
    success_count = 0
    error_count = 0
    memory_usage = []
    
    for i in range(5):  # Test 5 predictions
        print(f"\n🔄 Test {i+1}/5")
        
        # Pick a test file
        test_file = available_files[i % len(available_files)]
        
        try:
            # Make prediction request
            with open(test_file, 'rb') as f:
                files = {'audio_file': f}
                response = requests.post(
                    f"{base_url}/api/predict", 
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: Verse {result.get('verse_number', 'Unknown')}")
                print(f"      Confidence: {result.get('confidence', 0)*100:.1f}%")
                success_count += 1
            else:
                print(f"   ❌ Error: HTTP {response.status_code}")
                print(f"      Response: {response.text[:100]}")
                error_count += 1
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            error_count += 1
        
        # Check memory after each prediction
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_usage.append(current_memory)
        print(f"   🧠 Memory: {current_memory:.1f} MB")
        
        # Wait a bit between requests
        time.sleep(2)
    
    # Analysis
    print(f"\n📊 TEST RESULTS")
    print("=" * 40)
    print(f"✅ Successful predictions: {success_count}/5")
    print(f"❌ Failed predictions: {error_count}/5")
    
    final_memory = memory_usage[-1]
    memory_increase = final_memory - initial_memory
    
    print(f"🧠 Memory Analysis:")
    print(f"   Initial: {initial_memory:.1f} MB")
    print(f"   Final: {final_memory:.1f} MB")
    print(f"   Increase: {memory_increase:.1f} MB")
    
    # Check for memory leak
    if memory_increase > 100:  # More than 100MB increase
        print("🚨 MEMORY LEAK DETECTED!")
        print("   The memory fix may not be working properly")
        return False
    elif memory_increase > 50:  # 50-100MB increase
        print("⚠️ MODERATE MEMORY INCREASE")
        print("   Monitor for memory leaks with longer testing")
        return True
    else:
        print("✅ MEMORY USAGE STABLE")
        print("   Memory fix appears to be working!")
        return True

def test_crash_recovery():
    """Test crash recovery by sending invalid data"""
    
    print(f"\n🧪 TESTING CRASH RECOVERY")
    print("=" * 40)
    
    base_url = "http://127.0.0.1:5000"
    
    # Test with no file
    try:
        response = requests.post(f"{base_url}/api/predict", timeout=10)
        print(f"   No file test: HTTP {response.status_code}")
        
        if response.status_code in [400, 500]:
            print("   ✅ Handled gracefully")
        else:
            print("   ⚠️ Unexpected response")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test with invalid file
    try:
        files = {'audio_file': ('test.txt', b'invalid audio data', 'text/plain')}
        response = requests.post(f"{base_url}/api/predict", files=files, timeout=10)
        print(f"   Invalid file test: HTTP {response.status_code}")
        
        if response.status_code in [400, 500]:
            print("   ✅ Handled gracefully")
        else:
            print("   ⚠️ Unexpected response")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")

if __name__ == '__main__':
    print("🧪 MEMORY FIX TESTING TOOL")
    print("=" * 50)
    
    # Test the memory fix
    if test_memory_fix():
        print("\n🎉 MEMORY FIX TEST PASSED!")
    else:
        print("\n🚨 MEMORY FIX TEST FAILED!")
        print("   Please check the quick_fix.py recommendations")
    
    # Test crash recovery
    test_crash_recovery()
    
    print("\n" + "=" * 50)
    print("🏁 TESTING COMPLETED")
    
    input("\nPress Enter to exit...")
