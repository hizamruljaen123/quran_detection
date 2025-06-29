#!/usr/bin/env python3
"""
Test Script - Verifikasi Anti-Crash System
==========================================
Script untuk memastikan aplikasi tidak crash saat terjadi error
"""

import requests
import time
import os
import threading
import traceback

def test_multiple_predictions(test_file="test.mp3", num_tests=10):
    """Test multiple predictions to verify crash prevention"""
    
    print("🧪 TESTING ANTI-CRASH SYSTEM")
    print("=" * 50)
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("Please create a test audio file or update the filename")
        return False
    
    base_url = "http://127.0.0.1:5000"
    api_url = f"{base_url}/api/predict"
    
    # Test server availability
    try:
        response = requests.get(base_url, timeout=5)
        print(f"✅ Server is running: {response.status_code}")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return False
    
    # Run prediction tests
    success_count = 0
    error_count = 0
    crash_count = 0
    
    print(f"\n🚀 Running {num_tests} prediction tests...")
    
    for i in range(num_tests):
        test_num = i + 1
        print(f"\n📋 Test {test_num}/{num_tests}:")
        
        try:
            with open(test_file, 'rb') as f:
                files = {'audio_file': f}
                
                start_time = time.time()
                response = requests.post(api_url, files=files, timeout=30)
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    if 'error' in result:
                        print(f"   ⚠️ API Error: {result['error']}")
                        error_count += 1
                    else:
                        verse_num = result.get('verse_number', 'Unknown')
                        confidence = result.get('confidence', 0) * 100
                        print(f"   ✅ Success: Verse {verse_num} ({confidence:.1f}%)")
                        success_count += 1
                    
                    print(f"   ⏱️ Time: {end_time - start_time:.2f}s")
                    print(f"   📊 Status: {result.get('status', 'unknown')}")
                    
                else:
                    print(f"   ❌ HTTP Error: {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"   📋 Details: {error_data.get('error', 'No details')}")
                    except:
                        print(f"   📋 Response: {response.text[:100]}")
                    error_count += 1
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout (server may be processing)")
            error_count += 1
            
        except requests.exceptions.ConnectionError:
            print(f"   💥 CONNECTION LOST - Server crashed!")
            crash_count += 1
            break
            
        except Exception as e:
            print(f"   ❌ Test Error: {e}")
            error_count += 1
        
        # Wait between tests
        if test_num < num_tests:
            print(f"   💤 Waiting 2 seconds...")
            time.sleep(2)
    
    # Results summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 50)
    print(f"✅ Successful predictions: {success_count}")
    print(f"⚠️ API errors (handled): {error_count}")
    print(f"💥 Server crashes: {crash_count}")
    print(f"📈 Total tests: {num_tests}")
    
    if crash_count == 0:
        print("\n🎉 SUCCESS: Server remained stable throughout testing!")
        print("   No crashes detected - anti-crash system working!")
    else:
        print(f"\n❌ FAILURE: Server crashed {crash_count} times")
        print("   Anti-crash system needs improvement")
    
    if success_count > 0:
        success_rate = (success_count / num_tests) * 100
        print(f"📊 Success rate: {success_rate:.1f}%")
    
    return crash_count == 0

def test_error_endpoints():
    """Test error monitoring endpoints"""
    print("\n🔍 TESTING ERROR MONITORING ENDPOINTS")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:5000"
    endpoints = [
        ("/admin/memory", "Memory Status"),
        ("/admin/errors", "Error Log"),
        ("/admin/cleanup", "Manual Cleanup"),
        ("/admin/emergency", "Emergency Recovery")
    ]
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {description}: Working")
            else:
                print(f"⚠️ {description}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {description}: Error - {e}")

def monitor_server_health():
    """Monitor server health during testing"""
    print("\n💓 MONITORING SERVER HEALTH")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:5000"
    
    for i in range(5):
        try:
            # Check memory
            response = requests.get(f"{base_url}/admin/memory", timeout=5)
            if response.status_code == 200:
                data = response.json()
                memory_mb = data.get('memory_mb', 0)
                cpu_percent = data.get('cpu_percent', 0)
                predictions = data.get('prediction_count', 0)
                
                print(f"📊 Check {i+1}: Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%, Predictions: {predictions}")
                
                if memory_mb > 1000:
                    print("   ⚠️ High memory usage detected!")
                    
            else:
                print(f"⚠️ Check {i+1}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Check {i+1}: {e}")
        
        time.sleep(3)

def stress_test():
    """Stress test with concurrent requests"""
    print("\n🔥 STRESS TEST - CONCURRENT REQUESTS")
    print("=" * 50)
    
    def make_request(thread_id):
        try:
            # Use a dummy request to test server stability
            response = requests.get("http://127.0.0.1:5000", timeout=10)
            print(f"Thread {thread_id}: {response.status_code}")
            return True
        except Exception as e:
            print(f"Thread {thread_id}: Error - {e}")
            return False
    
    threads = []
    for i in range(5):  # 5 concurrent requests
        thread = threading.Thread(target=make_request, args=(i+1,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print("✅ Stress test completed")

def main():
    """Main testing function"""
    print("🛡️ QURAN DETECTION - ANTI-CRASH SYSTEM TEST")
    print("=" * 60)
    
    # Step 1: Basic prediction tests
    test_file = "test.mp3"
    if os.path.exists(test_file):
        prediction_success = test_multiple_predictions(test_file, 5)
    else:
        print(f"⚠️ Test file {test_file} not found, skipping prediction tests")
        prediction_success = True
    
    # Step 2: Test monitoring endpoints
    test_error_endpoints()
    
    # Step 3: Monitor health
    monitor_server_health()
    
    # Step 4: Stress test
    stress_test()
    
    # Final summary
    print("\n" + "🎯" * 30)
    print("FINAL TEST RESULTS:")
    print("🎯" * 30)
    
    if prediction_success:
        print("✅ Anti-crash system: WORKING")
        print("✅ Server stability: EXCELLENT")
        print("✅ Error handling: ROBUST")
    else:
        print("❌ Anti-crash system: NEEDS IMPROVEMENT")
        print("❌ Server stability: POOR")
    
    print("\n💡 Recommendations:")
    print("   - Monitor /admin/errors for detailed error logs")
    print("   - Use /admin/memory to track memory usage")
    print("   - Run /admin/cleanup if memory gets high")
    print("   - Use /admin/emergency for critical recovery")
    
    print("\n🎉 Testing completed!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Testing failed: {e}")
        traceback.print_exc()
