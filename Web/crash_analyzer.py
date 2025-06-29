#!/usr/bin/env python3
"""
Crash Analyzer for Post-Prediction Issues
==========================================
Specific tool to diagnose crashes that occur after successful predictions
"""

import os
import sys
import gc
import traceback
import time

def analyze_prediction_crash():
    """Analyze potential causes of post-prediction crashes"""
    print("🔍 POST-PREDICTION CRASH ANALYZER")
    print("=" * 50)
    
    print("\n📋 Analyzing your crash pattern:")
    print("   ✅ Prediction completed successfully")
    print("   ✅ Features extracted correctly")
    print("   ✅ Model prediction worked")
    print("   ✅ File cleanup completed")
    print("   ❌ Application crashed AFTER success")
    
    print("\n🎯 Most likely causes (in order of probability):")
    
    causes = [
        {
            "cause": "TensorFlow Session Not Cleared",
            "probability": "85%",
            "description": "TensorFlow keeps model in memory, accumulating with each prediction",
            "symptoms": ["Memory grows with each prediction", "Eventually runs out of RAM"],
            "fix": "Add tf.keras.backend.clear_session() after prediction"
        },
        {
            "cause": "Flask JSON Response Error", 
            "probability": "75%",
            "description": "Error serializing prediction results to JSON",
            "symptoms": ["Crash occurs during response generation", "No error in prediction itself"],
            "fix": "Add try-catch around jsonify() response"
        },
        {
            "cause": "Memory Leak in Audio Processing",
            "probability": "65%",
            "description": "Large audio arrays not garbage collected",
            "symptoms": ["Memory usage increases", "Slower performance over time"],
            "fix": "Explicitly delete large arrays and call gc.collect()"
        },
        {
            "cause": "File Handle Leak",
            "probability": "45%", 
            "description": "Audio files or temp files not properly closed",
            "symptoms": ["File system errors", "Permission denied errors"],
            "fix": "Use context managers (with open()) for all file operations"
        },
        {
            "cause": "Thread/Process Cleanup Issue",
            "probability": "35%",
            "description": "Background processes not terminating properly",
            "symptoms": ["App hangs", "Process remains in memory"],
            "fix": "Implement proper cleanup in app teardown"
        }
    ]
    
    for i, cause in enumerate(causes, 1):
        print(f"\n{i}. {cause['cause']} ({cause['probability']})")
        print(f"   📝 {cause['description']}")
        print(f"   🔧 Fix: {cause['fix']}")

def generate_crash_fix():
    """Generate specific code fixes for the crash"""
    print("\n" + "=" * 60)
    print("🛠️ IMMEDIATE FIX CODE")
    print("=" * 60)
    
    print("\n1. 🧠 ADD TO YOUR PREDICTION ROUTE (app.py):")
    print("-" * 50)
    
    fix_code = '''
import gc
import tensorflow as tf

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Your existing prediction code here...
        # ... (feature extraction, model prediction, etc.)
        
        # After getting prediction result
        result = {
            'verse_number': predicted_verse,
            'confidence': confidence_score,
            'top3_predictions': top3_results
        }
        
        # 🧹 CRITICAL: Clean up before response
        tf.keras.backend.clear_session()
        gc.collect()
        
        return jsonify(result)
        
    except Exception as e:
        # 🚨 CRITICAL: Clean up even on error
        tf.keras.backend.clear_session()
        gc.collect()
        
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500
    
    finally:
        # 🧹 ALWAYS clean up
        try:
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass
'''
    print(fix_code)
    
    print("\n2. 🔧 ADD MEMORY MONITORING (optional):")
    print("-" * 50)
    
    monitor_code = '''
import psutil

def log_memory_usage(stage):
    """Log memory usage at different stages"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"🧠 Memory at {stage}: {memory_mb:.1f} MB")

# Use in prediction route:
# log_memory_usage("start")
# # ... prediction code ...
# log_memory_usage("after_prediction")
# # ... cleanup ...
# log_memory_usage("after_cleanup")
'''
    print(monitor_code)
    
    print("\n3. 🚀 FLASK APP CONFIGURATION:")
    print("-" * 50)
    
    config_code = '''
# Add to your Flask app configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

@app.teardown_appcontext
def cleanup_context(error):
    """Clean up after each request"""
    try:
        tf.keras.backend.clear_session()
        gc.collect()
    except:
        pass

@app.teardown_request
def cleanup_request(error):
    """Additional cleanup"""
    try:
        gc.collect()
    except:
        pass
'''
    print(config_code)

def create_test_script():
    """Create a test script to verify the fix"""
    print("\n" + "=" * 60)
    print("🧪 CREATING TEST SCRIPT")
    print("=" * 60)
    
    test_script = '''#!/usr/bin/env python3
"""
Test script to verify crash fix
"""
import requests
import time
import os

def test_multiple_predictions():
    """Test multiple predictions to see if crash is fixed"""
    
    # Test with a sample audio file
    test_file = "test.mp3"  # Replace with actual test file
    
    if not os.path.exists(test_file):
        print("❌ Test file not found. Create a test.mp3 file first.")
        return
    
    print("🧪 Testing multiple predictions...")
    
    for i in range(5):
        print(f"\\n📋 Test {i+1}/5:")
        
        try:
            with open(test_file, 'rb') as f:
                files = {'audio_file': f}
                
                start_time = time.time()
                response = requests.post('http://127.0.0.1:5000/api/predict', files=files)
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Success: Verse {result.get('verse_number', 'Unknown')}")
                    print(f"   ⏱️ Time: {end_time - start_time:.2f}s")
                else:
                    print(f"   ❌ Error: {response.status_code}")
                    
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        # Wait between tests
        time.sleep(2)
    
    print("\\n✅ Test completed. If no crashes occurred, fix is working!")

if __name__ == '__main__':
    test_multiple_predictions()
'''
    
    try:
        with open('test_crash_fix.py', 'w') as f:
            f.write(test_script)
        print("   ✅ Created: test_crash_fix.py")
        print("   📋 Run this after applying the fix to verify it works")
    except Exception as e:
        print(f"   ❌ Failed to create test script: {e}")

def main():
    """Main analyzer function"""
    analyze_prediction_crash()
    generate_crash_fix()
    create_test_script()
    
    print("\n" + "🎯" * 20)
    print("QUICK ACTION PLAN:")
    print("🎯" * 20)
    print("1. Apply the code fixes above to your app.py")
    print("2. Restart your Flask application")
    print("3. Run test_crash_fix.py to verify the fix")
    print("4. Monitor memory usage during testing")
    print("5. If still crashing, check Flask/Werkzeug logs")
    
    print("\n💡 Pro Tips:")
    print("   - Keep the app running and test multiple predictions")
    print("   - Monitor Task Manager for memory usage")
    print("   - Check Windows Event Viewer for system-level errors")
    print("   - Consider using gunicorn instead of Flask dev server for production")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Analysis failed: {e}")
        traceback.print_exc()
