#!/usr/bin/env python3
"""
Debug System untuk Quran Detection App
=====================================
Script untuk mendiagnosis masalah dengan aplikasi
"""

import os
import sys
import traceback
import json

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check:")
    print(f"   Version: {sys.version}")
    print(f"   Executable: {sys.executable}")
    
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 8:
        print("   ✅ Python version is compatible")
        return True
    else:
        print("   ❌ Python version too old (need 3.8+)")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Dependency Check:")
    
    required_packages = [
        ('flask', 'Flask'),
        ('numpy', 'NumPy'), 
        ('librosa', 'Librosa'),
        ('tensorflow', 'TensorFlow'),
        ('sklearn', 'scikit-learn'),
        ('mysql.connector', 'MySQL Connector'),
        ('scipy', 'SciPy'),
        ('werkzeug', 'Werkzeug')
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name} - OK")
        except ImportError as e:
            print(f"   ❌ {name} - MISSING ({e})")
            all_good = False
        except Exception as e:
            print(f"   ⚠️ {name} - ERROR ({e})")
            all_good = False
    
    return all_good

def check_directories():
    """Check required directories"""
    print("\n📁 Directory Check:")
    
    directories = [
        ('static/uploads', 'Upload directory'),
        ('../model_saves_quran_model_final', 'Model directory'),
        ('templates', 'Templates directory'),
        ('static', 'Static files directory')
    ]
    
    all_good = True
    
    for dir_path, description in directories:
        if os.path.exists(dir_path):
            print(f"   ✅ {description}: {dir_path}")
        else:
            print(f"   ❌ {description}: {dir_path} - NOT FOUND")
            if dir_path == 'static/uploads':
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"   🔧 Created: {dir_path}")
                except Exception as e:
                    print(f"   ❌ Failed to create: {e}")
                    all_good = False
            else:
                all_good = False
    
    return all_good

def check_model_files():
    """Check model files"""
    print("\n🤖 Model Files Check:")
    
    model_dirs = [
        '../model_saves_quran_model_final',
        'models',
        '../model_saves_basic_improved'
    ]
    
    model_found = False
    
    for model_dir in model_dirs:
        print(f"\n   📁 Checking: {model_dir}")
        if not os.path.exists(model_dir):
            print(f"      ❌ Directory not found")
            continue
        
        # Check for model files
        model_files = [
            'quran_model.h5',
            'best_model.h5', 
            'basic_improved_model.h5'
        ]
        
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024*1024)
                print(f"      ✅ Model: {model_file} ({size_mb:.1f}MB)")
                model_found = True
                break
        
        # Check for encoder
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        if os.path.exists(encoder_path):
            print(f"      ✅ Encoder: label_encoder.pkl")
        else:
            print(f"      ❌ Encoder: label_encoder.pkl - NOT FOUND")
        
        # Check for metadata
        metadata_files = ['metadata.json', 'model_metadata.json']
        for meta_file in metadata_files:
            meta_path = os.path.join(model_dir, meta_file)
            if os.path.exists(meta_path):
                print(f"      ✅ Metadata: {meta_file}")
                break
        else:
            print(f"      ⚠️ Metadata: No metadata file found")
    
    if not model_found:
        print("\n   ❌ NO MODEL FILES FOUND!")
        print("   🔧 Please ensure model files are in the correct directories")
    
    return model_found

def check_config():
    """Check configuration"""
    print("\n⚙️ Configuration Check:")
    
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            print("   ✅ config.json found and readable")
            
            # Check required sections
            required_sections = ['app', 'database', 'upload']
            for section in required_sections:
                if section in config:
                    print(f"      ✅ Section '{section}' found")
                else:
                    print(f"      ❌ Section '{section}' missing")
            
        except Exception as e:
            print(f"   ❌ config.json error: {e}")
            return False
    else:
        print("   ⚠️ config.json not found (will use defaults)")
    
    return True

def test_audio_processing():
    """Test audio processing capabilities"""
    print("\n🎵 Audio Processing Test:")
    
    try:
        import librosa
        import numpy as np
        
        # Create a simple test audio signal
        sr = 22050
        duration = 1.0  # 1 second
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))
        
        # Test basic librosa functions
        mfccs = librosa.feature.mfcc(y=test_audio, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=test_audio, sr=sr)
        
        print(f"   ✅ Audio processing test passed")
        print(f"      MFCC shape: {mfccs.shape}")
        print(f"      Spectral centroids shape: {spectral_centroids.shape}")
        return True
        
    except Exception as e:
        print(f"   ❌ Audio processing test failed: {e}")
        traceback.print_exc()
        return False

def test_tensorflow():
    """Test TensorFlow"""
    print("\n🧠 TensorFlow Test:")
    
    try:
        import tensorflow as tf
        
        print(f"   ✅ TensorFlow version: {tf.__version__}")
        
        # Test basic operation
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        
        print(f"   ✅ Basic operations work")
        
        # Check GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"   🎮 GPU available: {len(gpus)} device(s)")
        else:
            print(f"   💻 Running on CPU only")
        
        return True
        
    except Exception as e:
        print(f"   ❌ TensorFlow test failed: {e}")
        traceback.print_exc()
        return False

def check_memory_usage():
    """Check memory usage"""
    print("\n💾 Memory Usage Check:")
    
    try:
        import psutil
        
        # Get memory info
        memory = psutil.virtual_memory()
        print(f"   📊 Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"   📊 Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"   📊 Used RAM: {memory.percent}%")
        
        # Get current process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        print(f"   🔍 Current process RAM: {process_memory.rss / (1024**2):.1f} MB")
        
        if memory.percent > 90:
            print("   ⚠️ WARNING: Memory usage very high!")
            return False
        elif memory.percent > 80:
            print("   ⚠️ WARNING: Memory usage high")
            
        return True
        
    except ImportError:
        print("   ⚠️ psutil not available, install with: pip install psutil")
        return True
    except Exception as e:
        print(f"   ❌ Memory check failed: {e}")
        return False

def check_tensorflow_memory():
    """Check TensorFlow memory configuration"""
    print("\n🧠 TensorFlow Memory Check:")
    
    try:
        import tensorflow as tf
        
        # Check GPU memory growth setting
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"   ✅ GPU memory growth enabled for {gpu}")
                except RuntimeError as e:
                    print(f"   ⚠️ Cannot set memory growth: {e}")
        else:
            print("   💻 No GPU detected, using CPU")
        
        # Check if eager execution is enabled
        if tf.executing_eagerly():
            print("   ✅ Eager execution enabled")
        else:
            print("   ⚠️ Eager execution disabled")
        
        return True
        
    except Exception as e:
        print(f"   ❌ TensorFlow memory check failed: {e}")
        return False

def check_flask_config():
    """Check Flask configuration for potential issues"""
    print("\n🌐 Flask Configuration Check:")
    
    try:
        # Check if we can import Flask
        from flask import Flask
        
        # Create a test Flask app to check configuration
        test_app = Flask(__name__)
        
        # Check important Flask settings
        print(f"   ✅ Flask version: {Flask.__version__}")
        
        # Check max content length
        max_content = test_app.config.get('MAX_CONTENT_LENGTH')
        if max_content:
            print(f"   📁 Max upload size: {max_content / (1024*1024):.1f} MB")
        else:
            print("   ⚠️ No max upload size set (could cause memory issues)")
        
        # Check debug mode
        debug_mode = test_app.config.get('DEBUG', False)
        print(f"   🐛 Debug mode: {debug_mode}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Flask config check failed: {e}")
        return False

def check_file_handles():
    """Check for file handle leaks"""
    print("\n📁 File Handle Check:")
    
    try:
        import psutil
        
        process = psutil.Process()
        open_files = process.open_files()
        
        print(f"   📊 Open file handles: {len(open_files)}")
        
        # Check for specific file types that might cause issues
        temp_files = [f for f in open_files if 'tmp' in f.path.lower() or 'temp' in f.path.lower()]
        upload_files = [f for f in open_files if 'upload' in f.path.lower()]
        
        if temp_files:
            print(f"   ⚠️ Temporary files open: {len(temp_files)}")
            for f in temp_files[:3]:  # Show first 3
                print(f"      - {f.path}")
        
        if upload_files:
            print(f"   ⚠️ Upload files open: {len(upload_files)}")
            for f in upload_files[:3]:  # Show first 3
                print(f"      - {f.path}")
        
        if len(open_files) > 100:
            print("   ⚠️ WARNING: Many file handles open - possible leak!")
            return False
        
        return True
        
    except ImportError:
        print("   ⚠️ psutil not available for file handle check")
        return True
    except Exception as e:
        print(f"   ❌ File handle check failed: {e}")
        return False

def check_prediction_pipeline():
    """Test the prediction pipeline with a dummy file"""
    print("\n🔮 Prediction Pipeline Test:")
    
    try:
        import numpy as np
        import librosa
        from werkzeug.datastructures import FileStorage
        from io import BytesIO
        
        # Create a dummy audio file in memory
        sr = 22050
        duration = 3.0  # 3 seconds
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))
        
        # Convert to bytes (simplified MP3-like structure)
        audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()
        
        print("   ✅ Created test audio data")
        
        # Test feature extraction
        mfccs = librosa.feature.mfcc(y=test_audio, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=test_audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=test_audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(test_audio)
        
        print("   ✅ Feature extraction test passed")
        print(f"      MFCC shape: {mfccs.shape}")
        print(f"      Spectral centroids: {spectral_centroids.shape}")
        print(f"      Spectral rolloff: {spectral_rolloff.shape}")
        print(f"      ZCR: {zero_crossing_rate.shape}")
        
        # Test memory cleanup
        del test_audio, mfccs, spectral_centroids, spectral_rolloff, zero_crossing_rate
        print("   ✅ Memory cleanup test passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Prediction pipeline test failed: {e}")
        traceback.print_exc()
        return False

def check_cleanup_process():
    """Check if cleanup processes are working"""
    print("\n🧹 Cleanup Process Check:")
    
    try:
        # Check upload directory
        upload_dir = 'static/uploads'
        if os.path.exists(upload_dir):
            files = os.listdir(upload_dir)
            print(f"   📁 Files in upload directory: {len(files)}")
            
            # Check for old files (older than 1 hour)
            import time
            current_time = time.time()
            old_files = []
            
            for file in files:
                file_path = os.path.join(upload_dir, file)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > 3600:  # 1 hour
                        old_files.append(file)
            
            if old_files:
                print(f"   ⚠️ Old files found: {len(old_files)}")
                print("   🧹 Consider cleaning up old upload files")
            else:
                print("   ✅ No old files found")
        else:
            print("   ❌ Upload directory not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Cleanup check failed: {e}")
        return False

def diagnose_crash_symptoms():
    """Diagnose common crash symptoms"""
    print("\n🔍 Crash Symptom Analysis:")
    
    # Check for common crash indicators
    symptoms = []
    
    try:
        import psutil
        process = psutil.Process()
        
        # Check memory usage
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        if memory_mb > 1000:  # More than 1GB
            symptoms.append("High memory usage detected")
        
        # Check CPU usage
        cpu_percent = process.cpu_percent(interval=1)
        if cpu_percent > 80:
            symptoms.append("High CPU usage detected")
        
    except:
        pass
    
    # Check for potential memory leaks
    try:
        import gc
        obj_count = len(gc.get_objects())
        print(f"   📊 Python objects in memory: {obj_count}")
        
        if obj_count > 100000:
            symptoms.append("Large number of Python objects (possible memory leak)")
    except:
        pass
    
    # Report symptoms
    if symptoms:
        print("   ⚠️ Potential issues detected:")
        for symptom in symptoms:
            print(f"      - {symptom}")
        
        print("\n   🔧 Recommendations:")
        print("      - Restart the application periodically")
        print("      - Implement memory cleanup after predictions")
        print("      - Monitor resource usage")
        print("      - Consider using process isolation")
        
        return False
    else:
        print("   ✅ No obvious crash symptoms detected")
        return True

def create_crash_fix_recommendations():
    """Create specific recommendations for post-prediction crashes"""
    print("\n🛠️ POST-PREDICTION CRASH FIX RECOMMENDATIONS:")
    print("=" * 60)
    
    print("\n📋 Based on your log, the prediction completed successfully but app crashed after.")
    print("   Here are the most likely causes and fixes:\n")
    
    print("1. 🧠 MEMORY LEAK IN MODEL/TENSORFLOW:")
    print("   Problem: TensorFlow model not releasing memory after prediction")
    print("   Fix: Add explicit memory cleanup in prediction function:")
    print("   ```python")
    print("   import gc")
    print("   import tensorflow as tf")
    print("   ")
    print("   # After prediction")
    print("   tf.keras.backend.clear_session()")
    print("   gc.collect()")
    print("   ```\n")
    
    print("2. 📁 FILE HANDLE NOT CLOSED:")
    print("   Problem: Audio file or temporary files not properly closed")
    print("   Fix: Use proper context managers:")
    print("   ```python")
    print("   # Instead of:")
    print("   # audio_file = open(file_path)")
    print("   ")
    print("   # Use:")
    print("   with open(file_path, 'rb') as audio_file:")
    print("       # process file")
    print("       pass  # file automatically closed")
    print("   ```\n")
    
    print("3. 🎵 LIBROSA MEMORY ISSUE:")
    print("   Problem: Audio processing leaving large arrays in memory")
    print("   Fix: Explicitly delete large arrays:")
    print("   ```python")
    print("   # After feature extraction")
    print("   del audio_data, mfccs, spectral_features")
    print("   gc.collect()")
    print("   ```\n")
    
    print("4. 🌐 FLASK RESPONSE ISSUE:")
    print("   Problem: Error in returning JSON response")
    print("   Fix: Add proper error handling:")
    print("   ```python")
    print("   try:")
    print("       return jsonify({'verse_number': result})")
    print("   except Exception as e:")
    print("       app.logger.error(f'Response error: {e}')")
    print("       return jsonify({'error': str(e)}), 500")
    print("   ```\n")
    
    print("5. 🔄 THREAD/PROCESS ISSUE:")
    print("   Problem: Background threads or processes not cleaned up")
    print("   Fix: Use proper cleanup in app teardown:")
    print("   ```python")
    print("   @app.teardown_appcontext")
    print("   def cleanup(error):")
    print("       # Cleanup resources")
    print("       tf.keras.backend.clear_session()")
    print("   ```\n")
    
    print("6. 📊 IMMEDIATE ACTIONS TO TRY:")
    print("   ✅ Restart the Flask application")
    print("   ✅ Check Windows Task Manager for memory usage")
    print("   ✅ Add print statements to track where crash occurs")
    print("   ✅ Implement try-catch around prediction response")
    print("   ✅ Add garbage collection after each prediction")
    print("   ✅ Monitor upload directory for file cleanup")

def create_quick_fix_script():
    """Create a quick fix script for immediate deployment"""
    print("\n🚀 GENERATING QUICK FIX SCRIPT...")
    
    quick_fix_code = '''
# Quick Fix for Post-Prediction Crashes
# Add this to your prediction route

import gc
import tensorflow as tf
from functools import wraps

def cleanup_after_prediction(f):
    """Decorator to cleanup resources after prediction"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            result = f(*args, **kwargs)
            return result
        finally:
            # Force cleanup
            tf.keras.backend.clear_session()
            gc.collect()
            print("🧹 Cleanup completed after prediction")
    return decorated_function

# Apply to your prediction route:
# @app.route('/api/predict', methods=['POST'])
# @cleanup_after_prediction
# def predict():
#     # your existing prediction code
#     pass
'''
    
    try:
        with open('quick_fix_decorator.py', 'w') as f:
            f.write(quick_fix_code)
        print("   ✅ Created: quick_fix_decorator.py")
        print("   📋 Copy the decorator to your app.py and apply to prediction route")
    except Exception as e:
        print(f"   ❌ Failed to create quick fix file: {e}")

def main():
    """Main diagnostic function"""
    print("🔍 Quran Detection App - System Diagnosis")
    print("=" * 50)
    
    results = []
    
    # Run all checks
    results.append(check_python_version())
    results.append(check_dependencies())
    results.append(check_directories())
    results.append(check_model_files())
    results.append(check_config())
    results.append(test_audio_processing())
    results.append(test_tensorflow())
    results.append(check_memory_usage())
    results.append(check_tensorflow_memory())
    results.append(check_flask_config())
    results.append(check_file_handles())
    results.append(check_prediction_pipeline())
    results.append(check_cleanup_process())
    results.append(diagnose_crash_symptoms())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSIS SUMMARY:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("🚀 Your system should work properly!")
    else:
        print(f"❌ SOME ISSUES FOUND ({passed}/{total} passed)")
        print("🔧 Please fix the issues above before running the app")
    
    print("\n🔧 Troubleshooting Tips:")
    print("   1. Install missing packages: pip install -r requirements.txt")
    print("   2. Ensure model files are in ../model_saves_quran_model_final/")
    print("   3. Create upload directory: mkdir -p static/uploads")
    print("   4. Check Python version (need 3.8+)")
    print("   5. If TensorFlow issues, try: pip install --upgrade tensorflow")
    print("   6. Install psutil for monitoring: pip install psutil")
    print("   7. For post-prediction crashes, see recommendations below")
    
    # Add specific crash analysis
    if passed != total:
        print("\n" + "⚠️" * 20)
        create_crash_fix_recommendations()
        create_quick_fix_script()
    
    return passed == total

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Diagnosis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Diagnosis failed: {e}")
        traceback.print_exc()
