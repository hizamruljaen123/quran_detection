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
    
    return passed == total

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Diagnosis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Diagnosis failed: {e}")
        traceback.print_exc()
