"""
Test Script untuk Aplikasi Deteksi Ayat Al-Quran
===============================================
Script untuk menguji komponen utama aplikasi
"""

import os
import sys
import json
import mysql.connector
from mysql.connector import Error

def test_python_packages():
    """Test apakah semua package Python terinstall"""
    print("🧪 Testing Python packages...")
    
    packages = [
        'flask', 'numpy', 'librosa', 'tensorflow', 
        'sklearn', 'mysql.connector', 'scipy'
    ]
    
    failed_packages = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Missing packages: {', '.join(failed_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All packages installed!")
        return True

def test_database_connection():
    """Test koneksi database"""
    print("\n🗄️ Testing database connection...")
    
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',
            database='quran_db',
            user='root',
            password='',
            port=3306
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM quran_id")
            count = cursor.fetchone()[0]
            print(f"  ✅ Database connected! Found {count} verses")
            return True
            
    except Error as e:
        print(f"  ❌ Database error: {e}")
        print("  Make sure MySQL is running and database 'quran_db' exists")
        return False
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def test_file_structure():
    """Test struktur file yang diperlukan"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'templates/index.html',
        'templates/upload.html',
        'static/css/style.css',
        'static/js/main.js'
    ]
    
    required_dirs = [
        'static/uploads',
        'models',
        'templates',
        'static/css',
        'static/js'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/")
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print(f"\n❌ Missing files: {missing_files}")
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required files and directories exist!")
        return True

def test_model_files():
    """Test apakah file model tersedia"""
    print("\n🤖 Testing model files...")
    
    model_paths = [
        '../model_saves_quran_model_final',
        '../model_saves',
        'models'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"  📂 {path}: {len(files)} files")
            if any('.keras' in f or '.h5' in f for f in files):
                print(f"    ✅ Model files found!")
                return True
    
    print("  ⚠️  No model files found. You may need to train a model first.")
    return False

def test_config_file():
    """Test file konfigurasi"""
    print("\n⚙️ Testing configuration...")
    
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            print("  ✅ config.json loaded successfully!")
            print(f"  📊 Database: {config['database']['host']}:{config['database']['port']}")
            print(f"  🌐 App: {config['app']['host']}:{config['app']['port']}")
            return True
        except json.JSONDecodeError:
            print("  ❌ config.json is not valid JSON!")
            return False
    else:
        print("  ⚠️  config.json not found (using default settings)")
        return True

def main():
    """Main test function"""
    print("🧪 Quran Verse Detection App - System Test")
    print("=" * 50)
    
    tests = [
        test_python_packages,
        test_file_structure,
        test_database_connection,
        test_model_files,
        test_config_file
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your application is ready to run.")
        print("\nTo start the application:")
        print("  • Windows: run.bat")
        print("  • Linux/Mac: ./run.sh")
        print("  • Manual: python app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the app.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
