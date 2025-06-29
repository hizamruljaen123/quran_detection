#!/usr/bin/env python3
"""
Auto-Patch Script for Memory Fix
==============================
Script ini akan otomatis menerapkan perbaikan memory leak ke app.py
"""

import os
import shutil
import re
from datetime import datetime

def backup_app_py():
    """Backup the current app.py"""
    if os.path.exists('app.py'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'app_backup_{timestamp}.py'
        shutil.copy2('app.py', backup_name)
        print(f"✅ Backup created: {backup_name}")
        return backup_name
    else:
        print("❌ app.py not found!")
        return None

def check_if_patched():
    """Check if app.py is already patched"""
    if not os.path.exists('app.py'):
        return False
    
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    return 'memory_cleanup_decorator' in content

def add_imports():
    """Add required imports to app.py"""
    imports_to_add = [
        "import gc",
        "import psutil",
        "from functools import wraps",
        "import traceback",
        "import atexit"
    ]
    
    return '\n'.join(imports_to_add) + '\n\n'

def get_decorator_code():
    """Get the memory cleanup decorator code"""
    return '''
def memory_cleanup_decorator(f):
    """Enhanced decorator with comprehensive memory cleanup"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Log memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            print(f"🧠 Memory before prediction: {memory_before:.1f} MB")
            
            # Execute the prediction
            result = f(*args, **kwargs)
            
            # Log successful prediction
            print(f"✅ Prediction function completed successfully")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            traceback.print_exc()
            raise
            
        finally:
            # CRITICAL: Always clean up, even on error
            try:
                print("🧹 Starting comprehensive memory cleanup...")
                
                # 1. Clear TensorFlow session (MOST IMPORTANT)
                tf.keras.backend.clear_session()
                print("   ✅ TensorFlow session cleared")
                
                # 2. Force garbage collection (multiple times)
                collected = gc.collect()
                print(f"   ✅ Garbage collected: {collected} objects")
                
                # 3. Log memory after cleanup
                process = psutil.Process()
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_freed = memory_before - memory_after
                print(f"🧠 Memory after cleanup: {memory_after:.1f} MB")
                print(f"🆓 Memory freed: {memory_freed:.1f} MB")
                
                print("✅ Comprehensive memory cleanup completed")
                
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup error (non-critical): {cleanup_error}")
                traceback.print_exc()
    
    return decorated_function

def emergency_memory_cleanup():
    """Emergency function to clean up memory when app starts crashing"""
    try:
        print("🚨 EMERGENCY MEMORY CLEANUP INITIATED")
        
        # Clear TensorFlow
        tf.keras.backend.clear_session()
        
        # Multiple garbage collections
        for i in range(3):
            collected = gc.collect()
            print(f"   Round {i+1}: Collected {collected} objects")
        
        # Log current memory
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"🧠 Memory after emergency cleanup: {memory_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Emergency cleanup failed: {e}")
        return False

'''

def get_flask_config():
    """Get Flask configuration code"""
    return '''
# Enhanced Flask Configuration for Memory Management
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.before_request
def before_request():
    '''Log request start and memory usage'''
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        app.logger.info(f"🔵 Request start - Memory: {memory_mb:.1f} MB")
    except:
        pass

@app.after_request
def after_request(response):
    '''Log request completion'''
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        app.logger.info(f"🟢 Request end - Memory: {memory_mb:.1f} MB")
    except:
        pass
    return response

@app.teardown_appcontext
def cleanup_context(error):
    '''Cleanup after each request'''
    try:
        tf.keras.backend.clear_session()
        gc.collect()
        if error:
            app.logger.error(f"Request error: {error}")
    except Exception as e:
        app.logger.error(f"Cleanup error: {e}")

@app.teardown_request
def cleanup_request(error):
    '''Additional cleanup'''
    try:
        gc.collect()
    except:
        pass

# Handle shutdown gracefully
def cleanup_on_exit():
    '''Cleanup when app shuts down'''
    try:
        tf.keras.backend.clear_session()
        gc.collect()
        print("🛑 App shutdown cleanup completed")
    except:
        pass

atexit.register(cleanup_on_exit)

'''

def patch_prediction_route():
    """Instructions for patching the prediction route"""
    return '''
# MANUALLY ADD @memory_cleanup_decorator TO YOUR PREDICTION ROUTE:
# 
# Change this:
# @app.route('/api/predict', methods=['POST'])
# def predict():
#
# To this:
# @app.route('/api/predict', methods=['POST'])
# @memory_cleanup_decorator
# def predict():
#
# Also add proper error handling in your prediction function:
# try:
#     # your prediction code
#     return jsonify(result)
# except Exception as e:
#     app.logger.error(f"Prediction error: {str(e)}")
#     return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500
# finally:
#     # cleanup any temporary files
#     if temp_file_path and os.path.exists(temp_file_path):
#         try:
#             os.remove(temp_file_path)
#         except:
#             pass
'''

def apply_auto_patch():
    """Apply automatic patches to app.py"""
    
    print("🔧 AUTO-PATCH SCRIPT FOR MEMORY FIX")
    print("=" * 50)
    
    # Check if already patched
    if check_if_patched():
        print("⚠️ app.py appears to already be patched")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Backup
    backup_file = backup_app_py()
    if not backup_file:
        return False
    
    try:
        # Read current app.py
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find where to insert code
        lines = content.split('\n')
        
        # Find import section
        import_end = 0
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                import_end = i
        
        # Insert new imports
        new_imports = add_imports().split('\n')
        for i, imp in enumerate(new_imports):
            lines.insert(import_end + 1 + i, imp)
        
        # Find Flask app creation
        app_creation_line = -1
        for i, line in enumerate(lines):
            if 'app = Flask(' in line:
                app_creation_line = i
                break
        
        if app_creation_line == -1:
            print("❌ Could not find Flask app creation line")
            return False
        
        # Insert decorator and helper functions after imports
        decorator_lines = get_decorator_code().split('\n')
        insert_pos = import_end + len(new_imports) + 2
        
        for i, line in enumerate(decorator_lines):
            lines.insert(insert_pos + i, line)
        
        # Insert Flask configuration after app creation
        config_lines = get_flask_config().split('\n')
        insert_pos = app_creation_line + len(decorator_lines) + 2
        
        for i, line in enumerate(config_lines):
            lines.insert(insert_pos + i, line)
        
        # Write patched file
        with open('app.py', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print("✅ Auto-patch applied successfully!")
        print(f"📝 Manual steps remaining:")
        print(get_patch_prediction_route())
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-patch failed: {e}")
        
        # Restore backup
        if backup_file and os.path.exists(backup_file):
            shutil.copy2(backup_file, 'app.py')
            print(f"✅ Restored from backup: {backup_file}")
        
        return False

if __name__ == '__main__':
    print("🔧 MEMORY FIX AUTO-PATCHER")
    print("=" * 50)
    print("This script will automatically patch your app.py with memory fixes")
    print("A backup will be created before making changes")
    print("")
    
    response = input("Continue with auto-patch? (y/n): ")
    if response.lower() == 'y':
        if apply_auto_patch():
            print("\n🎉 AUTO-PATCH COMPLETED!")
            print("\n📋 NEXT STEPS:")
            print("1. Add @memory_cleanup_decorator to your prediction route")
            print("2. Add proper error handling to prediction function")
            print("3. Test with: python test_memory_fix.py")
            print("4. Monitor memory usage during testing")
        else:
            print("\n❌ AUTO-PATCH FAILED")
            print("Please apply the fixes manually using quick_fix.py")
    else:
        print("Patch cancelled. Use quick_fix.py for manual instructions.")
