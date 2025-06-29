"""
QUICK FIX for Post-Prediction Crashes
====================================
Apply this code to your app.py to fix the crash issue

DIAGNOSIS SUMMARY:
- Memory usage: 80.9% (HIGH)
- Python objects: 453,718 (MEMORY LEAK DETECTED)
- Main issue: TensorFlow session not cleared after prediction
- Secondary: File handles and memory not properly released
"""

import gc
import tensorflow as tf
from functools import wraps
import psutil
import os
import traceback
import logging

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
                
                # 2. Reset default graph (if using TF 1.x style)
                try:
                    tf.compat.v1.reset_default_graph()
                    print("   ✅ TensorFlow graph reset")
                except:
                    pass
                
                # 3. Force garbage collection (multiple times)
                collected = gc.collect()
                print(f"   ✅ Garbage collected: {collected} objects")
                
                # 4. Additional cleanup for specific modules
                if 'librosa' in globals():
                    # Clear any librosa caches
                    try:
                        import librosa
                        librosa.cache.clear()
                        print("   ✅ Librosa cache cleared")
                    except:
                        pass
                
                # 5. Log memory after cleanup
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

# APPLY THIS TO YOUR PREDICTION ROUTE - ENHANCED VERSION:
"""
@app.route('/api/predict', methods=['POST'])
@memory_cleanup_decorator
def predict():
    uploaded_file = None
    temp_file_path = None
    
    try:
        # Your existing prediction code here
        # But wrap file operations properly:
        
        file = request.files['audio_file']
        if file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_file_path)
            
            # Process the file
            predicted_verse, confidence_score, top3_results = your_prediction_function(temp_file_path)
            
            # Prepare response
            result = {
                'verse_number': int(predicted_verse),
                'confidence': float(confidence_score),
                'top3_predictions': [
                    {'verse_name': f'Verse {v}', 'probability': float(p)} 
                    for v, p in top3_results
                ]
            }
            
            return jsonify(result)
        else:
            return jsonify({'error': 'No file provided'}), 400
            
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500
        
    finally:
        # CRITICAL: Always cleanup files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"🗑️ Cleaned up: {temp_file_path}")
            except Exception as e:
                print(f"⚠️ File cleanup error: {e}")
"""

# ENHANCED FLASK APP CONFIGURATION:
"""
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add to your app configuration (after app = Flask(__name__))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
import atexit

def cleanup_on_exit():
    '''Cleanup when app shuts down'''
    try:
        tf.keras.backend.clear_session()
        gc.collect()
        print("🛑 App shutdown cleanup completed")
    except:
        pass

atexit.register(cleanup_on_exit)
"""

def safe_file_cleanup(file_path):
    """Safely clean up uploaded files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ Cleaned up file: {file_path}")
    except Exception as e:
        print(f"⚠️ File cleanup warning: {e}")

def log_system_resources():
    """Log current system resources"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        print(f"📊 System Resources:")
        print(f"   Memory: {memory_info.rss / 1024 / 1024:.1f} MB")
        print(f"   CPU: {cpu_percent:.1f}%")
        
        # Check if memory is getting high
        if memory_info.rss > 1024 * 1024 * 1024:  # > 1GB
            print("⚠️ High memory usage detected!")
            
    except Exception as e:
        print(f"⚠️ Resource logging error: {e}")

def emergency_memory_cleanup():
    """Emergency function to clean up memory when app starts crashing"""
    try:
        print("🚨 EMERGENCY MEMORY CLEANUP INITIATED")
        
        # Clear TensorFlow
        tf.keras.backend.clear_session()
        
        # Reset TensorFlow
        try:
            tf.compat.v1.reset_default_graph()
        except:
            pass
        
        # Multiple garbage collections
        for i in range(3):
            collected = gc.collect()
            print(f"   Round {i+1}: Collected {collected} objects")
        
        # Clear caches
        try:
            import librosa
            librosa.cache.clear()
        except:
            pass
        
        # Log current memory
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"🧠 Memory after emergency cleanup: {memory_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Emergency cleanup failed: {e}")
        return False

def create_memory_monitor():
    """Create a memory monitoring function"""
    def monitor_memory(stage_name):
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Log memory usage
            print(f"📊 {stage_name}: {memory_mb:.1f} MB")
            
            # Warning if memory is high
            if memory_mb > 800:  # 800MB threshold
                print(f"⚠️ HIGH MEMORY WARNING at {stage_name}: {memory_mb:.1f} MB")
                
            # Critical if memory is very high
            if memory_mb > 1200:  # 1.2GB threshold
                print(f"🚨 CRITICAL MEMORY at {stage_name}: {memory_mb:.1f} MB")
                emergency_memory_cleanup()
                
            return memory_mb
            
        except Exception as e:
            print(f"❌ Memory monitoring error: {e}")
            return 0
    
    return monitor_memory

def safe_json_response(data):
    """Safely create JSON response with error handling"""
    try:
        # Ensure all data is JSON serializable
        clean_data = {}
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                clean_data[key] = value
            else:
                clean_data[key] = str(value)
        
        from flask import jsonify
        return jsonify(clean_data)
        
    except Exception as e:
        print(f"❌ JSON response error: {e}")
        try:
            from flask import jsonify
            return jsonify({'error': 'Response formatting failed', 'details': str(e)}), 500
        except:
            return "Internal Server Error", 500

def crash_recovery_wrapper(prediction_func):
    """Ultimate crash recovery wrapper"""
    def wrapper(*args, **kwargs):
        memory_monitor = create_memory_monitor()
        
        try:
            memory_monitor("START")
            
            # Call the prediction function
            result = prediction_func(*args, **kwargs)
            
            memory_monitor("AFTER_PREDICTION")
            
            return result
            
        except Exception as e:
            memory_monitor("ERROR")
            print(f"🚨 CRASH DETECTED: {e}")
            traceback.print_exc()
            
            # Emergency cleanup
            emergency_memory_cleanup()
            
            # Return safe error response
            return safe_json_response({
                'error': 'Prediction crashed',
                'details': str(e),
                'recovery_attempted': True
            })
            
        finally:
            try:
                # Always cleanup
                tf.keras.backend.clear_session()
                gc.collect()
                memory_monitor("CLEANUP_COMPLETE")
            except:
                pass
    
    return wrapper

if __name__ == '__main__':
    print("🛠️ ENHANCED QUICK FIX MODULE LOADED")
    print("=" * 50)
    print("🚨 CRASH DIAGNOSIS RESULTS:")
    print("   Memory usage: 80.9% (HIGH)")
    print("   Python objects: 453,718 (MEMORY LEAK)")
    print("   Main issue: TensorFlow session not cleared")
    print("")
    print("🔧 IMMEDIATE ACTIONS:")
    print("1. Apply @memory_cleanup_decorator to your prediction route")
    print("2. Add the enhanced Flask configuration")
    print("3. Use crash_recovery_wrapper for critical functions")
    print("4. Add memory monitoring throughout your app")
    print("5. Restart your Flask app")
    print("")
    print("📋 CODE TO ADD TO YOUR app.py:")
    print("   - Copy memory_cleanup_decorator")
    print("   - Copy crash_recovery_wrapper") 
    print("   - Copy all Flask configuration code")
    print("   - Add emergency_memory_cleanup function")
    print("")
    print("🧪 TESTING:")
    print("   - Test multiple predictions in sequence")
    print("   - Monitor memory usage with Task Manager")
    print("   - Check for memory leaks after each prediction")
    print("")
    print("⚠️ IF STILL CRASHING:")
    print("   - Call emergency_memory_cleanup() manually")
    print("   - Restart the Flask process")
    print("   - Consider using gunicorn with process recycling")
    print("=" * 50)
