"""
QUICK FIX for Post-Prediction Crashes
====================================
Apply this code to your app.py to fix the crash issue
"""

import gc
import tensorflow as tf
from functools import wraps
import psutil
import os

def memory_cleanup_decorator(f):
    """Decorator to ensure memory cleanup after prediction"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Log memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            print(f"🧠 Memory before prediction: {memory_before:.1f} MB")
            
            # Execute the prediction
            result = f(*args, **kwargs)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            raise
            
        finally:
            # CRITICAL: Always clean up, even on error
            try:
                print("🧹 Starting memory cleanup...")
                
                # Clear TensorFlow session
                tf.keras.backend.clear_session()
                
                # Force garbage collection
                gc.collect()
                
                # Log memory after
                process = psutil.Process()
                memory_after = process.memory_info().rss / 1024 / 1024
                print(f"🧠 Memory after cleanup: {memory_after:.1f} MB")
                
                print("✅ Memory cleanup completed")
                
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup error (non-critical): {cleanup_error}")
    
    return decorated_function

# APPLY THIS TO YOUR PREDICTION ROUTE:
"""
@app.route('/api/predict', methods=['POST'])
@memory_cleanup_decorator
def predict():
    # Your existing prediction code here
    # ... (keep all your current prediction logic)
    
    # Just make sure you return the result at the end:
    return jsonify({
        'verse_number': predicted_verse,
        'confidence': confidence_score,
        'top3_predictions': top3_results
    })
"""

# ALSO ADD THESE TO YOUR FLASK APP CONFIGURATION:
"""
# Add to your app configuration (after app = Flask(__name__))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.teardown_appcontext
def cleanup_context(error):
    '''Cleanup after each request'''
    try:
        tf.keras.backend.clear_session()
        gc.collect()
    except:
        pass
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

if __name__ == '__main__':
    print("🛠️ QUICK FIX MODULE LOADED")
    print("=" * 40)
    print("1. Copy the @memory_cleanup_decorator to your app.py")
    print("2. Apply it to your prediction route")
    print("3. Add the app configuration code")
    print("4. Restart your Flask app")
    print("5. Test multiple predictions")
    print("=" * 40)
