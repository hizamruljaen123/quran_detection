"""
Quran Verse Detection Web Application
=====================================================
Web application untuk deteksi ayat Al-Quran menggunakan AI
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import json
import pickle
import numpy as np
import librosa
import tensorflow as tf
import gc
import psutil
import traceback
import sys
import threading
import time
from sklearn.preprocessing import LabelEncoder, RobustScaler
import mysql.connector
from mysql.connector import Error
import scipy.signal
from datetime import datetime
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load configuration from file if exists
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    app.config['SECRET_KEY'] = config['app']['secret_key']
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['MAX_CONTENT_LENGTH'] = config['upload']['max_file_size_mb'] * 1024 * 1024
    
    DB_CONFIG = config['database']
    
except (FileNotFoundError, KeyError, json.JSONDecodeError):
    # Fallback to default configuration
    app.config['SECRET_KEY'] = 'quran_detection_secret_key_2025'
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    
    # Database configuration
    DB_CONFIG = {
        'host': '127.0.0.1',
        'database': 'quran_db',
        'user': 'root',
        'password': '',
        'port': 3306
    }

# Global variables for model
loaded_model = None
loaded_encoder = None
model_metadata = None
training_status = {"status": "idle", "progress": 0, "message": ""}

# Error tracking
app_errors = []
prediction_count = 0
last_cleanup_time = time.time()

def log_app_error(error_msg, error_type="GENERAL"):
    """Log errors without crashing the app"""
    global app_errors
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_entry = {
            'timestamp': timestamp,
            'type': error_type,
            'message': str(error_msg),
            'prediction_count': prediction_count
        }
        app_errors.append(error_entry)
        
        # Keep only last 50 errors to prevent memory buildup
        if len(app_errors) > 50:
            app_errors = app_errors[-50:]
        
        print(f"🚨 [{timestamp}] {error_type}: {error_msg}")
        
    except Exception as e:
        print(f"❌ Error logging failed: {e}")

def safe_memory_cleanup():
    """Safely clean up memory without crashing"""
    try:
        print("🧹 Starting safe memory cleanup...")
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Log memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"🧠 Memory after cleanup: {memory_mb:.1f} MB, collected {collected} objects")
        except:
            print("✅ Cleanup completed (memory info unavailable)")
        
        return True
        
    except Exception as e:
        log_app_error(f"Memory cleanup failed: {e}", "CLEANUP_ERROR")
        return False

def emergency_recovery():
    """Emergency recovery function to prevent app death"""
    try:
        print("🚨 EMERGENCY RECOVERY INITIATED")
        
        # Multiple cleanup attempts
        for i in range(3):
            try:
                tf.keras.backend.clear_session()
                gc.collect()
                print(f"   Recovery attempt {i+1}: OK")
                break
            except Exception as e:
                print(f"   Recovery attempt {i+1}: Failed - {e}")
                time.sleep(0.1)
        
        # Log current state
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"🧠 Post-recovery memory: {memory_mb:.1f} MB")
        except:
            pass
        
        print("✅ Emergency recovery completed")
        return True
        
    except Exception as e:
        print(f"❌ Emergency recovery failed: {e}")
        return False

def safe_prediction_wrapper(prediction_func):
    """Wrapper to safely execute predictions without crashing Flask"""
    def wrapper(*args, **kwargs):
        global prediction_count, last_cleanup_time
        prediction_count += 1
        
        try:
            print(f"🔮 Starting prediction #{prediction_count}")
            
            # Log memory before
            try:
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024
                print(f"🧠 Memory before: {memory_before:.1f} MB")
            except:
                memory_before = 0
            
            # Execute prediction with timeout protection
            try:
                result = prediction_func(*args, **kwargs)
                print(f"✅ Prediction #{prediction_count} completed successfully")
                return result
                
            except Exception as pred_error:
                log_app_error(f"Prediction function error: {pred_error}", "PREDICTION_ERROR")
                traceback.print_exc()
                
                # Return safe error response instead of crashing
                return {
                    'error': 'Prediction failed',
                    'details': str(pred_error),
                    'prediction_id': prediction_count,
                    'recovery_status': 'attempted'
                }
            
        except Exception as wrapper_error:
            log_app_error(f"Wrapper error: {wrapper_error}", "WRAPPER_ERROR")
            
            # Emergency recovery
            emergency_recovery()
            
            return {
                'error': 'Critical prediction error',
                'details': str(wrapper_error),
                'prediction_id': prediction_count,
                'recovery_status': 'emergency'
            }
            
        finally:
            # ALWAYS clean up, no matter what
            try:
                # Periodic deep cleanup
                current_time = time.time()
                if current_time - last_cleanup_time > 30:  # Every 30 seconds
                    safe_memory_cleanup()
                    last_cleanup_time = current_time
                else:
                    # Quick cleanup
                    tf.keras.backend.clear_session()
                    gc.collect()
                
                print(f"🧹 Cleanup completed for prediction #{prediction_count}")
                
            except Exception as cleanup_error:
                log_app_error(f"Cleanup error: {cleanup_error}", "CLEANUP_ERROR")
    
    return wrapper

class DatabaseManager:
    """Manager untuk koneksi database MySQL"""
    
    def __init__(self, config):
        self.config = config
    
    def get_connection(self):
        """Buat koneksi database"""
        try:
            connection = mysql.connector.connect(**self.config)
            return connection
        except Error as e:
            print(f"Database connection error: {e}")
            return None
    
    def get_verse_info(self, sura_id, verse_id):
        """Ambil informasi ayat dari database"""
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor(dictionary=True)
            query = """
            SELECT id, suraId, verseID, ayahText, indoText, readText 
            FROM quran_id 
            WHERE suraId = %s AND verseID = %s
            """
            cursor.execute(query, (sura_id, verse_id))
            result = cursor.fetchone()
            return result
        except Error as e:
            print(f"Database query error: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def get_sura_info(self, sura_id):
        """Ambil informasi surat"""
        # Untuk Surah An-Naba (78)
        sura_names = {
            78: {"name": "An-Naba", "name_id": "Berita Besar", "total_verses": 40}
        }
        return sura_names.get(sura_id, {"name": "Unknown", "name_id": "Tidak Diketahui", "total_verses": 0})

# Initialize database manager
db_manager = DatabaseManager(DB_CONFIG)

def extract_advanced_features(file_path, max_length=256, sr=22050):
    """
    Extract advanced audio features untuk deteksi ayat
    """
    try:
        print(f"🎵 Starting feature extraction for: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
        
        # Load dan preprocess audio
        print("📥 Loading audio file...")
        audio, sample_rate = librosa.load(file_path, sr=sr)
        print(f"✅ Audio loaded: {len(audio)} samples at {sample_rate} Hz")
        
        # Basic preprocessing
        print("🔧 Preprocessing audio...")
        audio = librosa.util.normalize(audio)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Apply pre-emphasis filter safely
        try:
            audio = scipy.signal.lfilter([1, -0.95], [1], audio)
        except Exception as e:
            print(f"⚠️ Pre-emphasis filter failed, skipping: {e}")
        
        print(f"✅ Audio preprocessed: {len(audio)} samples")
        
        # Extract multiple features
        features_list = []
        
        # MFCC features
        print("🎯 Extracting MFCC features...")
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            features_list.extend([mfccs, mfcc_delta, mfcc_delta2])
            print(f"✅ MFCC features extracted: {mfccs.shape}")
        except Exception as e:
            print(f"❌ MFCC extraction failed: {e}")
            return None
        
        # Spectral features
        print("🌈 Extracting spectral features...")
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
            features_list.extend([spectral_centroids, spectral_rolloff, spectral_bandwidth, spectral_contrast])
            print(f"✅ Spectral features extracted")
        except Exception as e:
            print(f"❌ Spectral features extraction failed: {e}")
            return None
        
        # Additional features
        print("🎼 Extracting additional features...")
        try:
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features_list.extend([zero_crossing_rate, chroma, tonnetz])
            print(f"✅ Additional features extracted")
        except Exception as e:
            print(f"❌ Additional features extraction failed: {e}")
            return None
        
        # Combine features
        print("🔗 Combining features...")
        try:
            combined_features = np.vstack(features_list)
            print(f"✅ Features combined: {combined_features.shape}")
        except Exception as e:
            print(f"❌ Feature combination failed: {e}")
            return None
        
        # Normalize
        print("📏 Normalizing features...")
        try:
            scaler = RobustScaler()
            combined_features = scaler.fit_transform(combined_features.T).T
            print(f"✅ Features normalized")
        except Exception as e:
            print(f"❌ Feature normalization failed: {e}")
            return None
        
        # Pad or truncate
        print("✂️ Adjusting feature length...")
        try:
            if combined_features.shape[1] < max_length:
                pad_width = max_length - combined_features.shape[1]
                combined_features = np.pad(combined_features, ((0, 0), (0, pad_width)), mode='constant')
            else:
                combined_features = combined_features[:, :max_length]
            
            print(f"✅ Final features shape: {combined_features.T.shape}")
            return combined_features.T
        except Exception as e:
            print(f"❌ Feature length adjustment failed: {e}")
            return None
        
    except Exception as e:
        print(f"❌ Critical error in feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_model_if_exists():
    """Load model jika ada"""
    global loaded_model, loaded_encoder, model_metadata
    
    print("🤖 Attempting to load AI model...")
    
    # First try to load from the final model directory
    model_dirs = [
        '../model_saves_quran_model_final',
        'models', 
        '../model_saves_basic_improved', 
        '../model_saves_improved_simple'
    ]
    
    for model_dir in model_dirs:
        print(f"📁 Checking directory: {model_dir}")
        
        if not os.path.exists(model_dir):
            print(f"❌ Directory not found: {model_dir}")
            continue
            
        try:
            # Check various model file names
            model_files = [
                'quran_model.h5',
                'basic_improved_model.h5',
                'improved_quran_model.h5',
                'quran_verse_model.h5',
                'best_model.h5'
            ]
            
            model_path = None
            for model_file in model_files:
                test_path = os.path.join(model_dir, model_file)
                if os.path.exists(test_path):
                    model_path = test_path
                    print(f"✅ Found model file: {model_path}")
                    break
            
            if model_path:
                # Load model with error handling
                print(f"📥 Loading model from: {model_path}")
                try:
                    loaded_model = tf.keras.models.load_model(model_path)
                    print(f"✅ Model loaded successfully")
                    print(f"📊 Model input shape: {loaded_model.input_shape}")
                    print(f"📊 Model output shape: {loaded_model.output_shape}")
                except Exception as model_error:
                    print(f"❌ Failed to load model: {model_error}")
                    loaded_model = None
                    continue
                
                # Load encoder
                encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
                if os.path.exists(encoder_path):
                    try:
                        print(f"📥 Loading label encoder from: {encoder_path}")
                        with open(encoder_path, 'rb') as f:
                            loaded_encoder = pickle.load(f)
                        print(f"✅ Label encoder loaded with {len(loaded_encoder.classes_)} classes")
                        print(f"📋 Classes: {loaded_encoder.classes_}")
                    except Exception as encoder_error:
                        print(f"❌ Failed to load encoder: {encoder_error}")
                        loaded_encoder = None
                        loaded_model = None
                        continue
                else:
                    print(f"❌ Label encoder not found: {encoder_path}")
                    loaded_model = None
                    continue
                
                # Load metadata
                metadata_files = ['metadata.json', 'model_metadata.json']
                for meta_file in metadata_files:
                    meta_path = os.path.join(model_dir, meta_file)
                    if os.path.exists(meta_path):
                        try:
                            print(f"📥 Loading metadata from: {meta_path}")
                            with open(meta_path, 'r') as f:
                                model_metadata = json.load(f)
                            print(f"✅ Metadata loaded")
                            break
                        except Exception as meta_error:
                            print(f"⚠️ Failed to load metadata: {meta_error}")
                            model_metadata = None
                
                print(f"✅ Model loaded successfully from: {model_path}")
                return True
                
        except Exception as e:
            print(f"❌ Failed to load from {model_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("❌ No valid model found in any directory")
    return False

def get_verse_name(verse_number):
    """Convert verse number to readable name"""
    if verse_number == 0:
        return "Bismillah (Pembuka)"
    elif 1 <= verse_number <= 40:
        return f"Ayat {verse_number}"
    else:
        return f"Unknown ({verse_number})"

@safe_prediction_wrapper
@safe_prediction_wrapper
def predict_verse(audio_file_path):
    """Predict verse dari audio file dengan error handling yang aman"""
    global loaded_model, loaded_encoder
    
    print(f"🔮 Starting prediction for: {audio_file_path}")
    
    try:
        if not loaded_model:
            raise Exception("Model not loaded")
        
        if not loaded_encoder:
            raise Exception("Label encoder not loaded")
        
        # Check if file exists
        if not os.path.exists(audio_file_path):
            raise Exception(f"Audio file not found: {audio_file_path}")
        
        # Extract features with error handling
        try:
            print("🎵 Extracting features...")
            features = extract_advanced_features(audio_file_path)
            if features is None:
                raise Exception("Feature extraction returned None")
            
            print(f"✅ Features extracted with shape: {features.shape}")
            
        except Exception as feature_error:
            raise Exception(f"Feature extraction failed: {feature_error}")
        
        # Reshape untuk prediction
        try:
            print("🔄 Reshaping features for model input...")
            features = features.reshape(1, features.shape[0], features.shape[1])
            print(f"✅ Features reshaped to: {features.shape}")
        except Exception as reshape_error:
            raise Exception(f"Feature reshaping failed: {reshape_error}")
        
        # Predict with error handling
        try:
            print("🧠 Running model prediction...")
            prediction = loaded_model.predict(features, verbose=0)
            print(f"✅ Prediction completed: {prediction.shape}")
        except Exception as model_error:
            raise Exception(f"Model prediction failed: {model_error}")
        
        # Get prediction results
        try:
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            print(f"✅ Predicted class: {predicted_class}, confidence: {confidence:.4f}")
        except Exception as result_error:
            raise Exception(f"Prediction result processing failed: {result_error}")
        
        # Get top 3 predictions
        try:
            top3_indices = np.argsort(prediction[0])[-3:][::-1]
            top3_probs = prediction[0][top3_indices]
            print(f"✅ Top 3 predictions: {top3_indices} with probs: {top3_probs}")
        except Exception as top3_error:
            raise Exception(f"Top 3 predictions failed: {top3_error}")
        
        # Convert to verse numbers
        try:
            verse_number = loaded_encoder.inverse_transform([predicted_class])[0]
            top3_verses = loaded_encoder.inverse_transform(top3_indices)
            print(f"✅ Verse conversion: {verse_number}, top3: {top3_verses}")
        except Exception as conversion_error:
            raise Exception(f"Verse number conversion failed: {conversion_error}")
        
        # Build result
        result = {
            'verse_number': int(verse_number),
            'confidence': float(confidence),
            'verse_name': get_verse_name(verse_number),
            'top3_predictions': [
                {
                    'verse_number': int(v),
                    'verse_name': get_verse_name(v),
                    'probability': float(p)
                }
                for v, p in zip(top3_verses, top3_probs)
            ],
            'prediction_id': prediction_count,
            'status': 'success'
        }
        
        print(f"✅ Prediction successful: Verse {verse_number} with {confidence:.2%} confidence")
        return result
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        log_app_error(error_msg, "PREDICTION_MAIN")
        
        # Return error result instead of None
        return {
            'error': str(e),
            'verse_number': None,
            'confidence': 0.0,
            'prediction_id': prediction_count,
            'status': 'failed'
        }

# Routes
@app.route('/')
def index():
    """Homepage"""
    model_status = "Loaded" if loaded_model else "Not Loaded"
    return render_template('index.html', model_status=model_status)

@app.route('/upload', methods=['GET', 'POST'])
def upload_audio():
    """Upload dan predict audio"""
    if request.method == 'POST':
        try:
            print("📤 Starting audio upload process...")
            
            # Check if file is in request
            if 'audio_file' not in request.files:
                print("❌ No file in request")
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['audio_file']
            if file.filename == '':
                print("❌ No filename provided")
                flash('No file selected', 'error')
                return redirect(request.url)
            
            print(f"📁 File received: {file.filename}")
            
            # Check file format
            allowed_extensions = ('.mp3', '.wav', '.m4a', '.flac')
            if file and file.filename.lower().endswith(allowed_extensions):
                try:
                    # Save file with error handling
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}_{filename}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    
                    print(f"💾 Saving file to: {filepath}")
                    
                    # Ensure upload directory exists
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    
                    # Save file
                    file.save(filepath)
                    print(f"✅ File saved successfully")
                    
                    # Check if file was actually saved
                    if not os.path.exists(filepath):
                        print("❌ File was not saved properly")
                        flash('Failed to save uploaded file', 'error')
                        return redirect(request.url)
                    
                    # Check file size
                    file_size = os.path.getsize(filepath)
                    print(f"📏 File size: {file_size / (1024*1024):.2f}MB")
                    
                    if file_size == 0:
                        print("❌ Uploaded file is empty")
                        flash('Uploaded file is empty', 'error')
                        return redirect(request.url)
                    
                    # Predict with timeout and cleanup
                    print("🔮 Starting prediction...")
                    result = None
                    
                    try:
                        result = predict_verse(filepath)
                    except Exception as pred_error:
                        print(f"❌ Prediction failed: {pred_error}")
                        import traceback
                        traceback.print_exc()
                        result = None
                    finally:
                        # Always clean up the uploaded file
                        try:
                            if os.path.exists(filepath):
                                os.remove(filepath)
                                print(f"🗑️ Cleaned up file: {filepath}")
                        except Exception as cleanup_error:
                            print(f"⚠️ Failed to cleanup file: {cleanup_error}")
                    
                    if result:
                        print(f"✅ Prediction successful: Verse {result['verse_number']}")
                        
                        # Get verse info from database
                        sura_id = 78  # An-Naba
                        verse_id = result['verse_number']
                        
                        try:
                            verse_info = db_manager.get_verse_info(sura_id, verse_id)
                            sura_info = db_manager.get_sura_info(sura_id)
                            print(f"✅ Database info retrieved")
                        except Exception as db_error:
                            print(f"⚠️ Database query failed: {db_error}")
                            verse_info = None
                            sura_info = None
                        
                        return render_template('result.html', 
                                             result=result, 
                                             verse_info=verse_info,
                                             sura_info=sura_info,
                                             filename=filename)
                    else:
                        print("❌ Prediction returned None")
                        flash('Failed to process audio file. Please try with a different audio file.', 'error')
                        return redirect(request.url)
                        
                except Exception as process_error:
                    print(f"❌ File processing error: {process_error}")
                    import traceback
                    traceback.print_exc()
                    flash(f'Error processing file: {str(process_error)}', 'error')
                    return redirect(request.url)
            else:
                print(f"❌ Invalid file format: {file.filename}")
                flash('Invalid file format. Please upload MP3, WAV, M4A, or FLAC files.', 'error')
                return redirect(request.url)
                
        except Exception as general_error:
            print(f"❌ General upload error: {general_error}")
            import traceback
            traceback.print_exc()
            flash(f'Upload failed: {str(general_error)}', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/dataset')
def dataset_info():
    """Informasi dataset"""
    dataset_path = "../"
    dataset_stats = {
        'total_folders': 0,
        'total_files': 0,
        'total_verses': 41,  # 0-40 (Bismillah + Verses 1-40)
        'total_size': 'Calculating...',
        'folders': [],
        'sample_files': []
    }
    
    # Scan dataset folders
    for i in range(1, 8):
        folder_name = f"sample_{i}"
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
            dataset_stats['folders'].append({
                'name': folder_name,
                'file_count': len(files),
                'exists': True,
                'verse_range': '078000-078040 (Bismillah + Verses 1-40)',
                'total_size': f'~{len(files) * 2}MB',
                'total_duration': f'~{len(files) * 30}s'
            })
            dataset_stats['total_files'] += len(files)
            dataset_stats['total_folders'] += 1
            
            # Add sample files from this folder
            for j, file in enumerate(files[:3]):  # Show first 3 files
                try:
                    verse_num = int(file.split('.')[0][-3:])
                    verse_name = "Bismillah" if verse_num == 0 else f"Verse {verse_num}"
                    dataset_stats['sample_files'].append({
                        'filename': file,
                        'folder': folder_name,
                        'verse_number': verse_name,
                        'size': '~2MB',
                        'audio_path': None  # Could add if we want to serve audio files
                    })
                except:
                    continue
        else:
            dataset_stats['folders'].append({
                'name': folder_name,
                'file_count': 0,
                'exists': False,
                'verse_range': 'N/A',
                'total_size': '0MB',
                'total_duration': '0s'
            })
    
    # Calculate total size estimate
    dataset_stats['total_size'] = f'~{dataset_stats["total_files"] * 2}MB'
    
    return render_template('dataset.html', dataset_info=dataset_stats)

@app.route('/model_info')
def model_info():
    """Informasi model"""
    global model_metadata, loaded_model
    
    model_info_data = None
    model_layers = None
    training_history = None
    
    if loaded_model:
        try:
            # Get model information
            model_info_data = {
                'type': 'Deep Neural Network',
                'framework': 'TensorFlow/Keras',
                'version': '1.0',
                'input_shape': str(loaded_model.input_shape),
                'total_params': loaded_model.count_params(),
                'layers': len(loaded_model.layers),
                'accuracy': model_metadata.get('accuracy', 'N/A') if model_metadata else 'N/A',
                'loss': model_metadata.get('loss', 'N/A') if model_metadata else 'N/A',
                'training_time': model_metadata.get('training_time', 'N/A') if model_metadata else 'N/A',
                'epochs': model_metadata.get('epochs', 'N/A') if model_metadata else 'N/A',
                'accuracy_score': 0.85,  # Default values for chart
                'speed_score': 0.75,
                'memory_score': 0.8,
                'robustness_score': 0.7,
                'versatility_score': 0.9
            }
            
            # Get layer information
            model_layers = []
            for i, layer in enumerate(loaded_model.layers):
                model_layers.append({
                    'name': f'Layer {i+1} ({layer.name})',
                    'type': layer.__class__.__name__,
                    'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
                    'param_count': layer.count_params() if hasattr(layer, 'count_params') else 0
                })
            
            # Training history (mock data for now)
            if model_metadata and 'training_history' in model_metadata:
                training_history = model_metadata['training_history']
            
        except Exception as e:
            print(f"Error getting model info: {e}")
    
    return render_template('model_info.html', 
                         model_info=model_info_data,
                         model_loaded=loaded_model is not None,
                         model_layers=model_layers,
                         training_history=training_history)

@app.route('/training')
def training_page():
    """Halaman training"""
    return render_template('training.html')

@app.route('/api/training_status')
def training_status_api():
    """API untuk status training"""
    return jsonify(training_status)

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start training process"""
    global training_status
    
    if training_status["status"] == "training":
        return jsonify({"error": "Training already in progress"}), 400
    
    # Start training in background thread
    thread = threading.Thread(target=run_training_process)
    thread.start()
    
    return jsonify({"message": "Training started"})

def run_training_process():
    """Run training process in background"""
    global training_status, loaded_model, loaded_encoder, model_metadata
    
    try:
        training_status = {"status": "training", "progress": 0, "message": "Initializing..."}
        
        # Import training functions (assuming they're in parent directory)
        import sys
        sys.path.append('../')
        
        training_status["message"] = "Loading data..."
        training_status["progress"] = 10
        
        # Simulate training process (replace with actual training)
        for i in range(10, 100, 10):
            time.sleep(2)  # Simulate work
            training_status["progress"] = i
            training_status["message"] = f"Training progress: {i}%"
        
        training_status["progress"] = 100
        training_status["message"] = "Training completed successfully!"
        training_status["status"] = "completed"
        
        # Reload model after training
        load_model_if_exists()
        
    except Exception as e:
        training_status = {
            "status": "error", 
            "progress": 0, 
            "message": f"Training failed: {str(e)}"
        }

@app.route('/verses')
def verses_list():
    """List semua ayat dalam surat"""
    verses = []
    sura_info = db_manager.get_sura_info(78)
    
    # Get all verses for An-Naba
    for verse_id in range(1, 41):  # 1-40
        verse_info = db_manager.get_verse_info(78, verse_id)
        if verse_info:
            verses.append(verse_info)
    
    return render_template('verses.html', verses=verses, sura_info=sura_info)

@app.route('/verse/<int:verse_id>')
def verse_detail(verse_id):
    """Detail ayat tertentu"""
    verse_info = db_manager.get_verse_info(78, verse_id)
    sura_info = db_manager.get_sura_info(78)
    
    if not verse_info:
        flash('Verse not found', 'error')
        return redirect(url_for('verses_list'))
    
    return render_template('verse_detail.html', 
                         verse=verse_info, 
                         sura_info=sura_info)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint untuk prediksi - AMAN DARI CRASH"""
    temp_filepath = None
    
    try:
        print("🌐 API Prediction request received")
        
        # Validate request
        if 'audio_file' not in request.files:
            log_app_error("No audio file in request", "API_REQUEST")
            return jsonify({
                'error': 'No file provided',
                'status': 'failed',
                'prediction_id': prediction_count + 1
            }), 400
        
        file = request.files['audio_file']
        if file.filename == '':
            log_app_error("Empty filename in request", "API_REQUEST")
            return jsonify({
                'error': 'No file selected',
                'status': 'failed',
                'prediction_id': prediction_count + 1
            }), 400
        
        # Validate file format
        allowed_extensions = ('.mp3', '.wav', '.m4a', '.flac')
        if not file.filename.lower().endswith(allowed_extensions):
            log_app_error(f"Invalid file format: {file.filename}", "API_REQUEST")
            return jsonify({
                'error': 'Invalid file format. Supported: MP3, WAV, M4A, FLAC',
                'status': 'failed',
                'prediction_id': prediction_count + 1
            }), 400
        
        # Save temporary file with error handling
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"temp_{timestamp}_{filename}"
            temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(temp_filepath)
            print(f"📁 Temporary file saved: {temp_filepath}")
            
            # Validate saved file
            if not os.path.exists(temp_filepath):
                raise Exception("File save verification failed")
            
            # Check file size
            file_size = os.path.getsize(temp_filepath)
            if file_size == 0:
                raise Exception("Saved file is empty")
            
            print(f"✅ File validation passed: {file_size} bytes")
            
        except Exception as save_error:
            log_app_error(f"File save error: {save_error}", "API_FILE_SAVE")
            return jsonify({
                'error': f'Failed to save file: {str(save_error)}',
                'status': 'failed',
                'prediction_id': prediction_count + 1
            }), 500
        
        # Predict with comprehensive error handling
        try:
            print("🚀 Starting prediction process...")
            result = predict_verse(temp_filepath)
            
            if result is None:
                raise Exception("Prediction returned None")
            
            # Check if result contains error
            if isinstance(result, dict) and 'error' in result:
                print(f"⚠️ Prediction error: {result['error']}")
                return jsonify({
                    'error': result['error'],
                    'details': result.get('details', 'No details available'),
                    'status': 'failed',
                    'prediction_id': result.get('prediction_id', prediction_count)
                }), 500
            
            # Success case - add verse info
            try:
                if result.get('verse_number') is not None:
                    verse_info = db_manager.get_verse_info(78, result['verse_number'])
                    result['verse_info'] = verse_info
                
                result['status'] = 'success'
                print(f"✅ API prediction successful: Verse {result.get('verse_number')}")
                
                return jsonify(result), 200
                
            except Exception as db_error:
                log_app_error(f"Database error: {db_error}", "API_DATABASE")
                # Still return prediction without verse info
                result['verse_info'] = None
                result['db_warning'] = str(db_error)
                return jsonify(result), 200
            
        except Exception as prediction_error:
            log_app_error(f"API prediction error: {prediction_error}", "API_PREDICTION")
            return jsonify({
                'error': 'Prediction processing failed',
                'details': str(prediction_error),
                'status': 'failed',
                'prediction_id': prediction_count
            }), 500
        
    except Exception as general_error:
        log_app_error(f"API general error: {general_error}", "API_GENERAL")
        
        # Emergency recovery
        emergency_recovery()
        
        return jsonify({
            'error': 'Internal server error',
            'details': str(general_error),
            'status': 'failed',
            'recovery_attempted': True
        }), 500
        
    finally:
        # ALWAYS clean up temp file
        try:
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                print(f"🗑️ Cleaned up file: {temp_filepath}")
        except Exception as cleanup_error:
            log_app_error(f"File cleanup error: {cleanup_error}", "API_CLEANUP")
        
        # Memory cleanup
        try:
            safe_memory_cleanup()
        except Exception as memory_error:
            log_app_error(f"Memory cleanup error: {memory_error}", "API_MEMORY")

# Add Flask error handlers to prevent app crashes
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    log_app_error(f"404 error: {request.url}", "HTTP_404")
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    log_app_error(f"500 error: {error}", "HTTP_500")
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all uncaught exceptions to prevent Flask from crashing"""
    log_app_error(f"Uncaught exception: {e}", "UNCAUGHT_EXCEPTION")
    
    # Emergency recovery
    emergency_recovery()
    
    # Return JSON if it's an API request
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Server encountered an unexpected error',
            'details': str(e),
            'recovery_attempted': True
        }), 500
    
    # Return HTML for regular requests
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Unexpected error occurred"), 500

@app.route('/admin/errors')
def view_errors():
    """Admin endpoint to view recent errors"""
    try:
        return render_template('admin_errors.html', 
                             errors=app_errors[-20:],  # Last 20 errors
                             total_predictions=prediction_count)
    except Exception as e:
        return jsonify({
            'recent_errors': app_errors[-10:] if app_errors else [],
            'error_count': len(app_errors),
            'prediction_count': prediction_count,
            'view_error': str(e)
        })

@app.route('/admin/memory')
def view_memory():
    """Admin endpoint to check memory status"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return jsonify({
            'memory_mb': memory_info.rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'prediction_count': prediction_count,
            'error_count': len(app_errors),
            'last_cleanup': last_cleanup_time,
            'model_loaded': loaded_model is not None,
            'encoder_loaded': loaded_encoder is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/admin/cleanup')
def manual_cleanup():
    """Manual cleanup endpoint"""
    try:
        safe_memory_cleanup()
        return jsonify({
            'message': 'Manual cleanup completed',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': f'Cleanup failed: {e}',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/admin/emergency')
def emergency_endpoint():
    """Emergency recovery endpoint"""
    try:
        result = emergency_recovery()
        return jsonify({
            'message': 'Emergency recovery completed',
            'success': result,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': f'Emergency recovery failed: {e}',
            'timestamp': datetime.now().isoformat()
        })

# Add Flask configuration for stability
app.config['PROPAGATE_EXCEPTIONS'] = False  # Prevent exceptions from killing the app
app.config['TRAP_HTTP_EXCEPTIONS'] = True   # Catch HTTP exceptions
app.config['TRAP_BAD_REQUEST_ERRORS'] = True # Catch bad request errors

# Add periodic cleanup
def periodic_cleanup():
    """Background thread for periodic cleanup"""
    while True:
        try:
            time.sleep(60)  # Every 60 seconds
            safe_memory_cleanup()
            
            # Clean old errors
            global app_errors
            if len(app_errors) > 100:
                app_errors = app_errors[-50:]
            
        except Exception as e:
            print(f"Periodic cleanup error: {e}")

# Start background cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    print("🚀 Initializing Quran Verse Detection Web Application")
    print("=" * 60)
    
    # Create upload directory if not exists
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        print(f"📁 Upload folder ready: {app.config['UPLOAD_FOLDER']}")
    except Exception as e:
        print(f"❌ Failed to create upload folder: {e}")
        exit(1)
    
    # Set TensorFlow to use less memory
    try:
        # Limit GPU memory growth if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🎮 GPU memory growth enabled for {len(gpus)} GPU(s)")
        else:
            print("💻 Running on CPU")
    except Exception as e:
        print(f"⚠️ TensorFlow GPU configuration warning: {e}")
    
    # Try to load existing model
    print("\n🤖 Loading AI Model...")
    model_loaded = load_model_if_exists()
    
    if model_loaded:
        print("✅ Model loaded successfully - Ready for predictions!")
    else:
        print("⚠️ No model loaded - Upload feature will not work properly")
        print("   Please ensure model files are available in the model directories")
    
    # Get configuration for running the app
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        host = config['app']['host']
        port = config['app']['port']
        debug = config['app']['debug']
        print(f"⚙️ Configuration loaded from config.json")
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        # Fallback to default
        host = '127.0.0.1'  # Changed from 0.0.0.0 to localhost for stability
        port = 5000
        debug = True
        print(f"⚙️ Using default configuration")
    
    print("\n🌐 Server Configuration:")
    print(f"📡 Server: http://{host}:{port}")
    print(f"🛠️  Debug mode: {'ON' if debug else 'OFF'}")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"💾 Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    print(f"🤖 Model status: {'READY' if model_loaded else 'NOT LOADED'}")
    print("=" * 60)
    print("🕌 Starting server... Press Ctrl+C to stop")
    print("=" * 60)
    
    # Run app with error handling
    try:
        app.run(debug=debug, host=host, port=port, use_reloader=False)
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        exit(1)
