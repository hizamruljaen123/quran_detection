# 🕌 Quran Verse Detection Web Application

Aplikasi web berbasis Flask untuk deteksi ayat Al-Quran Surah An-Naba menggunakan teknologi AI dan Machine Learning.

## ✨ Fitur Utama

- 🎵 **Upload & Deteksi Audio**: Upload file audio dan dapatkan prediksi ayat Al-Quran
- 📊 **Dataset Explorer**: Jelajahi dan analisis dataset audio yang tersedia
- 🤖 **Model Management**: Informasi model AI, training, dan evaluasi
- 📖 **Database Ayat**: Akses ayat, terjemahan, dan transliterasi lengkap
- 🎯 **API Integration**: RESTful API untuk integrasi dengan aplikasi lain
- 📱 **Responsive UI**: Antarmuka modern dan responsif

## 🚀 Quick Start

### Prasyarat

- Python 3.8+
- MySQL Server
- Git

### Installation

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd quran_detect/Web
   ```

2. **Setup otomatis**
   ```bash
   python setup.py
   ```
   
   Script ini akan:
   - Install semua dependencies
   - Membuat database dan tabel
   - Insert sample data
   - Setup direktori yang diperlukan

3. **Jalankan aplikasi**
   ```bash
   python app.py
   ```

4. **Akses aplikasi**
   Buka browser dan kunjungi: `http://localhost:5000`

### Manual Setup (Alternatif)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup database MySQL**
   - Buat database `quran_db`
   - Import schema dari `../sceheme.sql`
   - Import sample data dari `sample_data.sql`

3. **Konfigurasi database** (jika perlu)
   Edit konfigurasi database di `app.py`:
   ```python
   DB_CONFIG = {
       'host': '127.0.0.1',
       'database': 'quran_db',
       'user': 'root',
       'password': '',
       'port': 3306
   }
   ```

## 📁 Struktur Project

```
quran_detect/
├── Web/                          # Aplikasi web Flask
│   ├── app.py                   # Main Flask application
│   ├── requirements.txt         # Python dependencies
│   ├── setup.py                # Setup script
│   ├── sample_data.sql          # Sample database data
│   ├── static/                  # Static files
│   │   ├── css/style.css       # Custom CSS
│   │   ├── js/main.js          # Custom JavaScript
│   │   └── uploads/            # Upload directory
│   ├── templates/              # HTML templates
│   │   ├── index.html          # Homepage
│   │   ├── upload.html         # Upload page
│   │   ├── result.html         # Prediction results
│   │   ├── dataset.html        # Dataset explorer
│   │   ├── model_info.html     # Model information
│   │   ├── training.html       # Training interface
│   │   ├── verses.html         # Verses list
│   │   └── verse_detail.html   # Verse details
│   └── models/                 # Model storage directory
├── sample_1/ to sample_7/       # Audio dataset
├── model_saves_quran_model_final/ # Trained model files
├── sceheme.sql                  # Database schema
└── *.ipynb                     # Jupyter notebooks (training)
```

## 🎯 Cara Penggunaan

### 1. Upload & Prediksi Audio

1. Kunjungi halaman "Upload Audio"
2. Pilih file audio (.mp3, .wav, .flac)
3. Klik "Prediksi Ayat"
4. Lihat hasil prediksi beserta confidence score

### 2. Eksplorasi Dataset

1. Kunjungi halaman "Dataset"
2. Lihat statistik dataset
3. Browse file audio yang tersedia
4. Analisis distribusi data

### 3. Management Model

1. Kunjungi halaman "Model Info"
2. Lihat informasi model yang dimuat
3. Check model metadata dan performa
4. Akses training interface (jika diperlukan)

### 4. Browse Ayat Al-Quran

1. Kunjungi halaman "Ayat Al-Quran"
2. Browse daftar ayat Surah An-Naba
3. Klik ayat untuk melihat detail
4. Lihat teks Arab, terjemahan, dan transliterasi

## 🔧 API Endpoints

### Prediction API
```bash
POST /api/predict
Content-Type: multipart/form-data

# Upload file audio dan dapatkan prediksi
curl -X POST -F "audio=@file.mp3" http://localhost:5000/api/predict
```

### Training Status API
```bash
GET /api/training/status

# Cek status training model
curl http://localhost:5000/api/training/status
```

## 📊 Model AI

Aplikasi menggunakan model deep learning untuk klasifikasi audio:

- **Input**: File audio (MP3, WAV, FLAC)
- **Features**: MFCC, Spectral Features, Chroma, Tonnetz
- **Model**: Neural Network (TensorFlow/Keras)
- **Output**: Prediksi nomor ayat dengan confidence score

### Model Files

Model yang sudah ditraining tersimpan di:
- `model_saves_quran_model_final/quran_model_final.keras`
- `model_saves_quran_model_final/label_encoder_final.pkl`
- `model_saves_quran_model_final/model_metadata_final.json`

## 🗄️ Database Schema

```sql
CREATE TABLE quran_id (
    id INT PRIMARY KEY,
    suraId INT,
    verseID INT,
    ayahText TEXT,      -- Teks Arab
    indoText TEXT,      -- Terjemahan Indonesia
    readText TEXT       -- Transliterasi
);
```

## 🛠️ Development

### Menjalankan dalam Mode Development

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

### Training Model Baru

1. Siapkan dataset audio dalam folder `sample_*`
2. Buka Jupyter notebook `quran_detection_final.ipynb`
3. Jalankan semua cell untuk training
4. Model akan tersimpan otomatis

### Kustomisasi

- **UI/UX**: Edit file di `templates/` dan `static/`
- **Model**: Modifikasi fungsi di `app.py` atau notebook
- **Database**: Update schema di `sceheme.sql`

## 📝 Troubleshooting

### Error Database Connection
```bash
mysql.connector.errors.ProgrammingError: 1049 (42000): Unknown database 'quran_db'
```
**Solusi**: Jalankan `python setup.py` atau buat database manual

### Error Model Not Found
```bash
FileNotFoundError: Model file not found
```
**Solusi**: Pastikan file model ada di `model_saves_quran_model_final/`

### Error Audio Processing
```bash
librosa.util.exceptions.ParameterError
```
**Solusi**: Pastikan file audio valid dan format didukung

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

Jika ada pertanyaan atau masalah, silakan buat issue di repository ini.

---

**Developed with ❤️ for Islamic Technology**
