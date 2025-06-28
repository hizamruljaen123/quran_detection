"""
Setup Script untuk Aplikasi Deteksi Ayat Al-Quran
==================================================
Script ini membantu setup aplikasi web deteksi ayat Al-Quran
"""

import os
import sys
import subprocess
import mysql.connector
from mysql.connector import Error

def install_requirements():
    """Install Python packages dari requirements.txt"""
    print("📦 Installing Python packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages!")
        return False

def create_database():
    """Membuat database dan tabel yang diperlukan"""
    print("🗄️ Setting up database...")
    
    try:
        # Connect to MySQL server
        connection = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='',
            port=3306
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # Create database
            cursor.execute("CREATE DATABASE IF NOT EXISTS quran_db")
            cursor.execute("USE quran_db")
            
            # Create table
            with open('../sceheme.sql', 'r', encoding='utf-8') as file:
                sql_script = file.read()
                # Execute each statement
                for statement in sql_script.split(';'):
                    if statement.strip():
                        cursor.execute(statement)
            
            connection.commit()
            print("✅ Database created successfully!")
            
            # Insert sample data
            print("📝 Inserting sample data...")
            with open('sample_data.sql', 'r', encoding='utf-8') as file:
                sql_script = file.read()
                cursor.execute(sql_script)
            
            connection.commit()
            print("✅ Sample data inserted successfully!")
            
            return True
            
    except Error as e:
        print(f"❌ Database error: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def create_directories():
    """Membuat direktori yang diperlukan"""
    print("📁 Creating required directories...")
    
    directories = [
        'static/uploads',
        'models',
        'static/css',
        'static/js'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created successfully!")

def main():
    """Main setup function"""
    print("🚀 Setting up Quran Verse Detection Web Application")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Please run this script from the Web directory!")
        return
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        return
    
    # Setup database
    if not create_database():
        print("⚠️  Database setup failed. Please check your MySQL configuration.")
        print("   Make sure MySQL is running and accessible with:")
        print("   - Host: 127.0.0.1")
        print("   - User: root")
        print("   - Password: (empty)")
        print("   - Port: 3306")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("1. Make sure you have trained model files in '../model_saves_quran_model_final/'")
    print("2. Run: python app.py")
    print("3. Open your browser to: http://localhost:5000")
    print("\nEnjoy using the Quran Verse Detection application! 🕌")

if __name__ == "__main__":
    main()
