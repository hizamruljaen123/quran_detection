"""
Quick fix verification script
============================
Script untuk memverifikasi bahwa semua route dan template sudah cocok
"""

import sys
import os

# Add the Web directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app
    
    print("🧪 Testing Flask Routes and Templates...")
    print("=" * 50)
    
    with app.test_client() as client:
        # Test each route
        routes_to_test = [
            ('/', 'Homepage'),
            ('/upload', 'Upload page'),
            ('/dataset', 'Dataset page'),
            ('/model_info', 'Model info page'),
            ('/training', 'Training page'),
            ('/verses', 'Verses list page'),
        ]
        
        for route, name in routes_to_test:
            try:
                response = client.get(route)
                if response.status_code == 200:
                    print(f"✅ {name}: {route} - OK")
                else:
                    print(f"❌ {name}: {route} - Status {response.status_code}")
            except Exception as e:
                print(f"❌ {name}: {route} - Error: {str(e)}")
    
    print("\n🎉 Route testing completed!")
    print("If all routes show OK, the url_for issue should be fixed.")
    
except ImportError as e:
    print(f"❌ Could not import Flask app: {e}")
    print("Make sure you're running this from the Web directory and all dependencies are installed.")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
