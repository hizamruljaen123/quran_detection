"""
Flask Route Test
================
Test individual routes to ensure they work correctly
"""

import sys
import os

# Add the Web directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_info_route():
    """Test the model_info route specifically"""
    try:
        from app import app
        
        with app.test_client() as client:
            print("🧪 Testing /model_info route...")
            response = client.get('/model_info')
            
            if response.status_code == 200:
                print("✅ /model_info route works correctly!")
                return True
            else:
                print(f"❌ /model_info route failed with status {response.status_code}")
                print(f"Response: {response.data.decode('utf-8')[:500]}...")
                return False
                
    except Exception as e:
        print(f"❌ Error testing /model_info route: {e}")
        return False

def test_all_routes():
    """Test all main routes"""
    try:
        from app import app
        
        routes_to_test = [
            ('/', 'Homepage'),
            ('/upload', 'Upload page'),
            ('/dataset', 'Dataset page'),
            ('/model_info', 'Model info page'),
            ('/training', 'Training page'),
            ('/verses', 'Verses list page'),
        ]
        
        print("🧪 Testing all routes...")
        print("=" * 40)
        
        with app.test_client() as client:
            success_count = 0
            total_count = len(routes_to_test)
            
            for route, name in routes_to_test:
                try:
                    response = client.get(route)
                    if response.status_code == 200:
                        print(f"✅ {name}: {route}")
                        success_count += 1
                    else:
                        print(f"❌ {name}: {route} - Status {response.status_code}")
                except Exception as e:
                    print(f"❌ {name}: {route} - Error: {str(e)}")
            
            print("=" * 40)
            print(f"📊 Results: {success_count}/{total_count} routes working")
            
            if success_count == total_count:
                print("🎉 All routes are working correctly!")
                return True
            else:
                print("⚠️  Some routes have issues.")
                return False
                
    except Exception as e:
        print(f"❌ Error testing routes: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Flask Route Testing")
    print("=" * 50)
    
    # Test model_info route specifically
    if test_model_info_route():
        print("\n" + "=" * 50)
        # Test all routes if model_info works
        test_all_routes()
    
    print("\n" + "=" * 50)
    print("Testing completed!")
