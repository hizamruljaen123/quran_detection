"""
Route untuk halaman demo tajweed-detector
"""

from flask import render_template, url_for, current_app

def register_tajweed_routes(app):
    """Register tajweed related routes"""
    
    @app.route('/tajweed-demo')
    def tajweed_demo():
        """Render halaman demo tajweed detector"""
        return render_template('tajweed_demo.html')
