// Main JavaScript file for Quran Verse Detection Web App

document.addEventListener('DOMContentLoaded', function() {
    // Initialize animations
    initAnimations();
    
    // Initialize tooltips
    initTooltips();
    
    // Initialize audio features
    initAudioFeatures();
    
    // Initialize search features
    initSearchFeatures();
    
    // Initialize progress tracking
    initProgressTracking();
});

// Animation Functions
function initAnimations() {
    // Animate counter numbers
    const counters = document.querySelectorAll('.stat-card h3, .display-4');
    const observerOptions = {
        threshold: 0.5
    };
    
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateNumber(entry.target);
                counterObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    counters.forEach(counter => {
        counterObserver.observe(counter);
    });
    
    // Animate cards on scroll
    const cards = document.querySelectorAll('.card, .feature-card');
    const cardObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        cardObserver.observe(card);
    });
}

function animateNumber(element) {
    const text = element.textContent;
    const number = parseFloat(text.replace(/[^\d.]/g, ''));
    const suffix = text.replace(/[\d.]/g, '');
    
    if (isNaN(number)) return;
    
    let current = 0;
    const increment = number / 50;
    const timer = setInterval(() => {
        current += increment;
        if (current >= number) {
            current = number;
            clearInterval(timer);
        }
        element.textContent = Math.round(current) + suffix;
    }, 30);
}

// Tooltip Initialization
function initTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Audio Features
function initAudioFeatures() {
    // Audio file validation
    const audioInputs = document.querySelectorAll('input[type="file"][accept*="audio"]');
    audioInputs.forEach(input => {
        input.addEventListener('change', validateAudioFile);
    });
    
    // Audio preview functionality
    const audioFiles = document.querySelectorAll('input[type="file"]');
    audioFiles.forEach(input => {
        input.addEventListener('change', previewAudio);
    });
}

function validateAudioFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const validTypes = ['audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/x-m4a'];
    const maxSize = 50 * 1024 * 1024; // 50MB
    
    if (!validTypes.includes(file.type) && !file.name.toLowerCase().match(/\.(mp3|wav|m4a)$/)) {
        showNotification('Invalid file type. Please select MP3, WAV, or M4A files.', 'warning');
        event.target.value = '';
        return;
    }
    
    if (file.size > maxSize) {
        showNotification('File too large. Maximum size is 50MB.', 'warning');
        event.target.value = '';
        return;
    }
    
    showNotification('Audio file loaded successfully!', 'success');
}

function previewAudio(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const audioPreview = document.getElementById('audioPreview');
    if (audioPreview) {
        const url = URL.createObjectURL(file);
        audioPreview.src = url;
        audioPreview.style.display = 'block';
        
        // Clean up URL when audio is loaded
        audioPreview.addEventListener('loadeddata', () => {
            URL.revokeObjectURL(url);
        });
    }
}

// Search Features
function initSearchFeatures() {
    const searchInputs = document.querySelectorAll('input[type="search"], input[placeholder*="search"]');
    searchInputs.forEach(input => {
        input.addEventListener('input', debounce(performSearch, 300));
    });
}

function performSearch(event) {
    const searchTerm = event.target.value.toLowerCase();
    const searchableElements = document.querySelectorAll('.verse-item, .searchable');
    
    let visibleCount = 0;
    searchableElements.forEach(element => {
        const text = element.textContent.toLowerCase();
        const isVisible = text.includes(searchTerm);
        
        element.style.display = isVisible ? 'block' : 'none';
        if (isVisible) visibleCount++;
    });
    
    // Show/hide no results message
    const noResults = document.getElementById('noResults');
    if (noResults) {
        noResults.style.display = visibleCount === 0 ? 'block' : 'none';
    }
}

// Progress Tracking
function initProgressTracking() {
    // Track page views
    trackPageView();
    
    // Track user interactions
    trackInteractions();
    
    // Auto-save form data
    initAutoSave();
}

function trackPageView() {
    const page = window.location.pathname;
    const timestamp = new Date().toISOString();
    
    // Store in localStorage for analytics
    const analytics = JSON.parse(localStorage.getItem('quran_analytics') || '{}');
    analytics.pageViews = analytics.pageViews || [];
    analytics.pageViews.push({ page, timestamp });
    
    // Keep only last 100 page views
    if (analytics.pageViews.length > 100) {
        analytics.pageViews = analytics.pageViews.slice(-100);
    }
    
    localStorage.setItem('quran_analytics', JSON.stringify(analytics));
}

function trackInteractions() {
    // Track button clicks
    document.addEventListener('click', (event) => {
        if (event.target.matches('button, .btn')) {
            const action = event.target.textContent.trim() || event.target.className;
            console.log('Button clicked:', action);
        }
    });
    
    // Track form submissions
    document.addEventListener('submit', (event) => {
        const formId = event.target.id || event.target.className;
        console.log('Form submitted:', formId);
    });
}

function initAutoSave() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('change', () => {
                saveFormData(form);
            });
        });
        
        // Load saved data
        loadFormData(form);
    });
}

function saveFormData(form) {
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    const formId = form.id || 'default_form';
    localStorage.setItem(`form_${formId}`, JSON.stringify(data));
}

function loadFormData(form) {
    const formId = form.id || 'default_form';
    const savedData = localStorage.getItem(`form_${formId}`);
    
    if (savedData) {
        const data = JSON.parse(savedData);
        Object.keys(data).forEach(key => {
            const input = form.querySelector(`[name="${key}"]`);
            if (input) {
                input.value = data[key];
            }
        });
    }
}

// Utility Functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function showNotification(message, type = 'info', duration = 5000) {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification-toast');
    existingNotifications.forEach(notification => notification.remove());
    
    // Create notification
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} notification-toast position-fixed`;
    notification.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        max-width: 400px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border-radius: 8px;
        animation: slideInRight 0.3s ease;
    `;
    
    notification.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <span>${message}</span>
            <button type="button" class="btn-close ms-2" onclick="this.parentElement.parentElement.remove()"></button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after duration
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }
    }, duration);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }
}

// API Helper Functions
async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const config = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, config);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        } else {
            return await response.text();
        }
    } catch (error) {
        console.error('API request failed:', error);
        showNotification(`Request failed: ${error.message}`, 'danger');
        throw error;
    }
}

// Prediction Helper
async function predictVerse(audioFile) {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Prediction failed: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification(`Prediction failed: ${error.message}`, 'danger');
        throw error;
    }
}

// Training Status Helper
async function checkTrainingStatus() {
    try {
        const status = await apiRequest('/api/training_status');
        return status;
    } catch (error) {
        console.error('Failed to check training status:', error);
        return { status: 'error', message: 'Failed to check status' };
    }
}

// Keyboard Shortcuts
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (event) => {
        // Ctrl+U for upload page
        if (event.ctrlKey && event.key === 'u') {
            event.preventDefault();
            window.location.href = '/upload';
        }
        
        // Ctrl+D for dataset page
        if (event.ctrlKey && event.key === 'd') {
            event.preventDefault();
            window.location.href = '/dataset';
        }
        
        // Ctrl+M for model info page
        if (event.ctrlKey && event.key === 'm') {
            event.preventDefault();
            window.location.href = '/model_info';
        }
        
        // Ctrl+V for verses page
        if (event.ctrlKey && event.key === 'v') {
            event.preventDefault();
            window.location.href = '/verses';
        }
        
        // Escape to go home
        if (event.key === 'Escape') {
            window.location.href = '/';
        }
    });
}

// Theme Management
function initThemeManagement() {
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
    const currentTheme = localStorage.getItem('theme');
    
    if (currentTheme === 'dark' || (!currentTheme && prefersDarkScheme.matches)) {
        document.body.classList.add('dark-theme');
    }
    
    // Listen for theme changes
    prefersDarkScheme.addEventListener('change', (e) => {
        if (!localStorage.getItem('theme')) {
            document.body.classList.toggle('dark-theme', e.matches);
        }
    });
}

// Performance Monitoring
function initPerformanceMonitoring() {
    // Monitor page load performance
    window.addEventListener('load', () => {
        const perfData = performance.getEntriesByType('navigation')[0];
        console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
    });
    
    // Monitor memory usage (if available)
    if ('memory' in performance) {
        setInterval(() => {
            const memory = performance.memory;
            console.log('Memory usage:', {
                used: Math.round(memory.usedJSHeapSize / 1048576) + ' MB',
                total: Math.round(memory.totalJSHeapSize / 1048576) + ' MB',
                limit: Math.round(memory.jsHeapSizeLimit / 1048576) + ' MB'
            });
        }, 30000); // Check every 30 seconds
    }
}

// Error Handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    showNotification('An unexpected error occurred. Please refresh the page.', 'danger');
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    showNotification('A network error occurred. Please check your connection.', 'warning');
});

// Initialize additional features
document.addEventListener('DOMContentLoaded', function() {
    initKeyboardShortcuts();
    initThemeManagement();
    initPerformanceMonitoring();
});

// Export functions for global use
window.QuranApp = {
    showNotification,
    formatFileSize,
    formatTime,
    predictVerse,
    checkTrainingStatus,
    apiRequest
};
