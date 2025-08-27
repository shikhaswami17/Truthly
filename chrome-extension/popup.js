// Popup script for Fact Check Assistant
class PopupController {
  constructor() {
    this.WEBSITE_BASE = 'http://localhost:3000';
    this.API_BASE = 'http://localhost:5000';
    this.initializeElements();
    this.attachEventListeners();
    this.checkServiceStatus();
  }

  initializeElements() {
    // Buttons
    this.analyzeCurrentBtn = document.getElementById('analyze-current');
    this.analyzeManualBtn = document.getElementById('analyze-manual');
    
    // Input
    this.manualInput = document.getElementById('manual-input');
    
    // Loading indicators
    this.currentLoading = document.getElementById('current-loading');
    this.manualLoading = document.getElementById('manual-loading');
    
    // Message elements
    this.currentError = document.getElementById('current-error');
    this.currentSuccess = document.getElementById('current-success');
    this.manualError = document.getElementById('manual-error');
    this.manualSuccess = document.getElementById('manual-success');
    
    // Status elements
    this.backendStatus = document.getElementById('backend-status');
    this.pythonStatus = document.getElementById('python-status');
  }

  attachEventListeners() {
    // Analyze current page
    this.analyzeCurrentBtn.addEventListener('click', () => {
      this.handleAnalyzeCurrent();
    });
    
    // Analyze manual input
    this.analyzeManualBtn.addEventListener('click', () => {
      this.handleAnalyzeManual();
    });
    
    // Enter key in manual input
    this.manualInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.handleAnalyzeManual();
      }
    });
    
    // Clear input on focus
    this.manualInput.addEventListener('focus', () => {
      this.clearMessages();
    });
  }

  async handleAnalyzeCurrent() {
    try {
      this.showLoading('current');
      this.clearMessages();
      
      // Get current active tab
      const [activeTab] = await chrome.tabs.query({ 
        active: true, 
        currentWindow: true 
      });
      
      if (!activeTab || !activeTab.url) {
        throw new Error('Unable to get current tab URL');
      }
      
      // Check if it's a valid URL for analysis
      if (activeTab.url.startsWith('chrome://') || 
          activeTab.url.startsWith('chrome-extension://') ||
          activeTab.url.startsWith('edge://') ||
          activeTab.url.startsWith('about:')) {
        throw new Error('Cannot analyze browser internal pages');
      }
      
      // Send message to content script to trigger analysis
      try {
        await chrome.tabs.sendMessage(activeTab.id, { 
          action: 'analyzeCurrentPage' 
        });
        
        this.showSuccess('current', 'Opening detailed analysis...');
        
        // Close popup after a short delay
        setTimeout(() => {
          window.close();
        }, 1000);
        
      } catch (contentScriptError) {
        // If content script is not available, open directly
        const analysisUrl = `${this.WEBSITE_BASE}?analyze=${encodeURIComponent(activeTab.url)}`;
        
        await chrome.tabs.create({ 
          url: analysisUrl,
          active: true
        });
        
        this.showSuccess('current', 'Analysis page opened in new tab');
        
        setTimeout(() => {
          window.close();
        }, 1000);
      }
      
    } catch (error) {
      console.error('Error analyzing current page:', error);
      this.showError('current', error.message);
    } finally {
      this.hideLoading('current');
    }
  }

  async handleAnalyzeManual() {
    try {
      const input = this.manualInput.value.trim();
      
      if (!input) {
        this.showError('manual', 'Please enter a URL or search query');
        return;
      }
      
      this.showLoading('manual');
      this.clearMessages();
      
      let analysisUrl;
      
      // Check if input is a URL
      if (this.isValidUrl(input)) {
        // Direct URL analysis
        analysisUrl = `${this.WEBSITE_BASE}?analyze=${encodeURIComponent(input)}`;
      } else {
        // Search query
        analysisUrl = `${this.WEBSITE_BASE}?search=${encodeURIComponent(input)}`;
      }
      
      // Open analysis page
      await chrome.tabs.create({ 
        url: analysisUrl,
        active: true
      });
      
      this.showSuccess('manual', 'Analysis page opened in new tab');
      
      // Clear input and close popup
      this.manualInput.value = '';
      setTimeout(() => {
        window.close();
      }, 1000);
      
    } catch (error) {
      console.error('Error with manual analysis:', error);
      this.showError('manual', error.message);
    } finally {
      this.hideLoading('manual');
    }
  }

  async checkServiceStatus() {
    try {
      // Check backend service
      try {
        const backendResponse = await fetch(`${this.API_BASE}/api/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(5000)
        });
        
        if (backendResponse.ok) {
          this.updateStatus('backend', 'healthy', 'Online');
        } else {
          this.updateStatus('backend', 'error', 'Error');
        }
      } catch (error) {
        this.updateStatus('backend', 'error', 'Offline');
      }
      
      // Check Python service
      try {
        const pythonResponse = await fetch('http://localhost:5001/health', {
          method: 'GET',
          signal: AbortSignal.timeout(5000)
        });
        
        if (pythonResponse.ok) {
          const healthData = await pythonResponse.json();
          const modelCount = healthData.ensemble_info?.total_loaded || 0;
          this.updateStatus('python', 'healthy', `${modelCount} Models`);
        } else {
          this.updateStatus('python', 'error', 'Error');
        }
      } catch (error) {
        this.updateStatus('python', 'error', 'Offline');
      }
      
    } catch (error) {
      console.error('Error checking service status:', error);
    }
  }

  updateStatus(service, status, text) {
    const statusElement = service === 'backend' ? this.backendStatus : this.pythonStatus;
    const indicator = statusElement.querySelector('.status-indicator');
    const textElement = statusElement.querySelector('span:last-child');
    
    // Remove existing status classes
    indicator.classList.remove('status-healthy', 'status-warning', 'status-error');
    
    // Add new status class
    switch (status) {
      case 'healthy':
        indicator.classList.add('status-healthy');
        break;
      case 'warning':
        indicator.classList.add('status-warning');
        break;
      case 'error':
        indicator.classList.add('status-error');
        break;
    }
    
    // Update text
    textElement.innerHTML = `<span class="status-indicator ${indicator.className}"></span>${text}`;
  }

  isValidUrl(string) {
    try {
      const url = new URL(string);
      return ['http:', 'https:'].includes(url.protocol);
    } catch {
      return false;
    }
  }

  showLoading(type) {
    const loadingElement = type === 'current' ? this.currentLoading : this.manualLoading;
    const buttonElement = type === 'current' ? this.analyzeCurrentBtn : this.analyzeManualBtn;
    
    loadingElement.classList.add('show');
    buttonElement.disabled = true;
    buttonElement.style.opacity = '0.6';
  }

  hideLoading(type) {
    const loadingElement = type === 'current' ? this.currentLoading : this.manualLoading;
    const buttonElement = type === 'current' ? this.analyzeCurrentBtn : this.analyzeManualBtn;
    
    loadingElement.classList.remove('show');
    buttonElement.disabled = false;
    buttonElement.style.opacity = '1';
  }

  showError(type, message) {
    const errorElement = type === 'current' ? this.currentError : this.manualError;
    errorElement.textContent = message;
    errorElement.classList.add('show');
  }

  showSuccess(type, message) {
    const successElement = type === 'current' ? this.currentSuccess : this.manualSuccess;
    successElement.textContent = message;
    successElement.classList.add('show');
  }

  clearMessages() {
    [this.currentError, this.currentSuccess, this.manualError, this.manualSuccess].forEach(element => {
      element.classList.remove('show');
      element.textContent = '';
    });
  }
}

// Initialize popup controller when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new PopupController();
});