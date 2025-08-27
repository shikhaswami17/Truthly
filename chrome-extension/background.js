// Background service worker for Fact Check Assistant
class BackgroundService {
  constructor() {
    this.analysisCache = new Map();
    this.setupMessageHandlers();
    this.setupStorageHandlers();
  }

  setupMessageHandlers() {
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      switch (message.action) {
        case 'openTab':
          this.handleOpenTab(message, sendResponse);
          return true;
        
        case 'storeAnalysisData':
          this.handleStoreAnalysisData(message, sendResponse);
          return true;
        
        case 'getAnalysisData':
          this.handleGetAnalysisData(message, sendResponse);
          return true;
        
        case 'checkServiceHealth':
          this.handleHealthCheck(sendResponse);
          return true;
        
        default:
          sendResponse({ error: 'Unknown action' });
          return false;
      }
    });
  }

  setupStorageHandlers() {
    // Clean up old analysis data periodically
    setInterval(() => {
      this.cleanupOldData();
    }, 30 * 60 * 1000); // Every 30 minutes
  }

  handleOpenTab(message, sendResponse) {
    try {
      chrome.tabs.create({ 
        url: message.url,
        active: true
      }, (tab) => {
        if (chrome.runtime.lastError) {
          sendResponse({ 
            success: false, 
            error: chrome.runtime.lastError.message 
          });
        } else {
          sendResponse({ 
            success: true, 
            tabId: tab.id 
          });
        }
      });
    } catch (error) {
      sendResponse({ 
        success: false, 
        error: error.message 
      });
    }
  }

  handleStoreAnalysisData(message, sendResponse) {
    try {
      const { url, data } = message;
      
      // Store in memory cache
      this.analysisCache.set(url, {
        data: data,
        timestamp: Date.now()
      });
      
      // Also store in chrome storage for persistence
      const storageData = {};
      storageData[`analysis_${btoa(url).slice(0, 32)}`] = {
        url: url,
        data: data,
        timestamp: Date.now()
      };
      
      chrome.storage.local.set(storageData, () => {
        if (chrome.runtime.lastError) {
          sendResponse({ 
            success: false, 
            error: chrome.runtime.lastError.message 
          });
        } else {
          sendResponse({ success: true });
        }
      });
    } catch (error) {
      sendResponse({ 
        success: false, 
        error: error.message 
      });
    }
  }

  handleGetAnalysisData(message, sendResponse) {
    try {
      const { url } = message;
      
      // Check memory cache first
      if (this.analysisCache.has(url)) {
        const cached = this.analysisCache.get(url);
        const age = Date.now() - cached.timestamp;
        
        // Return if less than 1 hour old
        if (age < 60 * 60 * 1000) {
          sendResponse({ 
            success: true, 
            data: cached.data,
            source: 'memory_cache'
          });
          return;
        }
      }
      
      // Check chrome storage
      const storageKey = `analysis_${btoa(url).slice(0, 32)}`;
      chrome.storage.local.get([storageKey], (result) => {
        if (chrome.runtime.lastError) {
          sendResponse({ 
            success: false, 
            error: chrome.runtime.lastError.message 
          });
        } else if (result[storageKey]) {
          const stored = result[storageKey];
          const age = Date.now() - stored.timestamp;
          
          // Return if less than 24 hours old
          if (age < 24 * 60 * 60 * 1000) {
            // Update memory cache
            this.analysisCache.set(url, {
              data: stored.data,
              timestamp: stored.timestamp
            });
            
            sendResponse({ 
              success: true, 
              data: stored.data,
              source: 'storage_cache'
            });
          } else {
            // Data too old
            sendResponse({ 
              success: false, 
              error: 'Cached data expired' 
            });
          }
        } else {
          sendResponse({ 
            success: false, 
            error: 'No cached data found' 
          });
        }
      });
    } catch (error) {
      sendResponse({ 
        success: false, 
        error: error.message 
      });
    }
  }

  async handleHealthCheck(sendResponse) {
    try {
      // Check if backend services are available
      const healthStatus = {
        extension: 'healthy',
        timestamp: Date.now(),
        cache: {
          memory_entries: this.analysisCache.size,
          storage_check: 'pending'
        },
        services: {
          backend: 'unknown',
          python: 'unknown'
        }
      };

      // Check chrome storage
      chrome.storage.local.get(null, (items) => {
        const analysisItems = Object.keys(items).filter(key => 
          key.startsWith('analysis_')
        );
        healthStatus.cache.storage_entries = analysisItems.length;
        healthStatus.cache.storage_check = 'completed';
      });

      // Test backend connectivity
      try {
        const response = await fetch('http://localhost:5000/api/health', {
          method: 'GET',
          signal: AbortSignal.timeout(5000)
        });
        
        if (response.ok) {
          healthStatus.services.backend = 'healthy';
        } else {
          healthStatus.services.backend = 'error';
        }
      } catch (error) {
        healthStatus.services.backend = 'unavailable';
      }

      // Test Python service
      try {
        const response = await fetch('http://localhost:5001/health', {
          method: 'GET',
          signal: AbortSignal.timeout(5000)
        });
        
        if (response.ok) {
          healthStatus.services.python = 'healthy';
        } else {
          healthStatus.services.python = 'error';
        }
      } catch (error) {
        healthStatus.services.python = 'unavailable';
      }

      sendResponse({ 
        success: true, 
        health: healthStatus 
      });
    } catch (error) {
      sendResponse({ 
        success: false, 
        error: error.message 
      });
    }
  }

  cleanupOldData() {
    // Clean memory cache (older than 1 hour)
    const oneHourAgo = Date.now() - (60 * 60 * 1000);
    for (const [url, cached] of this.analysisCache.entries()) {
      if (cached.timestamp < oneHourAgo) {
        this.analysisCache.delete(url);
      }
    }

    // Clean storage cache (older than 24 hours)
    chrome.storage.local.get(null, (items) => {
      const oneDayAgo = Date.now() - (24 * 60 * 60 * 1000);
      const keysToRemove = [];
      
      for (const [key, value] of Object.entries(items)) {
        if (key.startsWith('analysis_') && value.timestamp < oneDayAgo) {
          keysToRemove.push(key);
        }
      }
      
      if (keysToRemove.length > 0) {
        chrome.storage.local.remove(keysToRemove);
        console.log(`[Fact Check Assistant] Cleaned up ${keysToRemove.length} old cache entries`);
      }
    });
  }
}

// Initialize the background service
const backgroundService = new BackgroundService();

// Handle extension installation and updates
chrome.runtime.onInstalled.addListener((details) => {
  console.log('[Fact Check Assistant] Extension installed/updated:', details.reason);
  
  if (details.reason === 'install') {
    // Show welcome notification
    chrome.action.setBadgeText({ text: 'NEW' });
    chrome.action.setBadgeBackgroundColor({ color: '#4285f4' });
    
    // Clear badge after 5 minutes
    setTimeout(() => {
      chrome.action.setBadgeText({ text: '' });
    }, 5 * 60 * 1000);
  }
});

// Handle browser startup
chrome.runtime.onStartup.addListener(() => {
  console.log('[Fact Check Assistant] Browser started, extension ready');
});

// Monitor tab updates to clean up cache for closed tabs
chrome.tabs.onRemoved.addListener((tabId, removeInfo) => {
  // Could implement tab-specific cleanup here if needed
});

// Handle context menu (optional feature)
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'factcheck-page',
    title: 'Fact-check this page',
    contexts: ['page'],
    documentUrlPatterns: ['http://*/*', 'https://*/*']
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'factcheck-page') {
    chrome.tabs.sendMessage(tab.id, { action: 'analyzeCurrentPage' });
  }
});