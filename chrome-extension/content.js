// Content script for Google search results fact-checking
class FactCheckAssistant {
  constructor() {
    this.API_BASE = 'http://localhost:5000';
    this.WEBSITE_BASE = 'http://localhost:3000';
    this.processedResults = new Set();
    this.cache = new Map();
    this.pendingRequests = new Map();
    
    this.init();
  }

  init() {
    // Wait for search results to load
    this.waitForSearchResults();
    
    // Set up mutation observer for dynamic content
    this.setupMutationObserver();
    
    console.log('[Fact Check Assistant] Extension loaded and monitoring Google search results');
  }

  waitForSearchResults() {
    const checkForResults = () => {
      const searchResults = document.querySelectorAll('[data-sokoban-container] a[href*="http"], .g a[href*="http"], .Gx5Zad a[href*="http"]');
      
      if (searchResults.length > 0) {
        this.processSearchResults();
      } else {
        setTimeout(checkForResults, 500);
      }
    };
    
    checkForResults();
  }

  setupMutationObserver() {
    const observer = new MutationObserver((mutations) => {
      let shouldProcess = false;
      
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
          const hasSearchResults = Array.from(mutation.addedNodes).some(node => {
            return node.nodeType === Node.ELEMENT_NODE && 
                   (node.querySelector('a[href*="http"]') || 
                    node.closest('.g') || 
                    node.closest('[data-sokoban-container]'));
          });
          
          if (hasSearchResults) {
            shouldProcess = true;
          }
        }
      });
      
      if (shouldProcess) {
        setTimeout(() => this.processSearchResults(), 1000);
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  async processSearchResults() {
    console.log('[Fact Check Assistant] Processing search results...');
    
    // Multiple selectors to catch different Google layouts
    const resultSelectors = [
      '.g', // Standard search results
      '[data-sokoban-container]', // Some mobile/new layouts
      '.Gx5Zad', // News results
      '.g .yuRUbf', // Updated layout
      '[jscontroller] .yuRUbf' // Alternative layout
    ];
    
    let searchResults = [];
    for (const selector of resultSelectors) {
      const results = document.querySelectorAll(selector);
      if (results.length > 0) {
        searchResults = Array.from(results);
        break;
      }
    }
    
    console.log(`[Fact Check Assistant] Found ${searchResults.length} search results`);
    
    const promises = searchResults.map(async (result, index) => {
      if (this.shouldSkipResult(result)) {
        return;
      }
      
      const linkElement = this.findLinkInResult(result);
      if (!linkElement) {
        return;
      }
      
      const url = linkElement.href;
      if (!this.isValidUrl(url)) {
        return;
      }
      
      const resultId = this.generateResultId(url);
      if (this.processedResults.has(resultId)) {
        return;
      }
      
      this.processedResults.add(resultId);
      
      // Add loading indicator immediately
      this.addLoadingIndicator(result);
      
      try {
        await this.addFactCheckLabel(result, url, index);
      } catch (error) {
        console.error('[Fact Check Assistant] Error processing result:', error);
        this.removeLoadingIndicator(result);
      }
    });
    
    await Promise.all(promises);
  }

  shouldSkipResult(result) {
    // Skip ads and sponsored results
    const isAd = result.querySelector('[data-text-ad], .ads-ad, .commercial-unit-desktop-rhs') ||
                 result.closest('.ads-ad') ||
                 result.textContent.includes('Ad') ||
                 result.textContent.includes('Sponsored');
    
    return isAd;
  }

  findLinkInResult(result) {
    // Try different selectors for finding the main link
    const linkSelectors = [
      'h3 a',
      '.yuRUbf a',
      'a[href*="http"]:first-of-type',
      '.r a',
      'a[data-href]'
    ];
    
    for (const selector of linkSelectors) {
      const link = result.querySelector(selector);
      if (link && link.href) {
        return link;
      }
    }
    
    return null;
  }

  isValidUrl(url) {
    if (!url || url.includes('google.com') || url.includes('youtube.com')) {
      return false;
    }
    
    try {
      const urlObj = new URL(url);
      return ['http:', 'https:'].includes(urlObj.protocol);
    } catch {
      return false;
    }
  }

  generateResultId(url) {
    return btoa(url).slice(0, 16);
  }

  addLoadingIndicator(result) {
    const existing = result.querySelector('.fact-check-loading');
    if (existing) return;
    
    const loading = document.createElement('div');
    loading.className = 'fact-check-loading';
    loading.innerHTML = `
      <div class="fact-check-badge loading">
        <div class="loading-spinner"></div>
        <span>Analyzing...</span>
      </div>
    `;
    
    const titleElement = result.querySelector('h3') || result.querySelector('a');
    if (titleElement) {
      titleElement.parentNode.insertBefore(loading, titleElement.nextSibling);
    }
  }

  removeLoadingIndicator(result) {
    const loading = result.querySelector('.fact-check-loading');
    if (loading) {
      loading.remove();
    }
  }

  async addFactCheckLabel(result, url, index) {
    try {
      // Check cache first
      if (this.cache.has(url)) {
        const cachedData = this.cache.get(url);
        this.renderFactCheckBadge(result, cachedData, url);
        return;
      }
      
      // Check if request is already pending
      if (this.pendingRequests.has(url)) {
        await this.pendingRequests.get(url);
        if (this.cache.has(url)) {
          const cachedData = this.cache.get(url);
          this.renderFactCheckBadge(result, cachedData, url);
        }
        return;
      }
      
      // Add delay to avoid overwhelming the API
      await new Promise(resolve => setTimeout(resolve, index * 200));
      
      // Create and track the request promise
      const requestPromise = this.analyzeUrl(url);
      this.pendingRequests.set(url, requestPromise);
      
      const analysisResult = await requestPromise;
      
      // Remove from pending requests
      this.pendingRequests.delete(url);
      
      if (analysisResult) {
        // Cache the result
        this.cache.set(url, analysisResult);
        
        // Render the badge
        this.renderFactCheckBadge(result, analysisResult, url);
      } else {
        this.removeLoadingIndicator(result);
      }
    } catch (error) {
      console.error('[Fact Check Assistant] Error in addFactCheckLabel:', error);
      this.removeLoadingIndicator(result);
      this.pendingRequests.delete(url);
    }
  }

  async analyzeUrl(url) {
    try {
      console.log(`[Fact Check Assistant] Analyzing: ${url}`);
      
      const response = await fetch(`${this.API_BASE}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
        signal: AbortSignal.timeout(30000) // 30 second timeout
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success && data.data) {
        console.log(`[Fact Check Assistant] Analysis complete for ${url}: ${data.data.label} (${data.data.confidence}%)`);
        return {
          label: data.data.label,
          confidence: data.data.confidence,
          summary: data.data.summary,
          reasoning: data.data.reasoning,
          url: url,
          title: data.data.title
        };
      } else {
        throw new Error(data.error || 'Analysis failed');
      }
    } catch (error) {
      console.error(`[Fact Check Assistant] API error for ${url}:`, error);
      return null;
    }
  }

  renderFactCheckBadge(result, analysisData, originalUrl) {
    // Remove loading indicator
    this.removeLoadingIndicator(result);
    
    // Check if badge already exists
    const existing = result.querySelector('.fact-check-badge:not(.loading)');
    if (existing) {
      return;
    }

    const isTrustworthy = analysisData.label === 'Trustworthy';
    const confidence = analysisData.confidence;
    
    // Determine badge style based on label and confidence
    let badgeClass = 'neutral';
    let badgeIcon = '❓';
    let badgeText = 'Unknown';
    
    if (isTrustworthy) {
      if (confidence >= 75) {
        badgeClass = 'trustworthy-high';
        badgeIcon = '✅';
        badgeText = 'Trustworthy';
      } else {
        badgeClass = 'trustworthy-medium';
        badgeIcon = '✅';
        badgeText = 'Likely Trustworthy';
      }
    } else {
      if (confidence >= 75) {
        badgeClass = 'untrustworthy-high';
        badgeIcon = '❌';
        badgeText = 'Untrustworthy';
      } else {
        badgeClass = 'untrustworthy-medium';
        badgeIcon = '⚠️';
        badgeText = 'Questionable';
      }
    }

    const badge = document.createElement('div');
    badge.className = `fact-check-badge ${badgeClass}`;
    badge.innerHTML = `
      <span class="badge-icon">${badgeIcon}</span>
      <span class="badge-text">${badgeText}</span>
      <span class="badge-confidence">${confidence}%</span>
      <span class="badge-arrow">→</span>
    `;
    
    // Add click handler to open detailed analysis
    badge.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.openDetailedAnalysis(originalUrl, analysisData);
    });
    
    // Add hover tooltip
    badge.title = `Click to see detailed analysis and evidence. Confidence: ${confidence}%`;
    
    // Insert badge after the title or link
    const titleElement = result.querySelector('h3') || result.querySelector('a');
    if (titleElement && titleElement.parentNode) {
      titleElement.parentNode.insertBefore(badge, titleElement.nextSibling);
    }
    
    // Add small delay for smooth appearance
    setTimeout(() => {
      badge.classList.add('visible');
    }, 100);
  }

  openDetailedAnalysis(url, analysisData) {
    // Create URL for detailed analysis page
    const detailUrl = `${this.WEBSITE_BASE}?analyze=${encodeURIComponent(url)}`;
    
    // Store analysis data for the website to use
    chrome.runtime.sendMessage({
      action: 'storeAnalysisData',
      url: url,
      data: analysisData
    });
    
    // Open in new tab
    chrome.runtime.sendMessage({
      action: 'openTab',
      url: detailUrl
    });
  }

  // Public method to manually analyze current page
  analyzeCurrentPage() {
    const currentUrl = window.location.href;
    this.openDetailedAnalysis(currentUrl, null);
  }
}

// Initialize the fact checker
let factChecker;

// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    factChecker = new FactCheckAssistant();
  });
} else {
  factChecker = new FactCheckAssistant();
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'analyzeCurrentPage') {
    factChecker.analyzeCurrentPage();
    sendResponse({ success: true });
  }
});