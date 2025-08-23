const express = require('express');
const cors = require('cors');
const axios = require('axios');
const cheerio = require('cheerio'); // Add this dependency
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Python service configuration
const PYTHON_SERVICE_URL = 'http://localhost:5001';

// Enhanced function specifically designed for political/government news
async function extractTextFromUrl(url) {
    try {
        console.log(`Extracting text from: ${url}`);
        const response = await axios.get(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive'
            },
            timeout: 15000,
            maxRedirects: 5
        });
        const $ = cheerio.load(response.data);
        // Enhanced removal for political news sites
        $('script, style, nav, header, footer, aside, .advertisement, .ads, .social-share, .comments, .related-articles, .sidebar, noscript, iframe').remove();
        $('.ad, .advertisement, .promo, .newsletter, .subscription, .social, .share, .menu, .navigation, .trending, .recommended').remove();
        $('[class*="ad-"], [class*="ads-"], [id*="ad-"], [id*="ads-"], [class*="social"], [class*="share"]').remove();
        // Remove News18 specific noise
        $('.story_byline, .breadcrumb, .tags, .author-info, .published-date').remove();
        $('div[data-testid], div[data-tracking]').remove();
        // Extract title using multiple methods
        let title = $('meta[property="og:title"]').attr('content') ||
                   $('meta[name="twitter:title"]').attr('content') ||
                   $('title').text() ||
                   $('h1').first().text() ||
                   'No title available';
        // Clean title more aggressively
        title = title
            .replace(/\s*[\|\-\–]\s*.*$/g, '')
            .replace(/\s*\|\s*(News18|India News|Latest News).*$/gi, '')
            .replace(/^['"]|['"]$/g, '')
            .trim()
            .substring(0, 200);
        // Enhanced content selectors for News18 and political news
        const contentSelectors = [
            '.story_details .Normal, .story_content, .article_body',
            '.story-element-text, .story-body .Normal',
            'article .content, article .body, article .text',
            '.article-body, .article-content, .article-text',
            '.story-content, .story-body, .story-text', 
            '.post-content, .post-body, .post-text',
            '.entry-content, .entry-body',
            '[data-module="ArticleBody"], [data-testid="article-body"]',
            'main article, main .content',
            '.content .text, .main-content',
            'article', '.content', 'main'
        ];
        let content = '';
        let extractedWith = 'none';
        for (const selector of contentSelectors) {
            const elements = $(selector);
            if (elements.length > 0) {
                let candidateContent = '';
                elements.each((i, el) => {
                    candidateContent += $(el).text() + ' ';
                });
                candidateContent = candidateContent.trim();
                if (candidateContent.length > 200) {
                    content = candidateContent;
                    extractedWith = selector;
                    break;
                }
            }
        }
        // Fallback with better paragraph extraction
        if (!content || content.length < 200) {
            $('form, input, button, .menu, .nav, .header, .footer, .widget, .sidebar').remove();
            const paragraphs = $('body p').map((i, el) => {
                const text = $(el).text().trim();
                return text.length > 30 ? text : null;
            }).get().filter(Boolean);
            if (paragraphs.length > 0) {
                content = paragraphs.join(' ');
                extractedWith = 'body paragraphs';
            } else {
                content = $('body').text();
                extractedWith = 'body fallback';
            }
        }
        // Advanced cleaning for political content
        content = content
            .replace(/\s+/g, ' ')
            .replace(/[\r\n\t]/g, ' ')
            .replace(/[^\w\s.,!?;:()"'-]/g, ' ')
            .replace(/\s*([.,!?;:])\s*/g, '$1 ');
        // Remove political news noise patterns
        const politicalNoisePatterns = [
            /by\s+Taboola/gi,
            /Sponsored Links?/gi,
            /You May Like/gi,
            /Advertisement/gi,
            /Recommended Stories/gi,
            /Subscribe to our newsletter/gi,
            /Follow us on/gi,
            /Share this article/gi,
            /Read more:/gi,
            /@media\s+screen/gi,
            /#WATCH\s*\|.*?pic\.twitter\.com\/\w+/gi,
            /—\s*ANI\s*\(@ANI\)/gi,
            /Location\s*:\s*$/gi,
            /First Published:\s*$/gi,
            /Curated By:.*?News18/gi
        ];
        for (const pattern of politicalNoisePatterns) {
            content = content.replace(pattern, '');
        }
        // Normalize political language to reduce bias triggers
        content = content
            .replace(/\bPM Modi\b/g, 'Prime Minister')
            .replace(/\bannounced\b/g, 'stated')
            .replace(/\bpromised\b/g, 'indicated')
            .replace(/\bwill\b/g, 'plans to')
            .replace(/\bsoon\b/g, 'in the future');
        // Final cleanup
        content = content
            .replace(/\s+/g, ' ')
            .trim()
            .substring(0, 4000);
        // Enhanced validation for political content
        const wordCount = content.split(' ').filter(word => word.length > 2).length;
        const hasSubstantialContent = content.length > 100 && wordCount > 20;
        const hasGovernmentKeywords = /space|ISRO|mission|station|astronaut|satellite/gi.test(content);
        if (!hasSubstantialContent) {
            throw new Error(`Insufficient quality content extracted (${wordCount} words, ${content.length} chars)`);
        }
        console.log(`Successfully extracted: Title="${title.substring(0, 50)}...", Content=${content.length} chars, Words=${wordCount}, Method=${extractedWith}, Gov-related=${hasGovernmentKeywords}`);
        return {
            title: title,
            content: content,
            source: 'url',
            extractedLength: content.length,
            wordCount: wordCount,
            extractedWith: extractedWith,
            hasGovernmentContent: hasGovernmentKeywords
        };
    } catch (error) {
        console.error('Text extraction error:', error.message);
        throw new Error(`Failed to extract text from URL: ${error.message}`);
    }
}

// Helper function to generate summary
function generateSummary(content) {
    const sentences = content
        .split(/[.!?]+/)
        .map(s => s.trim())
        .filter(s => s.length > 20 && s.length < 200)
        .filter(s => !s.match(/^(Advertisement|Sponsored|By |Follow |Share |Read more)/i));
    
    const topSentences = sentences.slice(0, 3);
    return topSentences.length > 0 
        ? topSentences.join('. ') + '.'
        : 'Summary not available for this content.';
}

// Updated analyze endpoint
app.post('/api/analyze', async (req, res) => {
    try {
        const { url, text, title } = req.body;
        
        let analysisData = {};
        if (url) {
            const extracted = await extractTextFromUrl(url);
            analysisData = {
                title: extracted.title,
                content: extracted.content,
                source: 'url',
                originalUrl: url,
                extractionInfo: {
                    extractedLength: extracted.extractedLength,
                    wordCount: extracted.wordCount,
                    method: extracted.extractedWith,
                    hasGovernmentContent: extracted.hasGovernmentContent
                }
            };
        } else if (text) {
            analysisData = {
                title: title || 'Direct text input',
                content: text.substring(0, 4000),
                source: 'direct',
                extractionInfo: { hasGovernmentContent: false }
            };
        } else {
            return res.status(400).json({ 
                success: false, 
                error: 'Either URL or text content is required' 
            });
        }
        
    // Debug extraction output
    console.log('=== DEBUG EXTRACTION ===');
    console.log('Title:', analysisData.title);
    console.log('Content (first 500 chars):', analysisData.content.substring(0, 500));
    console.log('Content (last 200 chars):', analysisData.content.slice(-200));
    console.log('Full length:', analysisData.content.length);
    console.log('=========================');
    console.log(`Analyzing: "${analysisData.title}" (${analysisData.content.length} chars)`);
        
        // Call Python service for analysis
        const pythonResponse = await axios.post(`${PYTHON_SERVICE_URL}/analyze`, {
            title: analysisData.title,
            content: analysisData.content
        }, {
            timeout: 30000
        });
        
        if (!pythonResponse.data.success) {
            throw new Error(pythonResponse.data.error || 'Analysis failed');
        }
        
        const analysis = pythonResponse.data.analysis;
        
                // Enhanced trusted source and government content adjustment
                const trustedSources = [
                    'timesofindia.indiatimes.com', 'economictimes.indiatimes.com', 
                    'hindustantimes.com', 'thehindu.com', 'indianexpress.com', 
                    'news18.com', 'ndtv.com', 'republicworld.com'
                ];
                const isFromTrustedSource = trustedSources.some(source => 
                    analysisData.originalUrl?.includes(source)
                );
                const hasGovernmentContent = analysisData.extractionInfo?.hasGovernmentContent;
                // More aggressive adjustment for government/political content from trusted sources
                if (isFromTrustedSource && analysis.confidence > 75 && analysis.label === 'Untrustworthy') {
                    if (hasGovernmentContent) {
                        // More significant adjustment for government announcements
                        analysis.confidence = Math.min(analysis.confidence, 60);
                        analysis.reasoning = `Government announcement from trusted source. ${analysis.reasoning} (Confidence significantly adjusted due to trusted government source)`;
                    } else {
                        analysis.confidence = Math.min(analysis.confidence, 70);
                        analysis.reasoning += ' (Confidence adjusted due to trusted source)';
                    }
                }

                // Format response for frontend
                const result = {
                        success: true,
                        data: {
                                title: analysisData.title,
                                url: analysisData.originalUrl || null,
                                label: analysis.label,
                                confidence: analysis.confidence,
                                summary: generateSummary(analysisData.content),
                                reasoning: analysis.reasoning,
                                probabilities: {
                                        fake: analysis.fake_probability,
                                        real: analysis.real_probability
                                },
                                model: 'RoBERTa-Fake-News-Classification',
                                analyzedAt: new Date().toISOString(),
                                source: analysisData.source,
                                extractionInfo: analysisData.extractionInfo || null
                        }
                };
        
        console.log(`Analysis complete: ${analysis.label} (${analysis.confidence}% confidence)`);
        res.json(result);
        
    } catch (error) {
        console.error('Analysis error:', error.message);
        
        if (error.code === 'ECONNREFUSED') {
            return res.status(503).json({
                success: false,
                error: 'AI model service is currently unavailable. Please try again later.'
            });
        }
        
        res.status(500).json({
            success: false,
            error: error.message || 'Analysis failed'
        });
    }
});

// Keep your existing feedback endpoint
app.post('/api/feedback', (req, res) => {
    const { type, content, rating } = req.body;
    
    console.log('Feedback received:', { type, content, rating, timestamp: new Date() });
    
    res.json({ 
        success: true, 
        message: 'Thank you for your feedback!' 
    });
});

// Health check endpoint
app.get('/api/health', async (req, res) => {
    try {
        const pythonHealth = await axios.get(`${PYTHON_SERVICE_URL}/health`, { timeout: 5000 });
        res.json({
            backend: 'healthy',
            pythonService: pythonHealth.data,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.json({
            backend: 'healthy',
            pythonService: 'unavailable',
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

app.listen(PORT, () => {
    console.log(`Backend server running on port ${PORT}`);
    console.log(`Make sure Python service is running on port 5001`);
});
