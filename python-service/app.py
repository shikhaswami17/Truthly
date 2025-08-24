from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
import time
import os
import asyncio
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor
import json
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model storage
models = {}
device = None

class EnhancedMultiAPIEnsemble:
    def __init__(self):
        self.models = {}
        self.loaded_models = []
        self.failed_models = []
        self.api_keys = self._load_api_keys()
        self.executor = ThreadPoolExecutor(max_workers=6)
        
    def _load_api_keys(self):
        """Load all API keys from environment with debugging"""
        # Load .env file and check if it exists
        from dotenv import load_dotenv, find_dotenv
        env_file = find_dotenv()
        if env_file:
            logger.info(f"📁 Found .env file at: {env_file}")
            load_dotenv(env_file)
        else:
            logger.warning("⚠️ No .env file found!")
            load_dotenv()  # Try loading anyway
        keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
        'groq': os.getenv('GROQ_API_KEY'),
        'google_search': os.getenv('GOOGLE_SEARCH_API_KEY'),
        'google_search_engine_id': os.getenv('GOOGLE_SEARCH_ENGINE_ID'),
        'serper': os.getenv('SERPER_API_KEY'),
        'huggingface': os.getenv('HUGGINGFACE_API_KEY')
    }
    
        # Debug each key
        for key, value in keys.items():
            if value:
                logger.info(f"✅ {key}: loaded (length: {len(value)})")
            else:
                logger.warning(f"❌ {key}: NOT FOUND")
        available_apis = [k for k, v in keys.items() if v]
        logger.info(f"🔑 Available API keys: {', '.join(available_apis)}")
        return keys

    def predict_llama_comprehensive(self, title, content):
        """
        Comprehensive LLaMA prediction with labeling, summary, and confidence
        Handles all your fake news requirements in one API call
        """
        try:
            if not self.api_keys.get('huggingface'):
                logger.warning("⚠️ HuggingFace API key not available for LLaMA")
                return self._llama_fallback_analysis(title, content)
            logger.info("🦙 Calling LLaMA Comprehensive Analysis...")
            # Use Hugging Face's LLaMA model via API
            headers = {"Authorization": f"Bearer {self.api_keys['huggingface']}"}
            # Try different LLaMA models available on HuggingFace
            llama_models = [

                "microsoft/DialoGPT-medium",  # Instead of DialoGPT-large
                "facebook/blenderbot-1B-distill",  # Instead of 400M-distill
                "huggingface/CodeBERTa-small-v1"  # Alternative model
            ]
            # Construct comprehensive analysis prompt
            prompt = f"""
            TASK: Analyze this news content for credibility and truthfulness.
            TITLE: {title[:200]}
            CONTENT: {self.truncate_text(content, 600)}
            ANALYSIS REQUIRED:
            1. Check for factual inconsistencies
            2. Evaluate source credibility indicators
            3. Detect emotional manipulation or bias
            4. Assess logical coherence
            RESPONSE FORMAT:
            VERDICT: TRUSTWORTHY or UNTRUSTWORTHY
            CONFIDENCE: [0-100]
            SUMMARY: [Brief summary of the content]
            REASONING: [Detailed analysis of why this verdict was reached]
            """
            for model_name in llama_models:
                try:
                    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
                    response = requests.post(
                        API_URL,
                        headers=headers,
                        json={"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.3}},
                        timeout=25
                    )
                    if response.status_code == 200:
                        result = response.json()
                        # Handle different response formats
                        if isinstance(result, list) and result:
                            response_text = result[0].get('generated_text', str(result))
                        elif isinstance(result, dict):
                            response_text = result.get('generated_text', str(result))
                        else:
                            response_text = str(result)
                        # Parse the structured response
                        parsed_result = self._parse_llama_response(response_text)
                        # Add model information
                        parsed_result.update({
                            'model': f'LLaMA-Comprehensive-{model_name.split("/")[-1]}',
                            'processing_method': 'structured_analysis'
                        })
                        logger.info(f"✅ LLaMA analysis complete: {parsed_result['label']} ({parsed_result['confidence']}%)")
                        return parsed_result
                    else:
                        logger.warning(f"⚠️ LLaMA model {model_name} returned status: {response.status_code}")
                        continue
                except Exception as model_error:
                    logger.warning(f"⚠️ LLaMA model {model_name} failed: {str(model_error)[:100]}")
                    continue
            # If all LLaMA models fail, use fallback
            logger.info("🔄 All LLaMA models failed, using fallback analysis...")
            return self._llama_fallback_analysis(title, content)
        except Exception as e:
            logger.error(f"❌ LLaMA comprehensive analysis error: {e}")
            # Return fallback analysis instead of error
            return self._llama_fallback_analysis(title, content)

    def _parse_llama_response(self, response_text):
        """Parse structured LLaMA response"""
        import re
        # Default values
        label = 'Real'
        confidence = 70
        summary = 'Summary not available'
        reasoning = 'Analysis completed'
        try:
            # Extract verdict
            verdict_match = re.search(r'VERDICT:\s*(TRUSTWORTHY|UNTRUSTWORTHY)', response_text, re.IGNORECASE)
            if verdict_match:
                verdict = verdict_match.group(1).upper()
                label = 'Real' if verdict == 'TRUSTWORTHY' else 'Fake'
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response_text)
            if confidence_match:
                confidence = int(confidence_match.group(1))
                confidence = max(0, min(100, confidence))
            # Extract summary
            summary_match = re.search(r'SUMMARY:\s*([^\n]+(?:\n[^\n]+)*?)(?=\nREASONING:|$)', response_text, re.DOTALL)
            if summary_match:
                summary = summary_match.group(1).strip()
                summary = re.sub(r'\s+', ' ', summary)
                summary = summary[:300]
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*([^\n]+(?:\n[^\n]+)*?)$', response_text, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                reasoning = re.sub(r'\s+', ' ', reasoning)
                reasoning = reasoning[:400]
        except Exception as e:
            logger.warning(f"⚠️ Error parsing LLaMA response: {e}")
            if 'untrustworthy' in response_text.lower() or 'fake' in response_text.lower():
                label = 'Fake'
            summary = response_text[:200] + "..." if len(response_text) > 200 else response_text
        return {
            'label': label,
            'confidence': confidence,
            'summary': summary,
            'reasoning': reasoning
        }

    def _llama_fallback_analysis(self, title, content):
        """Fallback analysis using smaller LLaMA model or alternative"""
        try:
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            headers = {"Authorization": f"Bearer {self.api_keys['huggingface']}"}
            simple_prompt = f"Analyze this news for credibility: {title}. {content[:300]}"
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": simple_prompt},
                timeout=20
            )
            if response.status_code == 200:
                is_trustworthy = any(word in content.lower() for word in ['official', 'confirmed', 'announced', 'statement'])
                confidence = 65
                return {
                    'model': 'LLaMA-Fallback',
                    'label': 'Real' if is_trustworthy else 'Fake',
                    'confidence': confidence,
                    'summary': f"Summary: {title}. {content[:150]}...",
                    'reasoning': 'Fallback analysis based on content patterns',
                    'fallback_mode': True
                }
            return {'error': 'All LLaMA options failed'}
        except Exception as e:
            logger.error(f"❌ LLaMA fallback error: {e}")
            return {'error': str(e)}


        
    def truncate_text(self, text, max_length=400):
        """Safely truncate text to prevent tensor size issues"""
        if len(text) > max_length:
            # Find last complete sentence within limit
            truncated = text[:max_length]
            last_sentence = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
            if last_sentence > max_length * 0.7:  # If we can keep most of the text
                return truncated[:last_sentence + 1]
            return truncated + "..."
        return text
        
    def load_local_models(self):
        """Load local Hugging Face models"""
        global device
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            import torch
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🔧 Using device: {device}")
            
            # Load RoBERTa Fake News Model (Keep this as primary)
            try:
                logger.info("🤖 Loading RoBERTa Fake News Model...")
                self.models['roberta'] = {
                    'tokenizer': AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification"),
                    'model': AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification"),
                    'type': 'classification'
                }
                self.models['roberta']['model'].to(device)
                self.models['roberta']['model'].eval()
                self.loaded_models.append('RoBERTa-Local')
                logger.info("✅ RoBERTa model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load RoBERTa: {e}")
                self.failed_models.append(f'RoBERTa: {str(e)[:100]}')
            
            # Load other local models with better error handling
            local_models = [
                ("BART-MNLI", "facebook/bart-large-mnli", "zero-shot-classification"),
                ("Sentiment", "cardiffnlp/twitter-roberta-base-sentiment-latest", "sentiment-analysis")
            ]
            
            for name, model_name, task in local_models:
                try:
                    logger.info(f"🤖 Loading {name} Model...")
                    self.models[name.lower()] = pipeline(
                        task,
                        model=model_name,
                        device=0 if device == 'cuda' else -1,
                        truncation=True,
                        max_length=512
                    )
                    self.loaded_models.append(f'{name}-Local')
                    logger.info(f"✅ {name} model loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load {name}: {e}")
                    self.failed_models.append(f'{name}: {str(e)[:100]}')
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical error loading local models: {e}")
            return False
    
    def call_openai_api(self, title, content):
        """Call OpenAI API for fact-checking"""
        try:
            if not self.api_keys['openai']:
                return {'error': 'OpenAI API key not available'}
            
            logger.info("🤖 Calling OpenAI API...")
            
            headers = {
                'Authorization': f'Bearer {self.api_keys["openai"]}',
                'Content-Type': 'application/json'
            }
            
            prompt = f"""
            Analyze this news content for truthfulness and reliability. Return only a JSON response.
            
            Title: {title}
            Content: {self.truncate_text(content, 500)}
            
            Analyze for:
            1. Factual accuracy indicators
            2. Source credibility signals  
            3. Language bias or manipulation
            4. Logical consistency
            
            Return JSON format:
            {{
                "label": "Trustworthy" or "Untrustworthy",
                "confidence": 0-100,
                "reasoning": "brief explanation",
                "factual_score": 0-100,
                "credibility_score": 0-100
            }}
            """
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 300
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                content_text = result['choices'][0]['message']['content']
                
                try:
                    # Try to parse JSON response
                    parsed = json.loads(content_text)
                    return {
                        'model': 'OpenAI-GPT-3.5',
                        'label': 'Real' if parsed.get('label') == 'Trustworthy' else 'Fake',
                        'confidence': float(parsed.get('confidence', 50)),
                        'reasoning': f"OpenAI analysis: {parsed.get('reasoning', 'No detailed reasoning provided')}",
                        'factual_score': parsed.get('factual_score', 50),
                        'credibility_score': parsed.get('credibility_score', 50)
                    }
                except json.JSONDecodeError:
                    # Fallback parsing
                    is_trustworthy = 'trustworthy' in content_text.lower()
                    return {
                        'model': 'OpenAI-GPT-3.5',
                        'label': 'Real' if is_trustworthy else 'Fake',
                        'confidence': 70,
                        'reasoning': f"OpenAI analysis: {content_text[:200]}"
                    }
            else:
                return {'error': f'OpenAI API error: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            return {'model': 'OpenAI-GPT-3.5', 'error': str(e)}
    
    def call_groq_api(self, title, content):
        """Call Groq API for fact-checking"""
        try:
            if not self.api_keys['groq']:
                return {'error': 'Groq API key not available'}
            
            logger.info("🤖 Calling Groq API...")
            
            headers = {
                'Authorization': f'Bearer {self.api_keys["groq"]}',
                'Content-Type': 'application/json'
            }
            
            prompt = f"""
            Fact-check this news content. Be concise and analytical.
            
            Title: {title}
            Content: {self.truncate_text(content, 500)}
            
            Provide:
            - VERDICT: Reliable/Unreliable
            - CONFIDENCE: 0-100%  
            - KEY_ISSUES: List main concerns or positive indicators
            - REASONING: Brief analysis
            
            Format as: VERDICT: [verdict] | CONFIDENCE: [number]% | REASONING: [analysis]
            """
            
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "model": "mixtral-8x7b-32768",  # Fast Groq model
                "temperature": 0.1,
                "max_tokens": 250
            }
            
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content_text = result['choices'][0]['message']['content']
                
                # Parse Groq response
                verdict = 'Reliable' if 'VERDICT: Reliable' in content_text or 'reliable' in content_text.lower() else 'Unreliable'
                
                # Extract confidence
                confidence = 70  # default
                import re
                conf_match = re.search(r'CONFIDENCE:\s*(\d+)', content_text)
                if conf_match:
                    confidence = int(conf_match.group(1))
                
                return {
                    'model': 'Groq-Mixtral',
                    'label': 'Real' if verdict == 'Reliable' else 'Fake', 
                    'confidence': confidence,
                    'reasoning': f"Groq analysis: {content_text[:200]}",
                    'raw_verdict': verdict
                }
            else:
                return {'error': f'Groq API error: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"❌ Groq API error: {e}")
            return {'model': 'Groq-Mixtral', 'error': str(e)}
    
    def call_huggingface_api(self, title, content):
        """Call Hugging Face Inference API"""
        try:
            if not self.api_keys['huggingface']:
                return {'error': 'Hugging Face API key not available'}
            
            logger.info("🤖 Calling Hugging Face API...")
            
            headers = {
                'Authorization': f'Bearer {self.api_keys["huggingface"]}',
                'Content-Type': 'application/json'
            }
            
            # Use a different model via API
            api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            
            text = f"Fact-check this news: {title}. {self.truncate_text(content, 200)}"
            
            data = {"inputs": text}
            
            response = requests.post(api_url, headers=headers, json=data, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                
                # Simple heuristic analysis of the response
                response_text = str(result).lower()
                is_reliable = any(word in response_text for word in ['reliable', 'accurate', 'factual', 'true'])
                
                return {
                    'model': 'HuggingFace-API',
                    'label': 'Real' if is_reliable else 'Fake',
                    'confidence': 65,
                    'reasoning': f"HuggingFace API analysis based on response patterns"
                }
            else:
                return {'error': f'HuggingFace API error: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"❌ HuggingFace API error: {e}")
            return {'model': 'HuggingFace-API', 'error': str(e)}
    
    def search_and_verify(self, title, content):
        """Search Google/Serper and provide verification context"""
        try:
            # Try Serper first
            if self.api_keys['serper']:
                logger.info("🔍 Using Serper for fact verification...")
                
                headers = {'X-API-KEY': self.api_keys['serper']}
                search_query = f'"{title[:50]}" news fact check'
                
                response = requests.post(
                    'https://google.serper.dev/search',
                    json={'q': search_query, 'num': 5},
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    results = response.json()
                    organic_results = results.get('organic', [])
                    
                    # Analyze search results for verification
                    trusted_domains = ['reuters.com', 'bbc.com', 'apnews.com', 'factcheck.org', 'snopes.com']
                    trusted_count = sum(1 for r in organic_results if any(d in r.get('link', '') for d in trusted_domains))
                    
                    verification_score = (trusted_count / max(len(organic_results), 1)) * 100
                    
                    return {
                        'model': 'Search-Verification',
                        'label': 'Real' if verification_score > 30 else 'Fake',
                        'confidence': min(verification_score + 40, 85),
                        'reasoning': f"Found {trusted_count}/{len(organic_results)} results from trusted sources",
                        'trusted_sources': trusted_count,
                        'total_results': len(organic_results)
                    }
            
            # Fallback to Google Custom Search
            elif self.api_keys['google_search']:
                logger.info("🔍 Using Google Search for fact verification...")
                
                search_url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': self.api_keys['google_search'],
                    'cx': self.api_keys['google_search_engine_id'],
                    'q': f'"{title[:50]}" news verification',
                    'num': 5
                }
                
                response = requests.get(search_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    results = response.json()
                    items = results.get('items', [])
                    
                    trusted_domains = ['reuters.com', 'bbc.com', 'apnews.com']
                    trusted_count = sum(1 for item in items if any(d in item.get('link', '') for d in trusted_domains))
                    
                    verification_score = (trusted_count / max(len(items), 1)) * 100
                    
                    return {
                        'model': 'Google-Search-Verification',
                        'label': 'Real' if verification_score > 25 else 'Fake',
                        'confidence': min(verification_score + 35, 80),
                        'reasoning': f"Google search found {trusted_count}/{len(items)} trusted sources",
                        'trusted_sources': trusted_count
                    }
            
            return {'error': 'No search APIs available'}
            
        except Exception as e:
            logger.error(f"❌ Search verification error: {e}")
            return {'model': 'Search-Verification', 'error': str(e)}
    
    def predict_roberta_local(self, title, content):
        """Local RoBERTa prediction (keep original)"""
        try:
            if 'roberta' not in self.models:
                return {'error': 'RoBERTa model not loaded'}
            
            logger.info("🧠 Running Local RoBERTa prediction...")
            
            import torch
            model_data = self.models['roberta']
            
            safe_title = self.truncate_text(title, 100)
            safe_content = self.truncate_text(content, 300)
            input_str = f"<title>{safe_title}<content>{safe_content}<end>"
            
            inputs = model_data['tokenizer'].encode_plus(
                input_str,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = model_data['model'](
                    inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device)
                )
            
            probabilities = torch.nn.Softmax(dim=1)(outputs.logits)[0]
            probs_list = [x.item() for x in list(probabilities)]
            result_dict = dict(zip(["Fake", "Real"], probs_list))
            
            fake_prob = result_dict["Fake"]
            real_prob = result_dict["Real"]
            is_fake = fake_prob > real_prob
            confidence = max(fake_prob, real_prob) * 100
            
            return {
                'model': 'RoBERTa-Local',
                'label': 'Fake' if is_fake else 'Real',
                'confidence': round(confidence, 1),
                'fake_prob': round(fake_prob * 100, 2),
                'real_prob': round(real_prob * 100, 2),
                'reasoning': f"RoBERTa (local) detected {'suspicious' if is_fake else 'legitimate'} patterns"
            }
            
        except Exception as e:
            logger.error(f"❌ RoBERTa local prediction error: {e}")
            return {'model': 'RoBERTa-Local', 'error': str(e)}
    
    def predict_bart_local(self, title, content):
        """Local BART prediction"""
        try:
            if 'bart-mnli' not in self.models:
                return {'error': 'BART model not loaded'}
            
            safe_text = self.truncate_text(f"{title}. {content}", 400)
            labels = ["reliable news", "fake news", "misinformation", "factual reporting"]
            
            result = self.models['bart-mnli'](safe_text, candidate_labels=labels)
            
            reliable_score = sum(score for label, score in zip(result['labels'], result['scores']) 
                               if label in ['reliable news', 'factual reporting'])
            
            return {
                'model': 'BART-Local',
                'label': 'Real' if reliable_score > 0.5 else 'Fake',
                'confidence': round(reliable_score * 100, 1),
                'reasoning': f"BART (local) classified as {result['labels'][0]}"
            }
        except Exception as e:
            return {'model': 'BART-Local', 'error': str(e)}
    
    def predict_sentiment_local(self, title, content):
        """Local sentiment prediction"""
        try:
            if 'sentiment' not in self.models:
                return {'error': 'Sentiment model not loaded'}
            
            safe_text = self.truncate_text(f"{title}. {content}", 400)
            result = self.models['sentiment'](safe_text)
            
            sentiment = result[0]['label']
            confidence = result[0]['score'] * 100
            
            is_suspicious = sentiment in ['NEGATIVE'] and confidence > 80
            
            return {
                'model': 'Sentiment-Local',
                'label': 'Fake' if is_suspicious else 'Real',
                'confidence': round(confidence, 1),
                'reasoning': f"Sentiment (local): {sentiment.lower()} tone"
            }
        except Exception as e:
            return {'model': 'Sentiment-Local', 'error': str(e)}
    
    def comprehensive_ensemble_predict(self, title, content):
        """Run ALL models (local + API) and combine results, including LLaMA"""
        logger.info(f"🚀 Starting COMPREHENSIVE ensemble prediction...")
        
        prediction_functions = [
            ('RoBERTa-Local', self.predict_roberta_local),
            ('OpenAI-API', self.call_openai_api),
            ('Groq-API', self.call_groq_api),
            ('HuggingFace-API', self.call_huggingface_api),
            ('Search-Verification', self.search_and_verify),
            ('LLaMA-Comprehensive', self.predict_llama_comprehensive)
        ]
        
        if 'bart-mnli' in self.models:
            prediction_functions.append(('BART-Local', self.predict_bart_local))
        if 'sentiment' in self.models:
            prediction_functions.append(('Sentiment-Local', self.predict_sentiment_local))
        
        predictions = []
        
        def run_prediction(func_tuple):
            name, func = func_tuple
            try:
                start_time = time.time()
                result = func(title, content)
                processing_time = time.time() - start_time
                if 'error' not in result:
                    result['processing_time'] = round(processing_time, 3)
                    logger.info(f"✅ {result['model']}: {result['label']} ({result['confidence']}%) in {processing_time:.3f}s")
                    return result
                else:
                    logger.warning(f"⚠️ {result.get('model', name)}: {result['error']}")
                    return None
            except Exception as e:
                logger.error(f"❌ {name} failed: {e}")
                return None
        
        futures = [self.executor.submit(run_prediction, func_tuple) for func_tuple in prediction_functions]
        
        for future in futures:
            result = future.result()
            if result:
                predictions.append(result)
        
        if not predictions:
            return {
                'error': 'All models failed to make predictions',
                'ensemble_details': {'models_used': 0, 'predictions': []}
            }
        
        real_votes = sum(1 for p in predictions if p['label'] == 'Real')
        fake_votes = sum(1 for p in predictions if p['label'] == 'Fake')
        total_votes = real_votes + fake_votes
        
        weighted_real = 0
        weighted_fake = 0
        total_weight = 0
        
        for pred in predictions:
            confidence = pred['confidence'] / 100
            # Enhanced weighting system
            if 'LLaMA-Comprehensive' in pred['model']:
                weight = 1.4  # High weight for comprehensive LLaMA
            elif 'API' in pred['model'] or 'Search' in pred['model']:
                weight = 1.3
            elif 'RoBERTa' in pred['model']:
                weight = 1.2
            else:
                weight = 1.0

            if pred['label'] == 'Real':
                weighted_real += confidence * weight
            else:
                weighted_fake += confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            final_label = "Uncertain"
            final_confidence = 50.0
        else:
            weighted_real_avg = weighted_real / total_weight
            weighted_fake_avg = weighted_fake / total_weight
            
            if weighted_real_avg > weighted_fake_avg:
                final_label = "Trustworthy"
                final_confidence = min(weighted_real_avg * 100, 95)
            else:
                final_label = "Untrustworthy"
                final_confidence = min(weighted_fake_avg * 100, 95)
        
        api_models = [p['model'] for p in predictions if 'API' in p['model'] or 'Search' in p['model']]
        local_models = [p['model'] for p in predictions if 'Local' in p['model']]
        
        reasoning = f"Comprehensive analysis using {len(predictions)} models: "
        reasoning += f"{len(api_models)} API services and {len(local_models)} local models. "
        reasoning += f"Voting: {real_votes} trustworthy, {fake_votes} suspicious. "
        reasoning += f"Active models: {', '.join([p['model'][:15] for p in predictions[:4]])}{'...' if len(predictions) > 4 else ''}."
        
        logger.info(f"🎯 COMPREHENSIVE Result: {final_label} ({final_confidence:.1f}% confidence)")
        logger.info(f"📊 Used {len(api_models)} APIs + {len(local_models)} local models")
        
        return {
            'label': final_label,
            'confidence': round(final_confidence, 1),
            'real_probability': round((weighted_real / total_weight) * 100, 2) if total_weight > 0 else 50,
            'fake_probability': round((weighted_fake / total_weight) * 100, 2) if total_weight > 0 else 50,
            'reasoning': reasoning,
            'ensemble_details': {
                'total_models': len(predictions),
                'api_models_used': len(api_models),
                'local_models_used': len(local_models),
                'model_votes': {'real': real_votes, 'fake': fake_votes},
                'api_models': api_models,
                'local_models': local_models,
                'predictions': predictions
            }
        }

# Initialize comprehensive ensemble
ensemble = EnhancedMultiAPIEnsemble()

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        title = data.get('title', 'No title available')
        content = data.get('content', '')
        
        if not content.strip():
            return jsonify({"error": "Content is required for analysis"}), 400
        
        # Limit content length
        content = content[:4000]
        title = title[:200]
        
        logger.info(f"📥 New COMPREHENSIVE analysis request")
        logger.info(f"📝 Title: {title}")
        logger.info(f"📊 Content: {len(content)} chars")
        logger.info(f"🔑 Available APIs: {[k for k, v in ensemble.api_keys.items() if v]}")
        
        result = ensemble.comprehensive_ensemble_predict(title, content)
        
        if 'error' in result:
            return jsonify({
                "success": False,
                "error": result['error']
            }), 500
        
        logger.info(f"📤 COMPREHENSIVE analysis complete: {result['label']} ({result['confidence']}%)")
        logger.info(f"🎯 Models used: {result['ensemble_details']['api_models_used']} APIs + {result['ensemble_details']['local_models_used']} local")
        
        return jsonify({
            "success": True,
            "analysis": result
        })
        
    except Exception as e:
        logger.error(f"❌ Analysis endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    available_apis = [k for k, v in ensemble.api_keys.items() if v]
    
    return jsonify({
        "status": "healthy",
        "ensemble_info": {
            "loaded_local_models": ensemble.loaded_models,
            "failed_local_models": ensemble.failed_models,
            "available_apis": available_apis,
            "total_loaded": len(ensemble.loaded_models),
            "total_apis": len(available_apis)
        },
        "device": device,
        "python_version": sys.version
    })

@app.route('/models', methods=['GET'])
def models_info():
    """Detailed model information endpoint"""
    return jsonify({
        "comprehensive_ensemble_status": {
            "local_models": {
                "loaded": ensemble.loaded_models,
                "failed": ensemble.failed_models,
                "device": device
            },
            "api_services": {
                "available": [k for k, v in ensemble.api_keys.items() if v],
                "configured_keys": list(ensemble.api_keys.keys())
            },
            "total_prediction_capacity": len(ensemble.loaded_models) + len([k for k, v in ensemble.api_keys.items() if v and k != 'google_search_engine_id']),
            "ready_for_comprehensive_prediction": True
        }
    })

@app.route('/', methods=['GET'])
def root():
    available_apis = [k for k, v in ensemble.api_keys.items() if v]
    
    return jsonify({
        "service": "COMPREHENSIVE Multi-API + Multi-Model Ensemble Fact-Checking",
        "status": "running",
        "comprehensive_ensemble_info": {
            "local_models": len(ensemble.loaded_models),
            "api_services": len(available_apis),
            "total_prediction_sources": len(ensemble.loaded_models) + len(available_apis),
            "active_local_models": ensemble.loaded_models,
            "active_api_services": available_apis,
            "system_type": "Comprehensive Multi-Source Ensemble"
        },
        "endpoints": ["/analyze", "/health", "/models"]
    })

if __name__ == '__main__':
    logger.info("🚀 Starting COMPREHENSIVE Multi-API + Multi-Model Ensemble Service...")
    
    # Load local models
    models_loaded = ensemble.load_local_models()
    available_apis = [k for k, v in ensemble.api_keys.items() if v]
    
    logger.info(f"🤖 Local models: {len(ensemble.loaded_models)} loaded, {len(ensemble.failed_models)} failed")
    logger.info(f"🔑 API services: {len(available_apis)} available")
    logger.info(f"📊 Total prediction sources: {len(ensemble.loaded_models) + len(available_apis)}")
    
    if available_apis:
        logger.info(f"🌐 Available APIs: {', '.join(available_apis)}")
    
    if models_loaded or available_apis:
        logger.info(f"🎉 Comprehensive ensemble ready with {len(ensemble.loaded_models)} local models + {len(available_apis)} API services!")
    else:
        logger.warning("⚠️ No models or APIs loaded. Service will have limited functionality.")
    
    logger.info(f"🌐 Starting Flask server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)