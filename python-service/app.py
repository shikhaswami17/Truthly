from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    global model, tokenizer, device
    
    try:
        logger.info("Loading required libraries...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        logger.info("Loading RoBERTa fake news model...")
        tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
        model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")
        
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def predict_fake_news(title, content):
    """
    Use the EXACT code from HuggingFace documentation for this model
    """
    if not model or not tokenizer:
        raise Exception("Model not loaded")
    
    try:
        import torch
        
        # Format input EXACTLY as required by the model
        input_str = "<title>" + title + "<content>" + content + "<end>"
        
        # Tokenize and encode
        input_ids = tokenizer.encode_plus(
            input_str, 
            max_length=512, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Make prediction using the EXACT HuggingFace code
        with torch.no_grad():
            output = model(
                input_ids["input_ids"].to(device), 
                attention_mask=input_ids["attention_mask"].to(device)
            )
        
        # Apply softmax EXACTLY as in HuggingFace documentation
        softmax_output = torch.nn.Softmax(dim=1)(output.logits)[0]
        
        # Convert to list and get probabilities
        probs_list = [x.item() for x in list(softmax_output)]
        
        # Create dictionary as in original code
        result_dict = dict(zip(["Fake", "Real"], probs_list))
        
        fake_prob = result_dict["Fake"]
        real_prob = result_dict["Real"]
        
        # Determine label and confidence
        is_fake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob) * 100
        
        return {
            "label": "Untrustworthy" if is_fake else "Trustworthy",
            "confidence": round(confidence, 1),
            "fake_probability": round(fake_prob * 100, 2),
            "real_probability": round(real_prob * 100, 2),
            "reasoning": generate_reasoning(is_fake, confidence),
            "raw_predictions": result_dict
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(f"Output logits shape: {output.logits.shape if 'output' in locals() else 'N/A'}")
        raise e

def generate_reasoning(is_fake, confidence):
    if is_fake:
        if confidence > 90:
            return "High confidence: Content shows strong linguistic patterns associated with misinformation."
        elif confidence > 75:
            return "Moderate confidence: Content contains several indicators typical of unreliable sources."
        else:
            return "Low confidence: Some language patterns suggest potential reliability issues."
    else:
        if confidence > 90:
            return "High confidence: Content demonstrates strong indicators of factual, reliable reporting."
        elif confidence > 75:
            return "Moderate confidence: Language patterns align with trustworthy journalism standards."
        else:
            return "Low confidence: Content appears to follow basic factual reporting patterns."

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        title = data.get('title', 'No title available')
        content = data.get('content', '')
        
        if not content.strip():
            return jsonify({"error": "Content is required for analysis"}), 400
        
        # Limit content length to avoid memory issues
        content = content[:4000]
        title = title[:200]
        
        logger.info(f"Analyzing content: {len(content)} chars, title: {len(title)} chars")
        
        result = predict_fake_news(title, content)
        
        logger.info(f"Analysis complete: {result['label']} ({result['confidence']}%)")
        
        return jsonify({
            "success": True,
            "analysis": result
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint"""
    try:
        # Test with simple content
        result = predict_fake_news("Test News", "This is a test article about technology.")
        return jsonify({
            "success": True,
            "test_result": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "python_version": sys.version
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "service": "RoBERTa Fake News Detection",
        "status": "running",
        "endpoints": ["/analyze", "/health", "/test"]
    })

if __name__ == '__main__':
    logger.info("Starting RoBERTa Fake News Detection Service...")
    
    model_loaded = load_model()
    if not model_loaded:
        logger.warning("Model failed to load. Service will start but may not work properly.")
    
    logger.info(f"Starting Flask server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
