# Truthly

Most fact-checking happens after you've already read something.
By then it's too late — you've formed an opinion, maybe shared it.
Truthly moves that check to the moment you see the headline in 
Google Search.

## What it does
Labels search results as credible or not, right there on the 
results page. Click any result and you get a score + the actual 
reasoning behind it — not just a red/green badge.

## How it actually works
Extension sends the URL to an Express backend.
Backend passes it to a Flask service that:
- Extracts the article content
- Runs it through NLP (spaCy, sentence-transformers)
- Sends the claim to OpenAI / Groq for verification
- Returns a credibility score with evidence

React frontend → Node/Express → Flask ML service → APIs

## Tech
- Frontend: React 18, TypeScript, Tailwind CSS, Webpack 5
- Backend: Node.js, Express
- ML: Python, Flask, PyTorch, Hugging Face Transformers,
  sentence-transformers, spaCy, NLTK
- APIs: Serper, Hugging Face Inference API

## Honest limitations
The reasoning quality depends on the LLM.
No database — stateless by design for now.
Doesn't yet handle paywalled content well.
