# Truthly — Real-Time Fake News Detection System

A browser-integrated misinformation detection system that labels 
news credibility at the point of discovery — directly in Google 
Search results.

## Problem
71% of users rely on online news. Most fact-checking tools work 
after content is consumed. Truthly moves verification to the 
point of discovery.

## Architecture
React Frontend → Node/Express Backend → Flask ML Service → 
External APIs (OpenAI, Groq, Google Search)

## Tech Stack
- Frontend: React 18, TypeScript, Webpack 5, Tailwind CSS
- Backend: Node.js, Express, node-cron
- ML Service: Python, Flask, PyTorch, Hugging Face Transformers,
  sentence-transformers, spaCy, NLTK, BeautifulSoup
- APIs: OpenAI GPT, Groq, Google Search (Serper)

## Features
- Chrome extension labels search results as credible/not in real time
- Evidence-backed reasoning explains why content is flagged
- Confidence score displayed per article
- User feedback loop for continuous improvement
- Context-aware: distinguishes satire, bias, and misinformation

## How It Works
1. Extension intercepts Google Search results
2. URLs sent to Node backend
3. Flask service runs NLP pipeline:
   - Content extraction (newspaper3k, BeautifulSoup)
   - Sentence embeddings (sentence-transformers)
   - Claim verification (OpenAI / Groq)
4. Credibility score + reasoning returned to extension

## Achievement
Built at Nexothon 2025 — Top 5 out of 350+ teams
Only second-year team among 29 finalists
