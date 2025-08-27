import React, { useState } from 'react';
import Navbar from './Navbar';
import Hero from './Hero';
import DynamicResult from './DynamicResult';
import ExampleResult from './ExampleResult';
import Features from './Features';
import FeedbackSection from './FeedbackSection';
import Footer from './Footer';
import { useNavigate } from 'react-router-dom';

const TruthlyApp = () => {
  const [newsInput, setNewsInput] = useState('');
  const [showResult, setShowResult] = useState(false);
  const [searchUrl, setSearchUrl] = useState<string | null>(null);
  const [feedbackText, setFeedbackText] = useState('');
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const navigate = useNavigate();

    const handleCheckCredibility = () => {
    if (!newsInput.trim()) {
      alert('Please enter a news link to analyze.');
      return;
    }
    setSearchUrl(newsInput.trim());
    setShowResult(true);
    navigate('/result', { state: { searchUrl: newsInput.trim() } });
  };


  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleCheckCredibility();
    }
  };

  const handleFeedbackSubmit = () => {
    if (!feedbackText.trim()) {
      alert('Please provide your feedback before submitting.');
      return;
    }
    setIsSubmittingFeedback(true);
    setTimeout(() => {
      alert('Thank you for your feedback! It helps us improve Truthly.');
      setFeedbackText('');
      setIsSubmittingFeedback(false);
    }, 1000);
  };

  const scrollToSection = (sectionId: string) => {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="bg-white text-gray-800">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        * {
          font-family: 'Inter', sans-serif;
        }
        @keyframes fadeIn {
          to { opacity: 1; }
        }
        .pulse-dot {
          animation: pulse 2s infinite;
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
      <Navbar onNav={scrollToSection} />
      <Hero 
        newsInput={newsInput}
        setNewsInput={setNewsInput}
        onKeyPress={handleKeyPress}

      />
    <ExampleResult />
      <Features />
      <FeedbackSection 
        feedbackText={feedbackText}
        setFeedbackText={setFeedbackText}
        isSubmittingFeedback={isSubmittingFeedback}
        onSubmit={handleFeedbackSubmit}
      />
      <Footer />
    </div>
  );
};

export default TruthlyApp;