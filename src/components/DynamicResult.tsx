import React, { useState, useEffect } from 'react';
import { CheckCircle, XCircle, AlertTriangle, ExternalLink, ThumbsUp, ThumbsDown, Brain, Target, FileText } from 'lucide-react';
import LoadingState from './LoadingState';
import FeedbackSection from './FeedbackSection';
import CustomFeedbackSection from './CustomFeedbackSection';

interface AnalysisResult {
  url?: string;
  title: string;
  label: 'Trustworthy' | 'Untrustworthy';
  confidence: number;
  summary: string;
  reasoning: string;
  probabilities?: {
    fake: number;
    real: number;
  };
  model: string;
  analyzedAt: string;
  source: string;
}

interface BackendResponse {
  success: boolean;
  data: AnalysisResult;
  error?: string;
}

interface DynamicResultProps {
  searchUrl: string;
  onBack: () => void;
}

const DynamicResult: React.FC<DynamicResultProps> = ({ searchUrl, onBack }) => {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showFeedback, setShowFeedback] = useState(false);

  useEffect(() => {
    analyzeUrl();
  }, [searchUrl]);

  const analyzeUrl = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: searchUrl }),
      });

      const responseData: BackendResponse = await response.json();

      if (!response.ok || !responseData.success) {
        throw new Error(responseData.error || 'Failed to analyze URL');
      }

      setResult(responseData.data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (feedback: 'agree' | 'disagree', userLabel?: string, evidence?: string) => {
    try {
      await fetch('http://localhost:5000/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: 'analysis_feedback',
          content: {
            url: searchUrl,
            feedback,
            originalLabel: result?.label,
            userLabel,
            evidence,
            confidence: result?.confidence,
            model: result?.model
          }
        }),
      });
    } catch (err) {
      console.error('Feedback error:', err);
    }
  };

  if (loading) {
    return <LoadingState />;
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 py-12">
        <div className="container mx-auto px-4">
          <button
            onClick={onBack}
            className="mb-6 px-6 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-all duration-300 backdrop-blur-sm"
          >
            ← Back to Search
          </button>
          
          <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 border border-white/20">
            <div className="text-center">
              <AlertTriangle className="mx-auto h-16 w-16 text-red-400 mb-4" />
              <h2 className="text-2xl font-bold text-white mb-4">Analysis Failed</h2>
              <p className="text-gray-300 mb-6">{error}</p>
              <button
                onClick={analyzeUrl}
                className="px-6 py-3 bg-gradient-to-r from-purple-600 to-violet-600 text-white rounded-xl font-medium hover:from-purple-700 hover:to-violet-700 transition-all duration-300"
              >
                Try Again
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const isTrustworthy = result.label === 'Trustworthy';
  const confidenceColor = result.confidence >= 80 ? 'text-green-400' : 
                         result.confidence >= 60 ? 'text-yellow-400' : 'text-red-400';


  // Enhanced summary display
  const hasSummary = result.summary && result.summary !== 'Summary not available for this content.' && result.summary.length > 10;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 py-12">
      <div className="container mx-auto px-4 max-w-4xl">
        <button
          onClick={onBack}
          className="mb-6 px-6 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-all duration-300 backdrop-blur-sm"
        >
          ← Back to Search
        </button>

        {/* Main Result Card */}
        <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 border border-white/20 mb-8">
          <div className="flex items-start justify-between mb-6">
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-white mb-2">{result.title}</h1>
              {result.url && (
                <a 
                  href={result.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-300 hover:text-blue-200 flex items-center gap-2 mb-4"
                >
                  {result.url}
                  <ExternalLink className="h-4 w-4" />
                </a>
              )}
            </div>
          </div>

          {/* Trustworthiness Label */}
          <div className="flex items-center gap-4 mb-6 flex-wrap">
            <div className={`flex items-center gap-2 px-4 py-2 rounded-xl ${
              isTrustworthy 
                ? 'bg-green-500/20 border border-green-500/30' 
                : 'bg-red-500/20 border border-red-500/30'
            }`}>
              {isTrustworthy ? (
                <CheckCircle className="h-6 w-6 text-green-400" />
              ) : (
                <XCircle className="h-6 w-6 text-red-400" />
              )}
              <span className={`font-bold text-lg ${
                isTrustworthy ? 'text-green-300' : 'text-red-300'
              }`}>
                {result.label}
              </span>
            </div>
            
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5 text-gray-400" />
              <span className="text-gray-300">Confidence:</span>
              <span className={`font-bold text-lg ${confidenceColor}`}> 
                {result.confidence}%
              </span>
            </div>

            {result.probabilities && (
              <div className="flex items-center gap-2 text-sm">
                <Brain className="h-4 w-4 text-purple-400" />
                <span className="text-gray-400">
                  Real: {result.probabilities.real}% | Fake: {result.probabilities.fake}%
                </span>
              </div>
            )}
          </div>

          {/* AI Model Info */}
          <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-3 mb-6">
            <div className="flex items-center gap-2 text-purple-300">
              <Brain className="h-4 w-4" />
              <span className="text-sm font-medium">
                Analyzed by: {result.model || 'AI Model'}
              </span>
            </div>
          </div>

          {/* Summary - Enhanced with better condition */}
          {hasSummary && (
            <div className="mb-6">
              <h3 className="text-xl font-semibold text-white mb-3 flex items-center gap-2">
                <FileText className="h-5 w-5 text-blue-400" />
                Content Summary
              </h3>
              <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                <p className="text-gray-300 leading-relaxed">
                  {result.summary}
                </p>
              </div>
            </div>
          )}

          {/* Reasoning */}
          <div className="mb-6">
            <h3 className="text-xl font-semibold text-white mb-3 flex items-center gap-2">
              <Brain className="h-5 w-5 text-green-400" />
              AI Analysis
            </h3>
            <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
              <p className="text-gray-300 leading-relaxed">
                {result.reasoning}
              </p>
            </div>
          </div>

          {/* Feedback Buttons */}
          <div className="flex items-center gap-4 pt-6 border-t border-white/10">
            <span className="text-gray-300">Was this analysis helpful?</span>
            <button
              onClick={() => {
                handleFeedback('agree');
                setShowFeedback(false);
              }}
              className="flex items-center gap-2 px-4 py-2 bg-green-500/20 hover:bg-green-500/30 text-green-300 rounded-lg transition-all duration-300"
            >
              <ThumbsUp className="h-4 w-4" />
              Yes
            </button>
            <button
              onClick={() => setShowFeedback(true)}
              className="flex items-center gap-2 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded-lg transition-all duration-300"
            >
              <ThumbsDown className="h-4 w-4" />
              No
            </button>
          </div>
        </div>

        {/* Feedback Section */}
        {showFeedback && (
          <CustomFeedbackSection
            onSubmit={(userLabel, evidence) => {
              handleFeedback('disagree', userLabel, evidence);
              setShowFeedback(false);
            }}
            onCancel={() => setShowFeedback(false)}
          />
        )}

        {/* Analysis Details */}
        <div className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10">
          <h3 className="text-lg font-semibold text-white mb-3">Analysis Details</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Analyzed at:</span>
              <span className="text-gray-300">{new Date(result.analyzedAt).toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Source:</span>
              <span className="text-gray-300 capitalize">{result.source}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Model:</span>
              <span className="text-gray-300">{result.model}</span>
            </div>
            {result.probabilities && (
              <div className="flex justify-between">
                <span className="text-gray-400">Raw Probabilities:</span>
                <span className="text-gray-300">
                  Real: {result.probabilities.real}%, Fake: {result.probabilities.fake}%
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DynamicResult;
