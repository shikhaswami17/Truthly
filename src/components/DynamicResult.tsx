import React, { useState, useEffect } from 'react';
import { 
  CheckCircle, XCircle, AlertTriangle, ExternalLink, ThumbsUp, ThumbsDown, 
  Brain, Target, FileText, Activity, Server, Clock, TrendingUp, Zap, Users
} from 'lucide-react';
import LoadingState from './LoadingState';
import CustomFeedbackSection from './CustomFeedbackSection';

interface APIFlags {
  model_loaded: boolean;
  text_processed: boolean;
  inference_successful: boolean;
}

interface TrackingInfo {
  request_id: number;
  timestamp: string;
  processing_node: string;
}

interface EnsembleDetails {
  total_models: number;
  active_predictions: number;
  model_votes: {
    real: number;
    fake: number;
  };
  predictions: Array<{
    model: string;
    label: string;
    confidence: number;
    reasoning: string;
    fake_prob?: number;
    real_prob?: number;
    is_toxic?: boolean;
    sentiment?: string;
  }>;
}

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
  processing_time?: number;
  model_status?: string;
  api_flags?: APIFlags;
  tracking_info?: TrackingInfo;
  ensemble_details?: EnsembleDetails;
}

interface BackendResponse {
  success: boolean;
  data: AnalysisResult;
  error?: string;
  api_flags?: APIFlags;
}

interface SystemHealth {
  status: string;
  uptime?: string;
  total_requests?: number;
  success_rate?: number;
  ensemble_info?: {
    loaded_models: string[];
    failed_models: string[];
    total_loaded: number;
    total_failed: number;
  };
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
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [lastAnalyzedUrl, setLastAnalyzedUrl] = useState<string | null>(null); // Add this

  useEffect(() => {
    // Prevent duplicate analysis of same URL
    if (searchUrl !== lastAnalyzedUrl) {
      analyzeUrl();
      fetchSystemHealth();
      setLastAnalyzedUrl(searchUrl);
    }
  }, [searchUrl, lastAnalyzedUrl]); // Add lastAnalyzedUrl dependency

  // Rest of your component stays the same...


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

  const fetchSystemHealth = async () => {
    try {
      const response = await fetch('http://localhost:5001/health');
      const healthData = await response.json();
      setSystemHealth(healthData);
    } catch (err) {
      console.error('Failed to fetch system health:', err);
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
          type: 'enhanced_analysis_feedback',
          content: {
            url: searchUrl,
            feedback,
            originalLabel: result?.label,
            userLabel,
            evidence,
            confidence: result?.confidence,
            model: result?.model,
            request_id: result?.tracking_info?.request_id
          }
        }),
      });
    } catch (err) {
      console.error('Feedback error:', err);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-400';
      case 'processing': return 'text-yellow-400';
      case 'loading': return 'text-blue-400';
      case 'failed': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4" />;
      case 'processing': return <Activity className="h-4 w-4 animate-pulse" />;
      case 'loading': return <Clock className="h-4 w-4 animate-spin" />;
      case 'failed': return <XCircle className="h-4 w-4" />;
      default: return <AlertTriangle className="h-4 w-4" />;
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

        {/* System Status Banner */}
        {systemHealth && (
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-4 border border-white/20 mb-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Server className="h-5 w-5 text-blue-400" />
                  <span className="text-white font-medium">System Status:</span>
                  <span className={`font-bold ${getStatusColor(systemHealth.status)}`}>
                    {systemHealth.status.toUpperCase()}
                  </span>
                </div>
                {systemHealth.uptime && (
                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    <Clock className="h-4 w-4" />
                    <span>Uptime: {systemHealth.uptime}</span>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-4">
                {systemHealth.success_rate !== undefined && (
                  <div className="text-sm text-gray-400">
                    <TrendingUp className="h-4 w-4 inline mr-1" />
                    Success Rate: <span className="text-green-400 font-bold">{systemHealth.success_rate}%</span>
                  </div>
                )}
                {systemHealth.total_requests !== undefined && (
                  <div className="text-sm text-gray-400">
                    Total Requests: <span className="text-blue-400 font-bold">{systemHealth.total_requests}</span>
                  </div>
                )}
                {systemHealth.ensemble_info && (
                  <div className="text-sm text-gray-400">
                    Models: <span className="text-purple-400 font-bold">{systemHealth.ensemble_info.total_loaded}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

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

          {/* Enhanced Status Display */}
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

            {result.processing_time && (
              <div className="flex items-center gap-2 text-sm">
                <Zap className="h-4 w-4 text-yellow-400" />
                <span className="text-gray-400">
                  Processed in {result.processing_time}s
                </span>
              </div>
            )}
          </div>

          {/* Multi-Model Ensemble Details - MOVED TO CORRECT LOCATION */}
          {result.ensemble_details && (
            <div className="bg-gray-800/50 rounded-lg p-4 mb-6">
              <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                <Users className="h-4 w-4" />
                Multi-Model Ensemble ({result.ensemble_details.active_predictions} Models Active)
              </h4>
              
              <div className="mb-4">
                <div className="flex items-center gap-4 mb-2">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span className="text-green-300 text-sm">Trustworthy: {result.ensemble_details.model_votes.real}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <span className="text-red-300 text-sm">Suspicious: {result.ensemble_details.model_votes.fake}</span>
                  </div>
                </div>
                
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full transition-all duration-1000" 
                    style={{
                      width: `${(result.ensemble_details.model_votes.real / (result.ensemble_details.model_votes.real + result.ensemble_details.model_votes.fake)) * 100}%`
                    }}
                  ></div>
                </div>
              </div>

              <div className="space-y-2">
                {result.ensemble_details.predictions.map((prediction, index) => (
                  <div key={index} className="bg-white/5 rounded p-3">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-medium text-white text-sm">{prediction.model}</span>
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 rounded text-xs font-bold ${
                          prediction.label === 'Real' 
                            ? 'bg-green-500/30 text-green-300' 
                            : 'bg-red-500/30 text-red-300'
                        }`}>
                          {prediction.label}
                        </span>
                        <span className="text-gray-400 text-xs">{prediction.confidence}%</span>
                      </div>
                    </div>
                    <p className="text-gray-400 text-xs">{prediction.reasoning}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* API Status Flags */}
          {result.api_flags && (
            <div className="bg-gray-800/50 rounded-lg p-4 mb-6">
              <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                <Activity className="h-4 w-4" />
                API Status Flags
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="flex items-center gap-2">
                  {getStatusIcon(result.api_flags.model_loaded ? 'healthy' : 'failed')}
                  <span className={`text-sm ${result.api_flags.model_loaded ? 'text-green-400' : 'text-red-400'}`}>
                    Model Loaded
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {getStatusIcon(result.api_flags.text_processed ? 'healthy' : 'failed')}
                  <span className={`text-sm ${result.api_flags.text_processed ? 'text-green-400' : 'text-red-400'}`}>
                    Text Processed
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {getStatusIcon(result.api_flags.inference_successful ? 'healthy' : 'failed')}
                  <span className={`text-sm ${result.api_flags.inference_successful ? 'text-green-400' : 'text-red-400'}`}>
                    Inference Complete
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* AI Model Info */}
          <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-3 mb-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-purple-300">
                <Brain className="h-4 w-4" />
                <span className="text-sm font-medium">
                  Analyzed by: {result.model || 'AI Model'}
                  {result.ensemble_details && ` (${result.ensemble_details.total_models} models)`}
                </span>
              </div>
              {result.tracking_info && (
                <div className="text-xs text-gray-400">
                  Request ID: #{result.tracking_info.request_id}
                </div>
              )}
            </div>
          </div>

          {/* Summary */}
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

        {/* Technical Details */}
        <div className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <Server className="h-5 w-5" />
              Technical Details
            </h3>
            <button
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
              className="text-sm px-3 py-1 bg-white/10 text-gray-300 rounded hover:bg-white/20 transition-all duration-300"
            >
              {showTechnicalDetails ? 'Hide' : 'Show'} Details
            </button>
          </div>
          
          {showTechnicalDetails && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div className="space-y-2">
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
                </div>
                <div className="space-y-2">
                  {result.processing_time && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Processing time:</span>
                      <span className="text-gray-300">{result.processing_time}s</span>
                    </div>
                  )}
                  {result.tracking_info && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Request ID:</span>
                        <span className="text-gray-300">#{result.tracking_info.request_id}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Processing node:</span>
                        <span className="text-gray-300">{result.tracking_info.processing_node}</span>
                      </div>
                    </>
                  )}
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
              
              {systemHealth?.ensemble_info && (
                <div className="border-t border-white/10 pt-4">
                  <h4 className="text-white font-medium mb-3">Ensemble System Status</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-gray-800/30 rounded p-3">
                      <div className="text-sm text-gray-400 mb-1">Loaded Models</div>
                      <div className="text-green-400 font-bold">{systemHealth.ensemble_info.total_loaded}</div>
                      <div className="text-xs text-gray-500 mt-1">
                        {(systemHealth.ensemble_info.loaded_models ?? []).join(', ')}
                      </div>
                    </div>
                    <div className="bg-gray-800/30 rounded p-3">
                      <div className="text-sm text-gray-400 mb-1">Failed Models</div>
                      <div className="text-red-400 font-bold">{systemHealth.ensemble_info.total_failed}</div>
                      {(systemHealth.ensemble_info.failed_models ?? []).length > 0 && (
                        <div className="text-xs text-gray-500 mt-1">
                          {(systemHealth.ensemble_info.failed_models ?? []).join(', ')}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DynamicResult;
