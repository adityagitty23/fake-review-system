// Analysis status for UI state
export enum AnalysisStatus {
  Idle = 'IDLE',
  Analyzing = 'ANALYZING',
  Complete = 'COMPLETE',
  Error = 'ERROR',
}

// Logged-in user
export interface User {
  email: string;
  name: string;
}

// Saved audit history item
export interface AnalysisHistoryItem {
  id: string;
  date: string;
  projectName: string;
  reviewCount: number;
  fakeCount: number;
  trustScore: number;
  summary: BulkAnalysisSummary;
}

// Individual review input
export interface ReviewData {
  id: string;
  text: string;
  rating: number; // 1–5 stars
  author?: string;
  date?: string;
  source?: string;
}

// Model prediction result
export interface AnalysisResult {
  reviewId: string;
  label: 'Genuine' | 'Fake';
  confidenceScore: number; // 0–100
  reason: string;
  sentiment: 'Positive' | 'Neutral' | 'Negative';
  sentimentExplanation?: string;
}

// Aggregated analysis statistics
export interface BulkAnalysisSummary {
  totalReviews: number;
  fakeCount: number;
  genuineCount: number;
  overallTrustScore: number;
  originalAvgRating: number;
  trueAvgRating: number;
  ratingExplanation: string;
}

// Final audit report
export interface ReportData {
  summary: string;
  keyPatterns: string[];
  recommendations: string[];
}

// Backend response for analysis
export interface AnalyzeResponse {
  results: AnalysisResult[];
  trueRating: number;
  ratingExplanation: string;
}
