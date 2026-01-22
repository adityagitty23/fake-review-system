// ---------------- ANALYSIS STATUS ----------------

export enum AnalysisStatus {
  Idle = 'IDLE',
  Analyzing = 'ANALYZING',
  Complete = 'COMPLETE',
  Error = 'ERROR',
}

// ---------------- USER ----------------

export interface User {
  email: string;
  name: string;
}

// ---------------- HISTORY ITEM ----------------

export interface AnalysisHistoryItem {
  id: string;
  date: string;
  projectName: string;

  // ✅ REQUIRED (FIX)
  engine: 'GEMINI' | 'LOCAL';

  reviewCount: number;
  fakeCount: number;
  trustScore: number;
  summary: BulkAnalysisSummary;
}

// ---------------- REVIEW INPUT ----------------

export interface ReviewData {
  id: string;
  text: string;
  rating: number; // 1–5 stars
  author?: string;
  date?: string;
  source?: string;
}

// ---------------- MODEL RESULT ----------------

export interface AnalysisResult {
  reviewId: string;
  label: 'Genuine' | 'Fake';
  confidenceScore: number; // 0–100
  reason: string;
  sentiment: 'Positive' | 'Neutral' | 'Negative';
  sentimentExplanation?: string;
}

// ---------------- SUMMARY ----------------

export interface BulkAnalysisSummary {
  totalReviews: number;
  fakeCount: number;
  genuineCount: number;
  overallTrustScore: number;
  originalAvgRating: number;
  trueAvgRating: number;
  ratingExplanation: string;
}

// ---------------- REPORT ----------------

export interface ReportData {
  summary: string;
  keyPatterns: string[];
  recommendations: string[];
}

// ---------------- BACKEND RESPONSE ----------------

export interface AnalyzeResponse {
  results: AnalysisResult[];
  trueRating: number;
  ratingExplanation: string;
}
