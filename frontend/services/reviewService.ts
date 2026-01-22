import {
  ReviewData,
  AnalysisResult,
  BulkAnalysisSummary,
  ReportData,
  AnalyzeResponse
} from "../types";

const API_URL = "http://localhost:5000";

/**
 * Analyze reviews (LOCAL or AI)
 */
export async function analyzeReviewsBatch(
  reviews: ReviewData[],
  mode: "local" | "ai"
): Promise<AnalyzeResponse> {

  const endpoint =
    mode === "local"
      ? `${API_URL}/predict`
      : `${API_URL}/predict-ai`;

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ reviews })
  });

  if (!response.ok) {
    throw new Error(
      mode === "local"
        ? "Local analysis server is not running"
        : "AI analysis service failed"
    );
  }

  // ✅ RETURNS OBJECT
  return await response.json();
}

/**
 * Generate audit report
 */
export async function generateAuditReport(
  summary: BulkAnalysisSummary,
  fakeSamples: string[]
): Promise<ReportData> {

  return {
    summary: `Out of ${summary.totalReviews} reviews analyzed, ${
      summary.fakeCount
    } were detected as fake.
The platform trust score is ${summary.overallTrustScore}%. 
After confidence-weighting, the corrected average rating is ${summary.trueAvgRating}.`,

    keyPatterns: fakeSamples.map(
      (_, i) => `Pattern ${i + 1}: Suspicious or repetitive language detected`
    ),

    recommendations: [
      "Remove detected fake reviews from listings.",
      "Monitor users repeatedly posting suspicious reviews.",
      "Encourage verified purchases.",
      "Run periodic review audits."
    ]
  };
}
