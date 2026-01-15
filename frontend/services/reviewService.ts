import {
  ReviewData,
  AnalysisResult,
  BulkAnalysisSummary,
  ReportData
} from "../types";

const API_URL = "http://localhost:5000";

/**
 * Analyze reviews using LOCAL backend only
 */
export async function analyzeReviewsBatch(
  reviews: ReviewData[]
): Promise<AnalysisResult[]> {

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ reviews })
  });

  if (!response.ok) {
    throw new Error("Local analysis server is not running");
  }

  return await response.json();
}

/**
 * Calculate true rating after removing fake reviews
 */
export async function calculateTrueRating(
  reviews: ReviewData[],
  results: AnalysisResult[]
): Promise<{ trueRating: number; explanation: string }> {

  const genuine = results.filter(r => r.label === "Genuine");

  const total = genuine.reduce((sum, r) => {
    const rev = reviews.find(rv => rv.id === r.reviewId);
    return sum + (rev?.rating || 0);
  }, 0);

  const trueRating = total / (genuine.length || 1);

  return {
    trueRating: Number(trueRating.toFixed(1)),
    explanation:
      "The true rating is calculated after excluding reviews detected as fake by local ML models."
  };
}

/**
 * Generate audit report (local logic)
 */
export async function generateAuditReport(
  summary: BulkAnalysisSummary,
  fakeSamples: string[]
): Promise<ReportData> {

  return {
    summary: `Out of ${summary.totalReviews} reviews analyzed, ${summary.fakeCount} were detected as fake.
The platform trust score is ${summary.overallTrustScore}%. After removal, the corrected average rating is ${summary.trueAvgRating}.`,

    keyPatterns: fakeSamples.map(
      (_, i) => `Pattern ${i + 1}: Suspicious or repetitive language detected`
    ),

    recommendations: [
      "Remove detected fake reviews from listings.",
      "Monitor users repeatedly posting fake reviews.",
      "Encourage verified purchases.",
      "Run periodic review audits."
    ]
  };
}
