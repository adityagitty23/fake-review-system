import React, { useState, useEffect } from 'react';
import { Activity, FileText, BarChart3, Shield, RotateCw, AlertCircle, Check, History, Chrome, LogOut, User as UserIcon } from 'lucide-react';
import ReviewInput from './components/ReviewInput';
import Dashboard from './components/Dashboard';
import ReportView from './components/ReportView';
import Auth from './components/Auth';
import HistoryView from './components/HistoryView';
import ExtensionView from './components/ExtensionView';
import { ReviewData, AnalysisResult, AnalysisStatus, BulkAnalysisSummary, ReportData, User, AnalysisHistoryItem } from './types';
import { analyzeReviewsBatch, generateAuditReport } from './services/reviewService';
import { getCurrentUser, signOut, saveHistory, getHistory } from './services/storageService';
import { v4 as uuidv4 } from 'uuid';

type AnalysisMode = 'local' | 'ai';


const App: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);
  const [reviews, setReviews] = useState<ReviewData[]>([]);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [status, setStatus] = useState<AnalysisStatus>(AnalysisStatus.Idle);
  const [activeTab, setActiveTab] = useState<'input' | 'dashboard' | 'report' | 'history' | 'extension'>('input');
  const [summary, setSummary] = useState<BulkAnalysisSummary | null>(null);
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>('local');
  const [report, setReport] = useState<ReportData | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);

  useEffect(() => {
    const currentUser = getCurrentUser();
    if (currentUser) {
      setUser(currentUser);
      loadHistory(currentUser.email);
    }
  }, []);

  const loadHistory = (email: string) => {
    setHistory(getHistory(email));
  };

  const handleLogin = (newUser: User) => {
    setUser(newUser);
    loadHistory(newUser.email);
    setActiveTab('input');
  };

  const handleLogout = () => {
    signOut();
    setUser(null);
    setReviews([]);
    setResults([]);
    setSummary(null);
    setReport(null);
  };

  const handleAddReviews = (newReviews: ReviewData[]) => {
    setReviews(prev => [...prev, ...newReviews]);
  };

  const handleClearReviews = () => {
      setReviews([]);
      setResults([]);
      setSummary(null);
      setReport(null);
      setStatus(AnalysisStatus.Idle);
      setErrorMsg(null);
  }

    const runAnalysis = async () => {
      if (reviews.length === 0) return;

      setStatus(AnalysisStatus.Analyzing);
      setErrorMsg(null);

      try {
        // ✅ API returns OBJECT now
        const response = await analyzeReviewsBatch(reviews, analysisMode);

        const batchResults = response.results;
        const trueRating = response.trueRating;
        const ratingExplanation = response.ratingExplanation;

        setResults(batchResults);

        const fakeCount = batchResults.filter(r => r.label === "Fake").length;
        const genuineCount = batchResults.filter(r => r.label === "Genuine").length;

        const originalAvg =
          reviews.reduce((sum, r) => sum + r.rating, 0) / reviews.length;

        const trustScore = Math.round(
          (genuineCount / reviews.length) * 100
        );

        const newSummary: BulkAnalysisSummary = {
          totalReviews: reviews.length,
          fakeCount,
          genuineCount,
          overallTrustScore: trustScore,
          originalAvgRating: originalAvg,
          trueAvgRating: trueRating,
          ratingExplanation
        };

        setSummary(newSummary);

        // Save history
        if (user) {
          const historyItem: AnalysisHistoryItem = {
            id: uuidv4(),
            date: new Date().toISOString(),
            projectName: reviews[0]?.source || "Manual Entry Batch",

            // ✅ REVIEW SHIELD / LOCAL FIX
            engine: analysisMode === 'ai' ? 'REVIEW SHIELD' : 'LOCAL',

            reviewCount: reviews.length,
            fakeCount,
            trustScore,
            summary: newSummary
          };

          saveHistory(user.email, historyItem);
          loadHistory(user.email);
        }

        // Generate report
        const sampleFakes = batchResults
          .filter(r => r.label === "Fake")
          .slice(0, 3)
          .map(r => {
            const originalText = reviews.find(rv => rv.id === r.reviewId)?.text || "";
            return originalText;
          });

        const reportData = await generateAuditReport(newSummary, sampleFakes);
        setReport(reportData);

        setStatus(AnalysisStatus.Complete);
        setActiveTab("dashboard");

      } catch (error: any) {
        console.error(error);
        setStatus(AnalysisStatus.Error);
        setErrorMsg(error.message || "An unexpected error occurred");
      }
    };


  if (!user) {
    return <Auth onLogin={handleLogin} />;
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col md:flex-row font-sans">
      
      {/* Sidebar Navigation */}
      <aside className="w-full md:w-64 bg-slate-900 border-r border-slate-800 flex flex-col fixed md:relative z-10 h-auto min-h-screen">
        <div className="p-6 border-b border-slate-800 flex items-center gap-3">
          <Shield className="w-8 h-8 text-primary-500" />
          <h1 className="text-xl font-bold tracking-tight text-white leading-tight">Fake Review<br/>System</h1>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          <div className="px-3 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
            Analyze
          </div>
          <button
            onClick={() => setActiveTab('input')}
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors text-sm font-medium ${
              activeTab === 'input' ? 'bg-primary-600 text-white' : 'text-slate-400 hover:bg-slate-800'
            }`}
          >
            <FileText className="w-4 h-4" />
            <span>Input Reviews</span>
          </button>
          
          <button
            onClick={() => setActiveTab('dashboard')}
            disabled={status !== AnalysisStatus.Complete}
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors text-sm font-medium ${
              activeTab === 'dashboard' ? 'bg-primary-600 text-white' : 'text-slate-400 hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            <span>Dashboard</span>
          </button>

          <button
            onClick={() => setActiveTab('report')}
            disabled={status !== AnalysisStatus.Complete}
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors text-sm font-medium ${
              activeTab === 'report' ? 'bg-primary-600 text-white' : 'text-slate-400 hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed'
            }`}
          >
            <Activity className="w-4 h-4" />
            <span>Audit Report</span>
          </button>

          <div className="mt-6 px-3 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
            Records
          </div>
          <button
             onClick={() => setActiveTab('history')}
             className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors text-sm font-medium ${
              activeTab === 'history' ? 'bg-primary-600 text-white' : 'text-slate-400 hover:bg-slate-800'
            }`}
          >
            <History className="w-4 h-4" />
            <span>Past Audits</span>
          </button>

           <div className="mt-6 px-3 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
            Tools
          </div>
          <button
             onClick={() => setActiveTab('extension')}
             className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-colors text-sm font-medium ${
              activeTab === 'extension' ? 'bg-primary-600 text-white' : 'text-slate-400 hover:bg-slate-800'
            }`}
          >
            <Chrome className="w-4 h-4" />
            <span>Chrome Extension</span>
          </button>
        </nav>

        <div className="p-4 border-t border-slate-800">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 rounded-full bg-primary-600/30 flex items-center justify-center text-primary-400 font-bold">
                {user.name.charAt(0)}
              </div>
              <div className="overflow-hidden">
                <p className="text-sm font-medium text-white truncate">{user.name}</p>
                <p className="text-xs text-slate-500 truncate">{user.email}</p>
              </div>
            </div>
            <button 
              onClick={handleLogout}
              className="w-full bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs py-2 rounded flex items-center justify-center gap-2 transition-colors"
            >
              <LogOut className="w-3 h-3" /> Sign Out
            </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 p-4 md:p-8 overflow-y-auto h-screen relative">
        <div className="max-w-6xl mx-auto">
            
            {/* Error Banner */}
            {status === AnalysisStatus.Error && errorMsg && (
                <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-4 rounded-xl mb-6 flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                    <div>
                        <h3 className="font-semibold">Analysis Failed</h3>
                        <p className="text-sm opacity-90">{errorMsg}</p>
                    </div>
                </div>
            )}

            {/* Header Area */}
            <div className="flex justify-between items-center mb-8">
                <div>
                    <h2 className="text-2xl font-bold text-white">
                        {activeTab === 'input' && 'Review Data Ingestion'}
                        {activeTab === 'dashboard' && 'Analytics Dashboard'}
                        {activeTab === 'report' && 'Final Audit Report'}
                        {activeTab === 'history' && 'Past Analysis History'}
                        {activeTab === 'extension' && 'Extension Setup'}
                    </h2>
                        <p className="text-slate-400 text-sm mt-1 flex items-center gap-3">
                          {status === AnalysisStatus.Analyzing ? (
                            'Processing data...'
                          ) : (
                            <>
                              Powered by
                              <span
                                className={`font-semibold ${
                                  analysisMode === 'local'
                                    ? 'text-green-400'
                                    : 'text-blue-400'
                                }`}
                              >
                                {analysisMode === 'local'
                                  ? 'Local ML Models (TF-IDF + BERT)'
                                  : 'REVIEW SHIELD'}
                              </span>

                              {/* Toggle Button */}
                              <button
                                onClick={() =>
                                  setAnalysisMode(prev => (prev === 'local' ? 'ai' : 'local'))
                                }
                                disabled={status === AnalysisStatus.Analyzing}
                                className="ml-2 px-3 py-1 rounded-full text-xs font-medium border border-slate-600 
                                          hover:bg-slate-800 transition disabled:opacity-50"
                              >
                                Switch to {analysisMode === 'local' ? 'AI' : 'Local'}
                              </button>
                            </>
                          )}
                        </p>

                </div>
                
                {activeTab === 'input' && reviews.length > 0 && (
                     <button
                        onClick={runAnalysis}
                        disabled={status === AnalysisStatus.Analyzing}
className="px-6 py-2.5 rounded-lg font-medium flex items-center gap-2 shadow-lg disabled:opacity-50 disabled:cursor-wait text-white transition-colors bg-green-600 hover:bg-green-500 shadow-green-900/20"

                    >
                        {status === AnalysisStatus.Analyzing ? (
                            <RotateCw className="w-4 h-4 animate-spin" />
                        ) : (
                            <Shield className="w-4 h-4" />
                        )}
                          {status === AnalysisStatus.Analyzing ? 'Analyzing...' : 'Run Fake Review Audit'}
                    </button>
                )}
            </div>

            {/* Content Switcher */}
            {activeTab === 'input' && (
                <div className="space-y-6">
                    <ReviewInput onAddReviews={handleAddReviews} />
                    
                    {reviews.length > 0 && (
                        <div className="bg-slate-850 border border-slate-800 rounded-xl overflow-hidden">
                            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center">
                                <h3 className="font-semibold text-slate-300">Staged Reviews ({reviews.length})</h3>
                                <button onClick={handleClearReviews} className="text-xs text-red-400 hover:text-red-300">Clear All</button>
                            </div>
                            <div className="divide-y divide-slate-800 max-h-[400px] overflow-y-auto">
                                {reviews.map((r, i) => (
                                    <div key={r.id} className="p-4 hover:bg-slate-800/50 transition-colors flex justify-between gap-4">
                                        <div>
                                            <div className="flex gap-2 items-center mb-1">
                                                <span className="text-xs font-mono text-slate-500 bg-slate-900 px-1 rounded">#{i + 1}</span>
                                                <div className="flex text-amber-400 text-xs">{'★'.repeat(r.rating)}</div>
                                            </div>
                                            <p className="text-sm text-slate-300 line-clamp-2">{r.text}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'dashboard' && summary && (
                <div className="space-y-8">
                    <Dashboard summary={summary} results={results} />
                    
                    {/* Detailed List */}
                    <div className="bg-slate-850 rounded-xl border border-slate-700 overflow-hidden">
                         <div className="px-6 py-4 border-b border-slate-700">
                             <h3 className="text-lg font-semibold text-white">Detailed Analysis</h3>
                         </div>
                         <div className="overflow-x-auto">
                             <table className="w-full text-left text-sm text-slate-400">
                                 <thead className="bg-slate-900 text-slate-200 uppercase text-xs font-semibold">
                                     <tr>
                                         <th className="px-6 py-3">Label</th>
                                         <th className="px-6 py-3">Confidence</th>
                                         <th className="px-6 py-3">Sentiment</th>
                                         <th className="px-6 py-3">Reasoning</th>
                                     </tr>
                                 </thead>
                                 <tbody className="divide-y divide-slate-700">
                                     {results.map((r) => (
                                         <tr key={r.reviewId} className="hover:bg-slate-800/50 transition-colors">
                                             <td className="px-6 py-4">
                                                 <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                     r.label === 'Genuine' 
                                                     ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                                                     : 'bg-red-500/10 text-red-400 border border-red-500/20'
                                                 }`}>
                                                     {r.label === 'Genuine' ? <Check className="w-3 h-3"/> : <AlertCircle className="w-3 h-3"/>}
                                                     {r.label}
                                                 </span>
                                             </td>
                                             <td className="px-6 py-4">
                                                 <div className="flex items-center gap-2">
                                                     <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                                         <div 
                                                            className={`h-full rounded-full ${r.confidenceScore > 80 ? 'bg-primary-500' : 'bg-amber-500'}`} 
                                                            style={{width: `${r.confidenceScore}%`}}
                                                         ></div>
                                                     </div>
                                                     <span className="text-xs">{r.confidenceScore}%</span>
                                                 </div>
                                             </td>
                                             <td className="px-6 py-4 text-white">{r.sentiment}</td>
                                             <td className="px-6 py-4 max-w-xs truncate" title={r.reason}>{r.reason}</td>
                                         </tr>
                                     ))}
                                 </tbody>
                             </table>
                         </div>
                    </div>
                </div>
            )}

            {activeTab === 'report' && report && summary && (
                <div className="pb-10">
                    <ReportView report={report} summary={summary} />
                </div>
            )}

            {activeTab === 'history' && (
              <HistoryView history={history} />
            )}

            {activeTab === 'extension' && (
              <ExtensionView />
            )}
        </div>
      </main>
    </div>
  );
};

export default App;
