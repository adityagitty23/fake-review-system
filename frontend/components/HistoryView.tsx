import React from 'react';
import { AnalysisHistoryItem } from '../types';
import { Calendar, AlertTriangle, ShieldCheck } from 'lucide-react';

interface Props {
  history: AnalysisHistoryItem[];
}

const HistoryView: React.FC<Props> = ({ history }) => {
  // ✅ Safety check (prevents crash if history is undefined or empty)
  if (!history || history.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-slate-400">
        <Calendar className="w-16 h-16 mb-4 opacity-20" />
        <h3 className="text-xl font-semibold mb-2">No History Yet</h3>
        <p>Run your first audit to see results here.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white mb-6">Audit History</h2>

      <div className="grid grid-cols-1 gap-4">
        {history.map((item) => (
          <div
            key={item.id}
            className="bg-slate-850 border border-slate-700 rounded-xl p-6 hover:border-primary-500/50 transition-colors"
          >
            <div className="flex flex-col md:flex-row justify-between md:items-center gap-4">
              {/* Left section */}
              <div>
                <div className="flex items-center gap-3 mb-1">
                  <h3 className="text-lg font-semibold text-white">
                    {item.projectName}
                  </h3>

                  <span
                    className={`px-2 py-0.5 rounded text-xs font-medium ${
                      item.engine === 'GEMINI'
                        ? 'bg-indigo-500/20 text-indigo-300'
                        : 'bg-green-500/20 text-green-300'
                    }`}
                  >
                    {item.engine}
                  </span>
                </div>

                <p className="text-sm text-slate-400 flex items-center gap-2">
                  <Calendar className="w-3 h-3" />
                  {new Date(item.date).toLocaleString()}
                </p>
              </div>

              {/* Right stats */}
              <div className="flex items-center gap-6">
                <div className="text-center">
                  <p className="text-xs text-slate-500 uppercase">Reviews</p>
                  <p className="font-bold text-white">{item.reviewCount}</p>
                </div>

                <div className="text-center">
                  <p className="text-xs text-slate-500 uppercase">Fakes</p>
                  <p className="font-bold text-red-400">{item.fakeCount}</p>
                </div>

                <div className="text-center">
                  <p className="text-xs text-slate-500 uppercase">Trust Score</p>
                  <div className="flex items-center gap-1">
                    {item.trustScore > 70 ? (
                      <ShieldCheck className="w-4 h-4 text-emerald-500" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-amber-500" />
                    )}
                    <span
                      className={`font-bold ${
                        item.trustScore > 70
                          ? 'text-emerald-400'
                          : 'text-amber-400'
                      }`}
                    >
                      {item.trustScore}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Verdict section */}
            <div className="mt-4 pt-4 border-t border-slate-700/50">
              <p className="text-sm text-slate-300">
                <span className="text-slate-500">Verdict:</span>{' '}
                {item.summary.genuineCount} genuine reviews found. True rating
                adjusted from{' '}
                {Number(item.summary.originalAvgRating).toFixed(1)} to{' '}
                <span className="text-amber-400 font-semibold">
                  {Number(item.summary.trueAvgRating).toFixed(1)}
                </span>.
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HistoryView;
