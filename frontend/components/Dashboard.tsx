import React from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { AlertTriangle, CheckCircle, TrendingUp, ShieldCheck } from 'lucide-react';
import { BulkAnalysisSummary, AnalysisResult } from '../types';

interface Props {
  summary: BulkAnalysisSummary | null;
  results: AnalysisResult[];
}

const COLORS = ['#ef4444', '#10b981', '#3b82f6']; // Red (Fake), Green (Genuine), Blue

const Dashboard: React.FC<Props> = ({ summary, results }) => {
  if (!summary) return null;

  const pieData = [
    { name: 'Fake', value: summary.fakeCount },
    { name: 'Genuine', value: summary.genuineCount },
  ];

  const sentimentData = [
    { name: 'Positive', value: results.filter(r => r.sentiment === 'Positive').length },
    { name: 'Neutral', value: results.filter(r => r.sentiment === 'Neutral').length },
    { name: 'Negative', value: results.filter(r => r.sentiment === 'Negative').length },
  ];

  const ratingData = [
    { name: 'Original', rating: summary.originalAvgRating },
    { name: 'True Rating', rating: summary.trueAvgRating },
  ];

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Top Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-850 p-4 rounded-xl border border-slate-700 shadow-md">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-blue-500/10 rounded-lg">
                <ShieldCheck className="w-5 h-5 text-blue-500" />
            </div>
            <p className="text-slate-400 text-sm">Trust Score</p>
          </div>
          <p className="text-3xl font-bold text-white">{summary.overallTrustScore}%</p>
        </div>

        <div className="bg-slate-850 p-4 rounded-xl border border-slate-700 shadow-md">
          <div className="flex items-center gap-3 mb-2">
             <div className="p-2 bg-red-500/10 rounded-lg">
                <AlertTriangle className="w-5 h-5 text-red-500" />
             </div>
            <p className="text-slate-400 text-sm">Fake Reviews</p>
          </div>
          <p className="text-3xl font-bold text-red-400">{summary.fakeCount}</p>
        </div>

        <div className="bg-slate-850 p-4 rounded-xl border border-slate-700 shadow-md">
          <div className="flex items-center gap-3 mb-2">
             <div className="p-2 bg-emerald-500/10 rounded-lg">
                <CheckCircle className="w-5 h-5 text-emerald-500" />
            </div>
            <p className="text-slate-400 text-sm">Genuine Reviews</p>
          </div>
          <p className="text-3xl font-bold text-emerald-400">{summary.genuineCount}</p>
        </div>

         <div className="bg-slate-850 p-4 rounded-xl border border-slate-700 shadow-md">
          <div className="flex items-center gap-3 mb-2">
             <div className="p-2 bg-amber-500/10 rounded-lg">
                <TrendingUp className="w-5 h-5 text-amber-500" />
             </div>
            <p className="text-slate-400 text-sm">True Rating</p>
          </div>
          <div className="flex items-end gap-2">
             <p className="text-3xl font-bold text-white">{summary.trueAvgRating.toFixed(1)}</p>
             <p className="text-sm text-slate-500 mb-1 line-through decoration-red-500">{summary.originalAvgRating.toFixed(1)}</p>
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Fake vs Genuine Chart */}
        <div className="bg-slate-850 p-6 rounded-xl border border-slate-700 shadow-lg">
          <h3 className="text-lg font-semibold text-white mb-6">Authenticity Breakdown</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                />
                <Legend verticalAlign="bottom" height={36}/>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Rating Correction */}
        <div className="bg-slate-850 p-6 rounded-xl border border-slate-700 shadow-lg flex flex-col">
          <h3 className="text-lg font-semibold text-white mb-2">Rating Impact</h3>
          <p className="text-sm text-slate-400 mb-6 flex-grow">{summary.ratingExplanation}</p>
          <div className="h-64">
             <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ratingData} layout="vertical">
                    <XAxis type="number" domain={[0, 5]} stroke="#94a3b8" />
                    <YAxis dataKey="name" type="category" stroke="#94a3b8" width={100} />
                    <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
                        cursor={{fill: '#334155', opacity: 0.2}}
                    />
                    <Bar dataKey="rating" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                        {ratingData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={index === 0 ? '#94a3b8' : '#eab308'} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
