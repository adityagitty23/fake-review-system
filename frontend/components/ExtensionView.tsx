import React from 'react';
import { Chrome, Copy, Check, Download, Info } from 'lucide-react';

const ExtensionView: React.FC = () => {
  const [copiedManifest, setCopiedManifest] = React.useState(false);
  const [copiedContent, setCopiedContent] = React.useState(false);

  const manifestCode = `{
  "manifest_version": 3,
  "name": "Fake Review Detector",
  "version": "1.0",
  "description": "Analyze reviews on any webpage",
  "permissions": ["activeTab", "scripting"],
  "action": {
    "default_popup": "popup.html"
  },
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content.js"]
    }
  ]
}`;

  const contentScriptCode = `// content.js
// This script runs on the webpage
console.log("Fake Review Detector Active");

function extractReviews() {
  // Example selector for Amazon/Generic reviews
  const reviews = [];
  const reviewElements = document.querySelectorAll('.review-text-content, .review, .comment');
  
  reviewElements.forEach(el => {
    reviews.push(el.innerText);
  });
  
  return reviews;
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "scan") {
    const data = extractReviews();
    sendResponse({ reviews: data });
  }
});`;

  const copyToClipboard = (text: string, setFn: (b: boolean) => void) => {
    navigator.clipboard.writeText(text);
    setFn(true);
    setTimeout(() => setFn(false), 2000);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="bg-blue-600 p-2 rounded-lg">
          <Chrome className="w-6 h-6 text-white" />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-white">Browser Extension</h2>
          <p className="text-slate-400">Run the Fake Review System on any website.</p>
        </div>
      </div>

      <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-xl flex items-start gap-3">
        <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
        <p className="text-sm text-slate-300">
          Since this is a web application, you need to manually add the extension to Chrome for development mode. 
          Follow the steps below to enable real-time analysis on Amazon, Yelp, or other sites.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Step 1: Manifest */}
        <div className="bg-slate-850 border border-slate-700 rounded-xl p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-semibold text-white">1. Create manifest.json</h3>
            <button 
              onClick={() => copyToClipboard(manifestCode, setCopiedManifest)}
              className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded flex items-center gap-2 transition-colors"
            >
              {copiedManifest ? <Check className="w-3 h-3 text-green-400"/> : <Copy className="w-3 h-3"/>}
              Copy
            </button>
          </div>
          <pre className="bg-slate-950 p-4 rounded-lg text-xs font-mono text-slate-400 overflow-x-auto">
            {manifestCode}
          </pre>
        </div>

        {/* Step 2: Content Script */}
        <div className="bg-slate-850 border border-slate-700 rounded-xl p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-semibold text-white">2. Create content.js</h3>
            <button 
              onClick={() => copyToClipboard(contentScriptCode, setCopiedContent)}
              className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded flex items-center gap-2 transition-colors"
            >
               {copiedContent ? <Check className="w-3 h-3 text-green-400"/> : <Copy className="w-3 h-3"/>}
               Copy
            </button>
          </div>
          <pre className="bg-slate-950 p-4 rounded-lg text-xs font-mono text-slate-400 overflow-x-auto">
            {contentScriptCode}
          </pre>
        </div>
      </div>

      {/* Instructions */}
      <div className="bg-slate-850 border border-slate-700 rounded-xl p-6">
        <h3 className="font-semibold text-white mb-4">Installation Instructions</h3>
        <ol className="list-decimal pl-5 space-y-3 text-slate-300 text-sm">
          <li>Create a folder named <code>fake-review-extension</code> on your computer.</li>
          <li>Create two files inside it: <code>manifest.json</code> and <code>content.js</code> using the code above.</li>
          <li>Open Google Chrome and go to <code>chrome://extensions</code>.</li>
          <li>Enable <strong>Developer mode</strong> in the top right corner.</li>
          <li>Click <strong>Load unpacked</strong> and select your folder.</li>
          <li>The extension is now active! It will log "Fake Review Detector Active" in the console of visited pages.</li>
        </ol>
      </div>
    </div>
  );
};

export default ExtensionView;
