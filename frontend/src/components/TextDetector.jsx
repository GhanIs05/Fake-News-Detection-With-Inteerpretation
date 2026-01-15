import { useState } from "react";
import { motion } from "framer-motion";
import { FileText, Loader2 } from "lucide-react";
import { analyzeText } from "../services/api";

export default function TextDetector({ onResult, model, setModel }) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setLoading(true);

    try {
      const result = await analyzeText(text, model);
      onResult({
        type: "text",
        ...result,
        model, // pass model to ResultCard if needed
      });
    } catch (err) {
      console.error(err);
      alert("Backend error. Is the server running?");
    }

    setLoading(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-3 rounded-2xl bg-indigo-100 text-indigo-600">
          <FileText size={22} />
        </div>
        <h2 className="text-xl font-bold text-gray-800">
          Text News Analysis
        </h2>
      </div>

      {/* ✅ Model Selector */}
      <div className="flex flex-col sm:flex-row gap-2 sm:items-center">
        <label className="text-sm font-semibold text-gray-600">
          Select Model
        </label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-full sm:w-60 rounded-xl border border-gray-300 p-2"
        >
          <option value="bert">BERT (Recommended)</option>
          <option value="xgboost">XGBoost</option>
          <option value="logistic">Logistic Regression</option>
        </select>
      </div>

      {/* Textarea */}
      <div className="relative">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste the complete news article or headline here..."
          rows={9}
          className="w-full rounded-2xl border border-gray-300 p-5 pr-16 focus:ring-2 focus:ring-indigo-500 outline-none resize-none text-gray-700"
        />
        <span className="absolute bottom-3 right-4 text-xs text-gray-400">
          {text.length} chars
        </span>
      </div>

      {/* Action Button */}
      <button
        onClick={handleAnalyze}
        disabled={loading || !text.trim()}
        className={`w-full sm:w-auto px-10 py-3 rounded-2xl font-semibold flex items-center justify-center gap-2 transition shadow 
          ${
            loading || !text.trim()
              ? "bg-gray-300 text-gray-500 cursor-not-allowed"
              : "bg-indigo-600 text-white hover:bg-indigo-700"
          }`}
      >
        {loading ? (
          <>
            <Loader2 className="animate-spin" size={18} /> Analyzing...
          </>
        ) : (
          "Analyze Text"
        )}
      </button>

      {/* UX Hint */}
      <p className="text-sm text-gray-500">
        Tip: Longer articles provide more accurate results with better explainability.
      </p>
    </motion.div>
  );
}
