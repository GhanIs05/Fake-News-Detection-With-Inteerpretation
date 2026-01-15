import { motion } from "framer-motion";
import { CheckCircle, XCircle, Info, Image as ImageIcon } from "lucide-react";

export default function ResultCard({ result }) {
  if (!result) return null;

  const isReal = result.verdict === "real";
  const isImage = result.type === "image";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="mt-10 rounded-3xl shadow-xl border border-gray-200 overflow-hidden"
    >
      {/* ================= HEADER ================= */}
      <div
        className={`p-6 flex items-center gap-4 ${
          isReal ? "bg-green-50" : "bg-red-50"
        }`}
      >
        <div
          className={`p-3 rounded-2xl ${
            isReal
              ? "bg-green-100 text-green-600"
              : "bg-red-100 text-red-600"
          }`}
        >
          {isReal ? <CheckCircle size={28} /> : <XCircle size={28} />}
        </div>

        <div>
          <h3 className="text-xl font-bold text-gray-800">
            {isReal ? "Real News" : "Fake News"}
          </h3>
          <p className="text-sm text-gray-600">
            Confidence: {result.confidence}%
          </p>
        </div>
      </div>

      {/* ================= BODY ================= */}
      <div className="p-6 space-y-6 bg-white">

        {/* 🔤 TEXT EXPLANATION */}
        {!isImage && (
          <div>
            <div className="flex items-center gap-2 mb-3 font-semibold text-gray-800">
              <Info size={18} /> AI Explanation
            </div>

            <p className="text-gray-600 mb-3">
              The model focused on the following important words:
            </p>

            <div className="flex flex-wrap gap-2">
              {result.explanation?.map((item, index) => (
                <span
                  key={index}
                  className={`px-3 py-1 rounded-full text-sm ${
                    isReal
                      ? "bg-green-100 text-green-700"
                      : "bg-red-100 text-red-700"
                  }`}
                >
                  {item}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* 🖼 IMAGE HEATMAP EXPLAINABILITY */}
        {isImage && result.heatmap && (
          <div>
            <div className="flex items-center gap-2 mb-3 font-semibold text-gray-800">
              <ImageIcon size={18} /> Visual Explainability (Grad-CAM)
            </div>

            <img
              src={`data:image/png;base64,${result.heatmap}`}
              alt="AI Heatmap"
              className="w-full max-w-lg rounded-xl border shadow-md"
            />

            <p className="text-sm mt-2 font-medium flex items-center gap-2">
  <span
    className={`w-3 h-3 rounded-full ${
      isReal ? "bg-green-500" : "bg-red-500"
    }`}
  ></span>

  {isReal
    ? "The areas show where the AI found strong evidence that the image is real."
    : "The areas show where the AI found suspicious or manipulated visual patterns."}
</p>

          </div>
        )}

        {/* 🔗 Sources (Text Real News Only) */}
        {!isImage && isReal && result.sources && (
          <div>
            <h4 className="font-semibold text-gray-800 mb-2">
              Verified Sources
            </h4>
            <ul className="list-disc list-inside text-indigo-600 text-sm space-y-1">
              {result.sources.map((src, index) => (
                <li key={index}>{src}</li>
              ))}
            </ul>
          </div>
        )}

      </div>
    </motion.div>
  );
}
