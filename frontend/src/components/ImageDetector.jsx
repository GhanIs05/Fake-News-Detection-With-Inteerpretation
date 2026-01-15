import { useState } from "react";
import { motion } from "framer-motion";
import { ImageIcon, UploadCloud, Loader2 } from "lucide-react";
import { analyzeImage } from "../services/api";

export default function ImageDetector({ onResult }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (file) => {
    if (!file) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleAnalyze = async () => {
    if (!image) return;
    setLoading(true);

    try {
      const result = await analyzeImage(image);

      onResult({
        type: "image",
        verdict: result.label.toLowerCase(),
        confidence: result.confidence,
        heatmap: result.heatmap
      });

    } catch (err) {
      alert("Image analysis failed. Is backend running?");
      console.error(err);
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
          <ImageIcon size={22} />
        </div>
        <h2 className="text-xl font-bold text-gray-800">Image News Analysis</h2>
      </div>

      {/* Upload Box */}
      <label className="group cursor-pointer">
        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleImageChange(e.target.files[0])}
        />

        <div className="border-2 border-dashed border-gray-300 rounded-3xl p-10 flex flex-col items-center justify-center text-center transition group-hover:border-indigo-400 group-hover:bg-indigo-50">
          {!preview ? (
            <>
              <UploadCloud size={36} className="text-gray-400 mb-4" />
              <p className="font-medium text-gray-700">
                Upload or drag & drop an image
              </p>
              <p className="text-sm text-gray-400 mt-1">
                PNG, JPG, JPEG supported
              </p>
            </>
          ) : (
            <img
              src={preview}
              alt="Preview"
              className="max-h-64 rounded-xl object-contain"
            />
          )}
        </div>
      </label>

      {/* Change image hint */}
      {preview && (
        <p className="text-sm text-gray-500 text-center">
          Click the image to upload another one
        </p>
      )}

      {/* Spacing */}
      <div className="mt-6" />

      {/* Analyze button */}
      <button
        onClick={handleAnalyze}
        disabled={!image || loading}
        className={`w-full sm:w-auto px-10 py-3 rounded-2xl font-semibold flex items-center justify-center gap-2 transition shadow
          ${!image || loading
            ? "bg-gray-300 text-gray-500 cursor-not-allowed"
            : "bg-indigo-600 text-white hover:bg-indigo-700"}`}
      >
        {loading ? (
          <>
            <Loader2 className="animate-spin" size={18} /> Analyzing…
          </>
        ) : (
          "Analyze Image"
        )}
      </button>

      <p className="text-sm text-gray-500 text-center">
        Tip: Clear, un-cropped images improve detection accuracy.
      </p>
    </motion.div>
  );
}
