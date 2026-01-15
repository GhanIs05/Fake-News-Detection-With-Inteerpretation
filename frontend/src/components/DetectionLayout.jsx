import { motion } from "framer-motion";
import { FileText, ImageIcon } from "lucide-react";

export default function DetectionLayout({ activeTab, setActiveTab, children }) {

  return (
    <section className="min-h-screen py-10 bg-linear-to-br from-gray-50 to-white">
      <div className="max-w-5xl mx-auto px-6">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >

          <h1 className="text-3xl md:text-4xl font-extrabold text-gray-900">
            Detect Fake News
          </h1>

          <p className="mt-4 text-gray-600">
            Paste news text or upload an image to verify authenticity with explainable AI.
          </p>

        </motion.div>

        {/* Tabs */}
        <div className="flex justify-center mb-10">
          <div className="inline-flex bg-gray-100 rounded-2xl p-1 shadow-sm">
            <button
              onClick={() => setActiveTab("text")}
              className={`flex items-center gap-2 px-6 py-3 rounded-2xl font-semibold transition cursor-pointer ${
                activeTab === "text"
                  ? "bg-white shadow text-indigo-600"
                  : "text-gray-600 hover:text-indigo-600"
              }`}
            >
              <FileText size={18} /> Text News
            </button>


            <button
              onClick={() => setActiveTab("image")}
              className={`flex items-center gap-2 px-6 py-3 rounded-2xl font-semibold transition cursor-pointer ${
                activeTab === "image"
                  ? "bg-white shadow text-indigo-600"
                  : "text-gray-600 hover:text-indigo-600"
              }`}
            >
              <ImageIcon size={18} /> Image News
            </button>
          </div>
        </div>

        {/* Content Area */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-white rounded-3xl shadow-xl p-6 md:p-10"
        >
          {children}
        </motion.div>

      </div>
    </section>
  );
}
