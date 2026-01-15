import { motion } from "framer-motion";
import { ShieldAlert, Sparkles } from "lucide-react";

export default function HeroSection() {
  return (
    <section className="relative py-10 bg-linear-to-br from-indigo-50 via-white to-purple-50 overflow-hidden">
      {/* Decorative Blur */}
      <div className="absolute -top-24 -left-24 w-96 h-96 bg-indigo-300/30 rounded-full blur-3xl" />
      <div className="absolute top-1/3 -right-24 w-96 h-96 bg-purple-300/30 rounded-full blur-3xl" />

      <div className="relative max-w-7xl mx-auto px-6 grid md:grid-cols-2 gap-14 items-center">
        {/* Left Content */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-100 text-indigo-700 text-sm font-medium mb-6">
            <Sparkles size={16} />
            AI-Powered Fake News Detection
          </div>

          <h1 className="text-4xl md:text-5xl xl:text-6xl font-extrabold leading-tight text-gray-900">
            Detect Fake News with
            <span className="block text-indigo-600">Interpretable AI</span>
          </h1>

          <p className="mt-6 text-gray-600 text-lg max-w-xl">
            Identify misinformation in text and images using advanced machine learning models with clear, human-understandable explanations.
          </p>

          {/* <div className="mt-10 flex flex-col sm:flex-row gap-4">
            <button className="px-7 py-3 rounded-2xl bg-indigo-600 text-white font-semibold hover:bg-indigo-700 transition shadow-lg">
              Start Detection
            </button>
            <button className="px-7 py-3 rounded-2xl border border-gray-300 text-gray-700 font-semibold hover:bg-gray-100 transition">
              Learn More
            </button>
          </div> */}
        </motion.div>

        {/* Right Visual */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="relative"
        >
          <div className="bg-white rounded-3xl shadow-xl p-6 md:p-8">
            <div className="flex items-center gap-3 mb-6">
              <ShieldAlert className="text-indigo-600" size={28} />
              <h3 className="text-lg font-semibold text-gray-800">News Analysis Preview</h3>
            </div>

            <div className="space-y-4">
              <div className="h-4 bg-gray-200 rounded w-3/4" />
              <div className="h-4 bg-gray-200 rounded w-full" />
              <div className="h-4 bg-gray-200 rounded w-5/6" />

              <div className="mt-6 flex items-center justify-between">
                <span className="px-4 py-1 rounded-full bg-red-100 text-red-600 text-sm font-semibold">
                  Fake News
                </span>
                <span className="text-sm text-gray-500">Confidence: 92%</span>
              </div>

              <div className="mt-4 p-3 bg-indigo-50 rounded-xl text-sm text-indigo-700">
                Key indicators highlighted using Interpretable AI
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
