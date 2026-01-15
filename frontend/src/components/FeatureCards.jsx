import { motion } from "framer-motion";
import { FileText, ImageIcon, BrainCircuit } from "lucide-react";

const features = [
  {
    icon: <FileText size={28} />,
    title: "Text News Detection",
    description:
      "Analyze textual news content using advanced NLP models like BERT to identify misinformation with high accuracy.",
  },
  {
    icon: <ImageIcon size={28} />,
    title: "Image News Detection",
    description:
      "Detect manipulated or misleading images using CNN-based models and visual feature analysis.",
  },
  {
    icon: <BrainCircuit size={28} />,
    title: "Interpretable AI",
    description:
      "Understand predictions clearly with LIME and SHAP explanations highlighting influential words or regions.",
  },
];

export default function FeatureCards() {
  return (
    <section className="py-10 bg-white">
      <div className="max-w-7xl mx-auto px-6">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-extrabold text-gray-900">
            Powerful Features
          </h2>
          <p className="mt-4 text-lg font-semibold text-gray-600 max-w-2xl mx-auto">
            Our system combines state-of-the-art machine learning with transparency to help users trust and verify news content.
          </p>
        </motion.div>

        {/* Cards */}
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="group bg-gray-50 rounded-3xl p-8 shadow-sm hover:shadow-xl transition"
            >
              <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-indigo-100 text-indigo-600 mb-6 group-hover:scale-110 transition">
                {feature.icon}
              </div>

              <h3 className="text-xl font-semibold text-black mb-3">
                {feature.title}
              </h3>

              <p className="text-gray-600 text-lg font-semibold leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
