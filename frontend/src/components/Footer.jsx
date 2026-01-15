import { ShieldCheck, Github, Linkedin, Mail } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-gray-900 text-gray-300 mt-20">
      <div className="max-w-7xl mx-auto px-6 py-14 grid grid-cols-1 md:grid-cols-3 gap-12">
        
        {/* Brand */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-white font-bold text-xl">
            <ShieldCheck className="text-indigo-400" />
            FakeDetect
          </div>
          <p className="text-sm text-gray-400 leading-relaxed">
            Detect fake news using advanced machine learning models with
            interpretable AI to build trust and transparency in digital media.
          </p>
        </div>

        {/* Quick Links */}
        <div>
          <h3 className="text-white font-semibold mb-4">Quick Links</h3>
          <ul className="space-y-2 text-sm">
            <li className="hover:text-white cursor-pointer">Home</li>
            <li className="hover:text-white cursor-pointer">Detect News</li>
            <li className="hover:text-white cursor-pointer">How It Works</li>
            <li className="hover:text-white cursor-pointer">About Project</li>
          </ul>
        </div>

        {/* Contact / Social */}
        <div>
          <h3 className="text-white font-semibold mb-4">Connect</h3>
          <div className="flex gap-4">
            <a
              href="#"
              className="p-3 rounded-xl bg-gray-800 hover:bg-indigo-600 transition"
            >
              <Github size={18} />
            </a>
            <a
              href="#"
              className="p-3 rounded-xl bg-gray-800 hover:bg-indigo-600 transition"
            >
              <Linkedin size={18} />
            </a>
            <a
              href="#"
              className="p-3 rounded-xl bg-gray-800 hover:bg-indigo-600 transition"
            >
              <Mail size={18} />
            </a>
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="border-t border-gray-800 py-6 text-center text-sm text-gray-500">
        © {new Date().getFullYear()} FakeDetect. Built with React, Tailwind & Interpretable AI.
      </div>
    </footer>
  );
}
