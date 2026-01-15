import { motion } from "framer-motion";
import { Newspaper, Search } from "lucide-react";
import { useState } from "react";
import { Menu, X, ShieldCheck } from "lucide-react";
import { useNavigate } from "react-router-dom";

export default function Navbar() {

  const navigate = useNavigate();

    const [open, setOpen] = useState(false);

  return (
    <motion.nav
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="w-full sticky top-0 z-50 backdrop-blur-lg bg-white/80 shadow-sm"
    >
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2 cursor-pointer"
          onClick={() => navigate('/')}>
          <Newspaper className="w-7 h-7 text-indigo-600" />
          <h1 className="text-xl font-bold tracking-tight text-gray-900">
            Fake<span className="text-indigo-600">Detect</span>
          </h1>
        </div>

        

        {/* Navigation
        <div className="hidden md:flex items-center gap-8 text-sm font-medium text-black">
          <a href="#home" className="text-lg font-bold hover:text-indigo-600 transition">Home</a>
          <a href="#detect" className="text-lg font-bold hover:text-indigo-600 transition">Detect</a>
          <a href="#about" className="text-lg font-bold hover:text-indigo-600 transition">About</a>
        </div> */}

        {/* Action Button */}
        <button className="hidden md:flex items-center text-center gap-2  px-5 py-2 rounded-xl bg-indigo-600 text-white text-md cursor-pointer font-semibold shadow-md hover:bg-indigo-700 transition"
          onClick={() => navigate('/detect')}>
          <Search className="w-4 h-4" />
          Start Detection
        </button>

        {/* Mobile Menu Button */}
        <button
        onClick={() => setOpen(!open)}
        className="md:hidden text-gray-800"
        >
            {open ? <X size={26} /> : <Menu size={26} />}
        </button>

      </div>

      {/* Mobile Menu */}
        {open && (
        <div className="md:hidden bg-white border-t border-gray-200">
            <div className="flex flex-col px-4 py-4 gap-4">
                <a href="#" className="text-md font-semibold text-black hover:text-indigo-600">Home</a>
                <a href="#" className="text-md font-semibold text-black hover:text-indigo-600">Detect</a>
                <a href="#" className="text-md font-semibold text-black hover:text-indigo-600">About</a>
                {/* Action Button */}
                <button className="flex items-center justify-center gap-2  px-5 py-2 rounded-xl bg-indigo-600 text-white text-md font-semibold shadow-md hover:bg-indigo-700 transition"
                  onClick={() => navigate('/detect')}>
                    <Search className="w-4 h-4" />
                    Start Detection
                </button>
            </div>
        </div>
        )}

    </motion.nav>
  );
}
