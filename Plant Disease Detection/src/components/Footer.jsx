import React from 'react';
import { Github, Linkedin, Mail } from 'lucide-react';

export function Footer() {
    return (
        <footer className="bg-slate-950 py-12 border-t border-slate-800/50">
            <div className="max-w-7xl mx-auto px-8 flex flex-col md:flex-row justify-between items-center gap-6">
                <div>
                    <h3 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 text-transparent bg-clip-text">
                        Gas Detection ML
                    </h3>
                    <p className="text-slate-500 text-sm mt-2">
                        Innovating safety with machine learning.
                    </p>
                </div>

                <div className="flex gap-6">
                    <a href="https://github.com/kushagragupta04/Multi-Class-Multimodal-Gas-Detection-using-ML" className="text-slate-400 hover:text-white transition-colors">
                        <Github size={20} />
                    </a>
                    <a href="https://www.linkedin.com/in/kushagragupta08/" className="text-slate-400 hover:text-white transition-colors">
                        <Linkedin size={20} />
                    </a>
                    <a href="mailto:kushagra2808gupta@gmail.com" className="text-slate-400 hover:text-white transition-colors">
                        <Mail size={20} />
                    </a>
                </div>

                {/* <div className="text-slate-600 text-xs">
                    © 2026 Project Team. All rights reserved.
                </div> */}
            </div>
        </footer>
    );
}
