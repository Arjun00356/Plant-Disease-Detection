import React from 'react';
import { Section } from './Section';
import { User, GraduationCap } from 'lucide-react';

export function Credits() {
    return (
        <Section id="credits" className="py-10">
            <div className="text-center mb-10">
                <h3 className="text-sm font-bold text-slate-500 uppercase tracking-[0.2em] mb-2">Academic Credits</h3>
                <div className="w-16 h-1 bg-gradient-to-r from-blue-500 to-emerald-500 mx-auto rounded-full" />
            </div>

            <div className="flex flex-col md:flex-row justify-center gap-8 max-w-4xl mx-auto">
                {/* Project Guide Card */}
                <div className="flex-1 p-8 rounded-3xl bg-slate-900/50 border border-slate-800 hover:border-blue-500/50 transition-all group relative overflow-hidden text-center">
                    <div className="absolute inset-0 bg-blue-500/5 group-hover:bg-blue-500/10 transition-colors" />

                    <div className="relative z-10">
                        <div className="w-16 h-16 mx-auto bg-blue-500/10 text-blue-400 rounded-full flex items-center justify-center mb-4 ring-1 ring-blue-500/20">
                            <GraduationCap size={32} />
                        </div>
                        <div className="text-xs font-bold text-blue-400 uppercase tracking-wider mb-2">Mentor</div>
                        <h3 className="text-2xl font-bold text-white mb-1">Ms. Vaishali Chauhan</h3>
                        {/* <p className="text-slate-500 text-sm">Designation / Department</p> */}
                    </div>
                </div>

                {/* Team Member Card */}
                <div className="flex-1 p-8 rounded-3xl bg-slate-900/50 border border-slate-800 hover:border-emerald-500/50 transition-all group relative overflow-hidden text-center">
                    <div className="absolute inset-0 bg-emerald-500/5 group-hover:bg-emerald-500/10 transition-colors" />

                    <div className="relative z-10">
                        <div className="w-16 h-16 mx-auto bg-emerald-500/10 text-emerald-400 rounded-full flex items-center justify-center mb-4 ring-1 ring-emerald-500/20">
                            <User size={32} />
                        </div>
                        <div className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-2">Student Name</div>
                        <h3 className="text-2xl font-bold text-white mb-2">Arjun Mishra</h3>
                        <p className="text-slate-400 font-mono text-sm bg-slate-800/50 inline-block px-3 py-1 rounded-full border border-slate-700/50">
                            Reg No: 23FE10CSE00356
                        </p>
                    </div>
                </div>
            </div>
        </Section>
    );
}
