import React from 'react';
import { Section } from './Section';
import { Leaf, ShieldCheck, Microscope } from 'lucide-react';

export function ProblemStatement() {
    return (
        <Section id="problem" className="relative">
            <div className="max-w-4xl mx-auto px-6">
                <div className="grid md:grid-cols-2 gap-12 items-center">
                    <div className="text-left">
                        <h2 className="text-3xl md:text-4xl font-bold mb-6">The Agricultural Challenge</h2>
                        <div className="space-y-4 text-slate-400 leading-relaxed">
                            <p>
                                Plant diseases are a major threat to global food security, causing significant yield losses annually. Traditional diagnostic methods are often slow and require expert knowledge.
                            </p>
                            <p>
                                Early and accurate detection of diseases in the field is critical for effective management and minimizing the use of pesticides, leading to more sustainable farming.
                            </p>
                        </div>
                    </div>

                    <div className="grid gap-6">
                        <div className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50 backdrop-blur-sm hover:bg-emerald-500/10 hover:border-emerald-500/50 transition-all group">
                            <div className="w-12 h-12 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                                <Leaf size={24} />
                            </div>
                            <h3 className="text-xl font-semibold mb-2">Crop Yield Loss</h3>
                            <p className="text-slate-400 text-sm">Diseases can destroy entire crops if not identified in time, leading to economic instability for farmers.</p>
                        </div>

                        <div className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50 backdrop-blur-sm hover:bg-blue-500/10 hover:border-blue-500/50 transition-all group translate-x-4 md:translate-x-8">
                            <div className="w-12 h-12 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                                <Microscope size={24} />
                            </div>
                            <h3 className="text-xl font-semibold mb-2">Expert Scarcity</h3>
                            <p className="text-slate-400 text-sm">Access to plant pathologists is limited in many regions, making automated tools essential for localized diagnosis.</p>
                        </div>
                    </div>
                </div>
            </div>
        </Section>
    );
}
