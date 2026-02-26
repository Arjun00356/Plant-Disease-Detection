import React from 'react';
import { Section } from './Section';
import { motion } from 'framer-motion';

export function Results() {
    const results = [
        { model: "ResNet18", accuracy: 95.96, formatted: "95.96%" },
        { model: "MobileNetV2", accuracy: 96.44, formatted: "96.44%" },
        { model: "Custom CNN", accuracy: 97.30, formatted: "97.30%", best: true },
    ];

    // Sorting for visualization if needed, but keeping list order is often better for comparison
    const maxVal = 100;

    return (
        <Section id="results">
            <div className="grid md:grid-cols-2 gap-12 items-center">
                <div>
                    <h2 className="text-3xl md:text-4xl font-bold mb-6 text-left">Performance Evaluation</h2>
                    <p className="text-slate-400 mb-6 leading-relaxed text-left">
                        Upon evaluating our models, <strong className="text-emerald-400">Custom CNN</strong> emerged as the most accurate classifier with <strong className="text-white">97.30%</strong> accuracy, showcasing the efficiency of a tailored architecture in identifying subtle disease patterns.
                        <br /><br />
                        While MobileNetV2 and ResNet18 provide strong competitive performance, the Custom CNN architecture provides the robust feature extraction and precision necessary for high-fidelity classification across diverse plant species.
                    </p>

                    <div className="grid grid-cols-2 gap-4">
                        <div className="bg-slate-800/30 p-4 rounded-xl border border-slate-700/30">
                            <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">Best Accuracy</div>
                            <div className="text-2xl font-bold text-white">97.30%</div>
                            <div className="text-emerald-400 text-xs">Custom CNN Model</div>
                        </div>
                        <div className="bg-slate-800/30 p-4 rounded-xl border border-slate-700/30">
                            <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">Dataset Size</div>
                            <div className="text-2xl font-bold text-white">54,000+</div>
                            <div className="text-emerald-400 text-xs">Leaf Images</div>
                        </div>
                    </div>
                </div>

                <div className="space-y-6 bg-slate-900/40 p-8 rounded-3xl border border-slate-800/50">
                    <h3 className="text-lg font-bold mb-4">Accuracy Comparison</h3>
                    {results.map((item, index) => (
                        <div key={index} className="space-y-2">
                            <div className="flex justify-between text-sm font-medium">
                                <span className={item.best ? "text-purple-400" : "text-slate-300"}>{item.model}</span>
                                <span className="text-white font-mono">{item.formatted}</span>
                            </div>
                            <div className="h-4 w-full bg-slate-800 rounded-full overflow-hidden relative">
                                {/* Background lines for visual scale */}
                                <div className="absolute inset-0 flex justify-between px-[1px]">
                                    {[0, 25, 50, 75, 100].map(m => <div key={m} className="h-full w-[1px] bg-slate-700/20" />)}
                                </div>

                                <motion.div
                                    initial={{ width: 0 }}
                                    whileInView={{ width: `${item.accuracy}%` }}
                                    transition={{ duration: 1, delay: 0.2 + (index * 0.1) }}
                                    className={`h-full rounded-full ${item.best ? "bg-gradient-to-r from-purple-500 to-pink-500" : "bg-slate-600"}`}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </Section>
    );
}
