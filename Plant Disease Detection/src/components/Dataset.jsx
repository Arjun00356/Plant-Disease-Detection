import React from 'react';
import { Section } from './Section';
import { Database, FileSpreadsheet, Users } from 'lucide-react';

export function Dataset() {
    return (
        <Section id="dataset" className="bg-slate-900/30">
            <div className="text-center mb-12">
                <h2 className="text-3xl md:text-4xl font-bold mb-4">Dataset Overview</h2>
                <p className="text-slate-400">
                    Our model is trained on the extensive PlantVillage dataset, containing thousands of high-resolution images across multiple plant species and disease categories.
                </p>
            </div>

            <div className="bg-slate-800/40 rounded-3xl border border-slate-700/50 overflow-hidden backdrop-blur-sm">
                <div className="grid md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-slate-700/50">
                    {[
                        { label: "Total Samples", value: "54,300+", icon: <Database /> },
                        { label: "Categories", value: "38 Classes", icon: <FileSpreadsheet /> },
                        { label: "Plant Species", value: "14 Types", icon: <Users /> }
                    ].map((stat, i) => (
                        <div key={i} className="p-8 flex flex-col items-center text-center hover:bg-slate-700/20 transition-colors">
                            <div className="mb-4 text-emerald-400 bg-emerald-500/10 p-3 rounded-full">{stat.icon}</div>
                            <div className="text-3xl font-bold text-white mb-1">{stat.value}</div>
                            <div className="text-sm text-slate-400 uppercase tracking-wider">{stat.label}</div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="mt-12 overflow-x-auto">
                <table className="w-full text-left text-sm text-slate-400">
                    <thead className="text-xs uppercase bg-slate-800/50 text-slate-300">
                        <tr>
                            <th className="px-6 py-4 rounded-tl-lg">Plant Species</th>
                            <th className="px-6 py-4 text-left">Common Diseases Tracked</th>
                            <th className="px-6 py-4 rounded-tr-lg">Pathogen Type</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Apple</td>
                            <td className="px-6 py-4 text-left">Scab, Black Rot, Cedar Apple Rust, Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Fungal/Bacterial</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Blueberry</td>
                            <td className="px-6 py-4 text-left">Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Normal</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Cherry</td>
                            <td className="px-6 py-4 text-left">Powdery Mildew, Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Fungal</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Corn (Maize)</td>
                            <td className="px-6 py-4 text-left">Cercospora Leaf Spot, Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Fungal</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Grape</td>
                            <td className="px-6 py-4 text-left">Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Fungal/Bacterial</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Orange</td>
                            <td className="px-6 py-4 text-left">Haunglongbing (Citrus Greening)</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Viral/Bacterial</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Peach</td>
                            <td className="px-6 py-4 text-left">Bacterial Spot, Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Bacterial</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Pepper, Bell</td>
                            <td className="px-6 py-4 text-left">Bacterial Spot, Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Bacterial</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Potato</td>
                            <td className="px-6 py-4 text-left">Early Blight, Late Blight, Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Fungal/Oomycete</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Raspberry</td>
                            <td className="px-6 py-4 text-left">Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Normal</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Soybean</td>
                            <td className="px-6 py-4 text-left">Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Normal</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Squash</td>
                            <td className="px-6 py-4 text-left">Powdery Mildew</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Fungal</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Strawberry</td>
                            <td className="px-6 py-4 text-left">Leaf Scorch, Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Fungal</td>
                        </tr>

                        <tr className="bg-slate-900/20">
                            <td className="px-6 py-4 font-medium text-white">Tomato</td>
                            <td className="px-6 py-4 text-left">Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Yellow Leaf Curl, Mosaic Virus, Healthy</td>
                            <td className="px-6 py-4 font-mono text-xs text-emerald-400">Multimodal</td>
                        </tr>
                    </tbody>

                </table>
            </div>
        </Section>
    );
}
