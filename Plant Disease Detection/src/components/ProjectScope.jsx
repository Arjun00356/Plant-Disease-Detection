import React from 'react';
import { Section } from './Section';
import { CheckCircle2, Cpu, Eye } from 'lucide-react';

export function ProjectScope() {
    return (
        <Section id="scope">
            <div className="text-center mb-16">
                <h2 className="text-3xl md:text-4xl font-bold mb-4">Project Scope & Objectives</h2>
                <p className="text-slate-400 max-w-2xl mx-auto">
                    Our system goes beyond simple detection, leveraging machine learning to classify gases and provide actionable intelligence.
                </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
                {[
                    {
                        icon: <CheckCircle2 className="w-6 h-6" />,
                        title: "Multi-Species Support",
                        desc: "Accurately identifies various leaf diseases across 14 diverse plant species including Tomato, Potato, and Apple.",
                        color: "text-emerald-400",
                        bg: "bg-emerald-500/10"
                    },
                    {
                        icon: <Cpu className="w-6 h-6" />,
                        title: "Deep Learning Analysis",
                        desc: "Utilizes state-of-the-art CNN architectures to automatically extract and learn complex visual disease signatures.",
                        color: "text-blue-400",
                        bg: "bg-blue-500/10"
                    },
                    {
                        icon: <Eye className="w-6 h-6" />,
                        title: "Automated Diagnosis",
                        desc: "Designed to provide instant disease classification, enabling rapid intervention and targeted crop treatment.",
                        color: "text-purple-400",
                        bg: "bg-purple-500/10"
                    }
                ].map((item, idx) => (
                    <div key={idx} className="p-8 rounded-2xl bg-slate-900 border border-slate-800 hover:border-slate-700 transition-all hover:shadow-2xl hover:shadow-blue-900/10 group">
                        <div className={`w-12 h-12 rounded-xl ${item.bg} ${item.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                            {item.icon}
                        </div>
                        <h3 className="text-xl font-bold mb-3">{item.title}</h3>
                        <p className="text-slate-400 leading-relaxed">{item.desc}</p>
                    </div>
                ))}
            </div>


        </Section>
    );
}
