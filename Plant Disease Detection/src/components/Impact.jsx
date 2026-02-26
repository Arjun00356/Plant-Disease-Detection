import React from 'react';
import { Section } from './Section';
import { Zap, ShieldCheck, TrendingUp, Network, Building2 } from 'lucide-react';

export function Impact() {
    const impacts = [
        {
            icon: <Zap className="text-yellow-400" />,
            text: "Rapid and automated plant disease identification, reducing the time for diagnostic results from days to seconds."
        },
        {
            icon: <ShieldCheck className="text-emerald-400" />,
            text: "Early detection of pathogens, enabling targeted pesticide application and preventing widespread crop failure."
        },
        {
            icon: <TrendingUp className="text-blue-400" />,
            text: "Significant improvement in crop yield and quality by ensuring timely intervention and healthy plant growth."
        },
        {
            icon: <Network className="text-purple-400" />,
            text: "Empowers small-scale farmers with expert-level diagnostic tools accessible through low-cost mobile devices."
        },
        {
            icon: <Building2 className="text-orange-400" />,
            text: "Promotes sustainable agriculture by reducing unnecessary chemical usage through precisely localized treatments."
        }
    ];

    return (
        <Section id="impact">
            <div className="bg-slate-900/40 rounded-3xl p-8 md:p-12 border border-slate-800">
                <div className="text-center mb-10">
                    <h2 className="text-3xl md:text-4xl font-bold mb-6 text-center">Impact of the Project</h2>
                    <p className="text-slate-400 max-w-3xl mx-auto leading-relaxed text-center">
                        This project has significant real-world impact in the domain of digital agriculture and precision farming. As global food demand increases, detecting leaf diseases early is crucial for maintaining agricultural productivity and sustainability.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-10">
                    {impacts.map((item, i) => (
                        <div key={i} className="p-6 rounded-2xl bg-slate-950/50 border border-slate-800 hover:border-slate-700 transition-colors flex flex-col gap-4">
                            <div className="w-10 h-10 rounded-lg bg-slate-900 flex items-center justify-center">
                                {item.icon}
                            </div>
                            <p className="text-slate-300 text-sm leading-relaxed text-left">
                                {item.text}
                            </p>
                        </div>
                    ))}
                </div>

                <div className="text-center border-t border-slate-800/50 pt-8">
                    <p className="text-slate-400 font-medium max-w-2xl mx-auto text-center">
                        Overall, this project demonstrates how Deep Learning can revolutionize environmental stewardship and agricultural stability through intelligent plant diagnostics.
                    </p>
                </div>
            </div>
        </Section>
    );
}
