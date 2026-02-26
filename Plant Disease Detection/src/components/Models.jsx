import React from 'react';
import { Section } from './Section';
import { BrainCircuit, Network, Zap } from 'lucide-react';
import { motion } from 'framer-motion';

export function Models() {
    const mainModel = {
        id: "cnn",
        name: "Custom CNN",
        icon: <BrainCircuit />,
        accuracy: "97.30%",
        desc: "A custom-built Convolutional Neural Network designed for efficient leaf feature extraction and classification. Serving as our high-performance architecture.",
        best: true,
        confusionMatrixImg: "/assets/confusionMatrix.png",
        color: "text-emerald-400",
        bg: "bg-emerald-500/10"
    };

    const sideModels = [
        {
            id: "mobilenet",
            name: "MobileNetV2",
            icon: <Zap />,
            accuracy: "96.44%",
            desc: "Efficient inverted residual architecture optimized for mobile and edge devices.",
            color: "text-blue-400",
            bg: "bg-blue-500/10"
        },
        {
            id: "resnet",
            name: "ResNet18",
            icon: <Network />,
            accuracy: "95.96%",
            desc: "Residual network enabling deeper learning by mitigating vanishing gradients.",
            color: "text-purple-400",
            bg: "bg-purple-500/10"
        }
    ];

    return (
        <Section id="models">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-center">Deep Learning Architectures</h2>
            <p className="text-center text-slate-400 mb-12 max-w-2xl mx-auto">
                We evaluated three state-of-the-art architectures to determine the most effective model for plant disease classification.
            </p>

            <div className="grid lg:grid-cols-12 gap-8 items-stretch">
                {/* Main Model - Large Left Card */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    className="lg:col-span-7 flex"
                >
                    <div className="w-full rounded-3xl border border-emerald-500/50 shadow-lg shadow-emerald-500/10 bg-slate-900/50 overflow-hidden flex flex-col">
                        <div className="p-8 border-b border-slate-800 bg-slate-900/30">
                            <div className="flex items-start justify-between mb-4">
                                <div className={`w-14 h-14 rounded-2xl ${mainModel.bg} ${mainModel.color} flex items-center justify-center`}>
                                    {mainModel.icon}
                                </div>
                                <div className="text-right">
                                    <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Accuracy</div>
                                    <div className="text-3xl font-bold text-white font-mono">{mainModel.accuracy}</div>
                                </div>
                            </div>

                            <h3 className="text-2xl font-bold flex items-center gap-3 mb-2">
                                {mainModel.name}
                                <span className="text-xs bg-emerald-500 text-white px-3 py-1 rounded-full uppercase tracking-wider font-medium">Best Model</span>
                            </h3>
                            <p className="text-slate-400 text-sm leading-relaxed max-w-xl">
                                {mainModel.desc}
                            </p>
                        </div>

                        <div className="p-6 bg-slate-950/30 flex-1 flex flex-col items-center justify-center">
                            <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 w-full text-center text-left">Confusion Matrix Visualization</h4>
                            <div className="relative group w-full max-w-[520px] rounded-xl overflow-hidden border border-slate-800 shadow-xl bg-black aspect-[519/416]">
                                <img
                                    src={mainModel.confusionMatrixImg}
                                    alt={`Inference for ${mainModel.name}`}
                                    className="w-full h-auto object-cover"
                                    width="519"
                                    height="416"
                                />
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* Side Models - Stacked Right Column */}
                <div className="lg:col-span-5 flex flex-col gap-6">
                    {sideModels.map((model, idx) => (
                        <motion.div
                            key={model.id}
                            initial={{ opacity: 0, x: 20 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.1 * (idx + 1) }}
                            className="flex-1 flex"
                        >
                            <div className="w-full p-6 rounded-3xl border border-slate-800 bg-slate-900/50 hover:bg-slate-900/80 transition-colors flex flex-col justify-center">
                                <div className="flex items-center justify-between mb-4">
                                    <div className={`w-12 h-12 rounded-xl ${model.bg} ${model.color} flex items-center justify-center`}>
                                        {model.icon}
                                    </div>
                                    <div className="text-right">
                                        <div className="text-slate-500 text-[10px] uppercase tracking-wider mb-0.5">Accuracy</div>
                                        <div className="text-2xl font-bold text-white font-mono">{model.accuracy}</div>
                                    </div>
                                </div>
                                <h3 className="text-xl font-bold text-white mb-2">{model.name}</h3>
                                <p className="text-slate-400 text-xs leading-relaxed">
                                    {model.desc}
                                </p>
                                <div className="mt-4 pt-4 border-t border-slate-800/50 flex items-center justify-between">
                                    {/* <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Model Efficiency</span> */}
                                    <div className="h-1.5 w-24 bg-slate-800 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full ${model.color.replace('text', 'bg')}`}
                                            style={{ width: model.accuracy }}
                                        />
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>
        </Section>
    );
}
