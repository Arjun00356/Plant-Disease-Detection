import React from 'react';
import { Section } from './Section';
import { ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';

export function Workflow() {
    const steps = [
        {
            title: "Data Collection",
            description: "Acquiring the PlantVillage dataset containing 54,000+ high-resolution leaf images across 38 distinct plant-disease categories."
        },
        {
            title: "Preprocessing",
            description: "Resizing images to model input sizes (e.g., 224x224), normalizing pixel values, and applying data augmentation to improve generalization."
        },
        {
            title: "Data Splitting",
            description: "Dividing the image dataset into training (80%) and validation (20%) sets to monitor model performance during the learning process."
        },
        {
            title: "Architecture Selection",
            description: "Selecting Deep Learning architectures including Custom CNN, ResNet18, and MobileNetV2 for feature extraction and classification."
        },
        {
            title: "Model Training",
            description: "Training the selected models using transfer learning and fine-tuning techniques to achieve high accuracy on leaf disease patterns."
        },
        {
            title: "Performance Tuning",
            description: "Optimizing hyperparameters and using techniques like dropout and batch normalization to ensure robust and stable model convergence."
        },
        {
            title: "Final Evaluation",
            description: "Testing models on unseen images and generating accuracy metrics and confusion matrices for comprehensive performance analysis."
        }
    ];

    return (
        <Section id="workflow">
            <div className="text-center mb-16">
                <h2 className="text-3xl md:text-4xl font-bold mb-4">Implementation Workflow</h2>
                <p className="text-slate-400">A detailed breakdown of our machine learning pipeline.</p>
            </div>

            <div className="relative max-w-5xl mx-auto">
                {/* Central timeline line */}
                <div className="hidden md:block absolute left-[50%] top-0 bottom-0 w-0.5 bg-slate-800 -translate-x-1/2" />

                <div className="space-y-12 relative">
                    {steps.map((step, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true, margin: "-50px" }}
                            transition={{ delay: index * 0.1 }}
                            className={`flex flex-col md:flex-row items-center gap-8 ${index % 2 === 0 ? 'md:flex-row-reverse' : ''}`}
                        >
                            {/* Content Card */}
                            <div className="flex-1 w-full relative group">
                                <div className="p-8 rounded-3xl bg-slate-900/80 border border-slate-800 hover:border-blue-500/30 transition-all relative overflow-hidden">
                                    <h3 className="text-xl font-bold text-slate-100 mb-3">{step.title}</h3>
                                    <p className="text-slate-400 text-sm leading-relaxed border-t border-slate-700/50 pt-3">
                                        {step.description}
                                    </p>
                                </div>

                                {/* Connection Arrow */}
                                <div className={`hidden md:block absolute top-1/2 ${index % 2 === 0 ? '-left-12' : '-right-12'} -translate-y-1/2 text-slate-700`}>
                                    {index < steps.length - 1 && (
                                        index % 2 === 0 ? <ArrowRight className="rotate-180" /> : <ArrowRight />
                                    )}
                                </div>
                            </div>

                            {/* Center Node */}
                            <div className="relative z-10 flex-shrink-0 w-12 h-12 rounded-full bg-slate-950 border-4 border-slate-800 flex items-center justify-center text-slate-300 font-bold text-sm shadow-xl">
                                {index + 1}
                            </div>

                            {/* Spacer for alternate side */}
                            <div className="flex-1 hidden md:block" />
                        </motion.div>
                    ))}
                </div>
            </div>
        </Section>
    );
}
