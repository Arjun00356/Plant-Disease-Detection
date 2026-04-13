import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Upload, Leaf, Brain, Zap, AlertTriangle, CheckCircle,
    FlaskConical, Shield, ChevronDown, ChevronUp, Loader2,
    Microscope, Sprout, X,
} from 'lucide-react';

// ── Config ────────────────────────────────────────────────────────────────
const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

// ── Helpers ───────────────────────────────────────────────────────────────
const pct = (v) => `${(v * 100).toFixed(1)}%`;

const agreementConfig = {
    strong:  { color: 'emerald', label: 'Strong Agreement',  Icon: CheckCircle },
    partial: { color: 'yellow',  label: 'Partial Agreement', Icon: AlertTriangle },
    conflict:{ color: 'red',     label: 'Models Disagree',   Icon: AlertTriangle },
};

const severityColor = {
    None:     'text-emerald-400',
    Mild:     'text-yellow-400',
    Moderate: 'text-orange-400',
    Severe:   'text-red-400',
};

// ── Sub-components ─────────────────────────────────────────────────────────

function ConfBar({ value, colorClass = 'bg-emerald-500' }) {
    return (
        <div className="h-2 w-full rounded-full bg-slate-700/60 overflow-hidden">
            <motion.div
                className={`h-full rounded-full ${colorClass}`}
                initial={{ width: 0 }}
                animate={{ width: `${(value * 100).toFixed(1)}%` }}
                transition={{ duration: 0.7, ease: 'easeOut' }}
            />
        </div>
    );
}

function Badge({ text, color = 'emerald' }) {
    const map = {
        emerald: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
        yellow:  'bg-yellow-500/15  text-yellow-400  border-yellow-500/30',
        red:     'bg-red-500/15     text-red-400     border-red-500/30',
        blue:    'bg-blue-500/15    text-blue-400    border-blue-500/30',
        purple:  'bg-purple-500/15  text-purple-400  border-purple-500/30',
    };
    return (
        <span className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold border ${map[color] ?? map.emerald}`}>
            {text}
        </span>
    );
}

function SectionHeader({ icon: Icon, label, color = 'text-slate-300' }) {
    return (
        <div className="flex items-center gap-2 mb-4">
            <Icon size={18} className={color} />
            <h3 className={`text-sm font-bold uppercase tracking-widest ${color}`}>{label}</h3>
        </div>
    );
}

function ExpandableList({ items, label, icon: Icon, limit = 3 }) {
    const [expanded, setExpanded] = useState(false);
    if (!items || items.length === 0) return null;
    const shown = expanded ? items : items.slice(0, limit);
    return (
        <div className="mt-3">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                <Icon size={12} /> {label}
            </p>
            <ul className="space-y-1">
                {shown.map((item, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                        <span className="text-emerald-500 mt-0.5 shrink-0">•</span>
                        <span>{item}</span>
                    </li>
                ))}
            </ul>
            {items.length > limit && (
                <button
                    onClick={() => setExpanded(!expanded)}
                    className="mt-1.5 text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1 transition-colors"
                >
                    {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    {expanded ? 'Show less' : `+${items.length - limit} more`}
                </button>
            )}
        </div>
    );
}

// ─── CNN Panel ─────────────────────────────────────────────────────────────
function CNNPanel({ data }) {
    const topConf = data.top1_confidence;
    const confColor =
        topConf > 0.85 ? 'bg-emerald-500' :
        topConf > 0.60 ? 'bg-yellow-500' : 'bg-red-500';

    return (
        <div className="rounded-2xl border border-slate-700/60 bg-slate-900/70 p-5 flex flex-col gap-4">
            <SectionHeader icon={Brain} label="CNN Classifier" color="text-blue-400" />

            {/* Top-1 result */}
            <div className="rounded-xl bg-blue-500/10 border border-blue-500/20 p-4">
                <p className="text-xs text-blue-400 uppercase tracking-wider mb-1">Top prediction</p>
                <p className="text-lg font-bold text-slate-50 leading-tight">{data.top1_plant}</p>
                <p className="text-sm text-slate-400 mb-3">{data.top1_disease}</p>
                <div className="flex items-center gap-3">
                    <ConfBar value={topConf} colorClass={confColor} />
                    <span className="text-sm font-mono text-slate-300 shrink-0">{pct(topConf)}</span>
                </div>
            </div>

            {/* Top-5 */}
            <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Top-5 predictions</p>
                <div className="space-y-2.5">
                    {data.top_k.map((item, i) => (
                        <div key={i} className="flex flex-col gap-1">
                            <div className="flex justify-between items-baseline">
                                <span className="text-xs text-slate-300 truncate max-w-[75%]">
                                    <span className="text-slate-500 mr-1">#{i + 1}</span>
                                    {item.plant} — {item.disease}
                                </span>
                                <span className="text-xs font-mono text-slate-400 shrink-0">{pct(item.confidence)}</span>
                            </div>
                            <ConfBar
                                value={item.confidence}
                                colorClass={i === 0 ? 'bg-blue-500' : 'bg-slate-600'}
                            />
                        </div>
                    ))}
                </div>
            </div>

            {data.demo_mode && (
                <p className="text-xs text-yellow-500/80 bg-yellow-500/10 rounded-lg px-3 py-2 border border-yellow-500/20">
                    ⚠ CNN running in demo mode — model .pth not found.
                </p>
            )}
            <p className="text-xs text-slate-600 mt-auto">{data.model}</p>
        </div>
    );
}

// ─── VLM Panel ─────────────────────────────────────────────────────────────
function VLMPanel({ data }) {
    const sev = data.severity ?? 'Unknown';
    const treatment = data.treatment ?? {};

    return (
        <div className="rounded-2xl border border-slate-700/60 bg-slate-900/70 p-5 flex flex-col gap-4">
            <SectionHeader icon={Microscope} label="VLM Visual Analysis" color="text-purple-400" />

            {/* Top result */}
            <div className="rounded-xl bg-purple-500/10 border border-purple-500/20 p-4">
                <p className="text-xs text-purple-400 uppercase tracking-wider mb-1">VLM diagnosis</p>
                <p className="text-lg font-bold text-slate-50 leading-tight">{data.plant_species ?? '—'}</p>
                <p className="text-sm text-slate-400 mb-3">{data.disease_name ?? '—'}</p>
                <div className="flex items-center gap-2 flex-wrap">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                        <ConfBar value={data.confidence} colorClass="bg-purple-500" />
                        <span className="text-sm font-mono text-slate-300 shrink-0">{pct(data.confidence)}</span>
                    </div>
                    <Badge
                        text={`Severity: ${sev}`}
                        color={sev === 'None' ? 'emerald' : sev === 'Mild' ? 'yellow' : sev === 'Moderate' ? 'yellow' : 'red'}
                    />
                </div>
            </div>

            {/* Explanation */}
            {data.diagnosis_explanation && (
                <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Clinical reasoning</p>
                    <p className="text-sm text-slate-300 leading-relaxed">{data.diagnosis_explanation}</p>
                </div>
            )}

            {/* Symptoms */}
            <ExpandableList
                items={data.visual_symptoms}
                label="Visual symptoms"
                icon={Leaf}
                limit={3}
            />

            {/* Treatment */}
            {(treatment.immediate?.length > 0 || treatment.chemical?.length > 0 || treatment.biological?.length > 0) && (
                <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1">
                        <FlaskConical size={12} /> Treatment
                    </p>
                    {treatment.immediate?.length > 0 && (
                        <div className="mb-2">
                            <p className="text-xs text-orange-400 mb-1">Immediate actions</p>
                            {treatment.immediate.map((t, i) => (
                                <p key={i} className="text-sm text-slate-300">• {t}</p>
                            ))}
                        </div>
                    )}
                    {treatment.chemical?.length > 0 && (
                        <div className="mb-2">
                            <p className="text-xs text-red-400 mb-1">Chemical control</p>
                            {treatment.chemical.map((t, i) => (
                                <p key={i} className="text-sm text-slate-300">• {t}</p>
                            ))}
                        </div>
                    )}
                    {treatment.biological?.length > 0 && (
                        <div>
                            <p className="text-xs text-emerald-400 mb-1">Biological control</p>
                            {treatment.biological.map((t, i) => (
                                <p key={i} className="text-sm text-slate-300">• {t}</p>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Prevention */}
            <ExpandableList
                items={data.prevention}
                label="Prevention"
                icon={Shield}
                limit={3}
            />

            {data.error && (
                <p className="text-xs text-red-400/80 bg-red-500/10 rounded-lg px-3 py-2 border border-red-500/20">
                    ⚠ VLM analysis failed. Check your API key and provider settings.
                </p>
            )}
            <p className="text-xs text-slate-600 mt-auto">{data.model}</p>
        </div>
    );
}

// ─── Consensus Panel ────────────────────────────────────────────────────────
function ConsensusPanel({ data }) {
    const agr = agreementConfig[data.agreement] ?? agreementConfig.conflict;
    const AgrIcon = agr.Icon;
    const confVal = data.final_confidence;

    const confColor =
        confVal > 0.80 ? 'bg-emerald-500' :
        confVal > 0.55 ? 'bg-yellow-500' : 'bg-red-500';

    const confTextColor =
        confVal > 0.80 ? 'text-emerald-400' :
        confVal > 0.55 ? 'text-yellow-400' : 'text-red-400';

    return (
        <div className="rounded-2xl border border-emerald-500/30 bg-gradient-to-b from-emerald-950/40 to-slate-900/70 p-5 flex flex-col gap-4">
            <SectionHeader icon={Zap} label="Consensus Diagnosis" color="text-emerald-400" />

            {/* Agreement badge */}
            <div className={`flex items-center gap-2 rounded-xl px-4 py-2.5 border
                ${agr.color === 'emerald' ? 'bg-emerald-500/10 border-emerald-500/25' :
                  agr.color === 'yellow'  ? 'bg-yellow-500/10  border-yellow-500/25' :
                                            'bg-red-500/10     border-red-500/25'}`}>
                <AgrIcon size={16} className={
                    agr.color === 'emerald' ? 'text-emerald-400' :
                    agr.color === 'yellow'  ? 'text-yellow-400'  : 'text-red-400'} />
                <span className={`text-sm font-semibold
                    ${agr.color === 'emerald' ? 'text-emerald-300' :
                      agr.color === 'yellow'  ? 'text-yellow-300'  : 'text-red-300'}`}>
                    {data.agreement_label}
                </span>
            </div>

            {/* Final answer */}
            <div className="rounded-xl bg-slate-800/60 p-4">
                <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Final verdict</p>
                <p className="text-2xl font-bold text-slate-50 leading-tight">{data.final_plant}</p>
                <p className="text-base text-slate-400 mb-4">{data.final_disease}</p>

                <div className="flex items-center gap-3 mb-1">
                    <ConfBar value={confVal} colorClass={confColor} />
                    <span className={`text-sm font-mono font-bold shrink-0 ${confTextColor}`}>
                        {pct(confVal)}
                    </span>
                </div>
                <p className="text-xs text-slate-600">Consensus confidence</p>
            </div>

            {/* Weight breakdown */}
            <div className="grid grid-cols-2 gap-3">
                <div className="rounded-xl bg-blue-500/10 border border-blue-500/20 p-3 text-center">
                    <p className="text-lg font-bold text-blue-400">{(data.cnn_weight * 100).toFixed(0)}%</p>
                    <p className="text-xs text-slate-500 mt-0.5">CNN weight</p>
                </div>
                <div className="rounded-xl bg-purple-500/10 border border-purple-500/20 p-3 text-center">
                    <p className="text-lg font-bold text-purple-400">{(data.vlm_weight * 100).toFixed(0)}%</p>
                    <p className="text-xs text-slate-500 mt-0.5">VLM weight</p>
                </div>
            </div>

            {/* Per-model calls */}
            <div className="space-y-1 text-xs text-slate-500">
                <div className="flex justify-between">
                    <span className="text-blue-400/80">CNN top-1:</span>
                    <span className="text-slate-400 truncate ml-2 max-w-[70%] text-right">{data.cnn_top1 ?? '—'}</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-purple-400/80">VLM pick:</span>
                    <span className="text-slate-400 truncate ml-2 max-w-[70%] text-right">{data.vlm_prediction ?? '—'}</span>
                </div>
            </div>

            {/* Explanation */}
            {data.explanation && (
                <div className="rounded-xl bg-slate-800/40 px-4 py-3 border border-slate-700/40">
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1.5">How we decided</p>
                    <p className="text-sm text-slate-300 leading-relaxed">{data.explanation}</p>
                </div>
            )}

            {/* Top-5 consensus */}
            {data.top5 && data.top5.length > 0 && (
                <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Top-5 consensus</p>
                    <div className="space-y-2">
                        {data.top5.map((item, i) => (
                            <div key={i} className="flex flex-col gap-1">
                                <div className="flex justify-between items-baseline">
                                    <span className="text-xs text-slate-300 truncate max-w-[75%]">
                                        <span className="text-slate-600 mr-1">#{i + 1}</span>
                                        {item.plant} — {item.disease}
                                    </span>
                                    <span className="text-xs font-mono text-slate-400 shrink-0">{pct(item.confidence)}</span>
                                </div>
                                <ConfBar
                                    value={item.confidence}
                                    colorClass={i === 0 ? 'bg-emerald-500' : 'bg-slate-600'}
                                />
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── Upload Zone ────────────────────────────────────────────────────────────
function UploadZone({ onFileSelect, preview, onClear, loading }) {
    const inputRef = useRef(null);
    const [dragging, setDragging] = useState(false);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file && file.type.startsWith('image/')) onFileSelect(file);
    }, [onFileSelect]);

    const handleChange = (e) => {
        const file = e.target.files?.[0];
        if (file) onFileSelect(file);
    };

    if (preview) {
        return (
            <div className="relative rounded-2xl overflow-hidden border border-slate-700/60 bg-slate-900/60">
                <img
                    src={preview}
                    alt="Selected leaf"
                    className="w-full max-h-72 object-contain"
                />
                {!loading && (
                    <button
                        onClick={onClear}
                        className="absolute top-3 right-3 p-1.5 rounded-full bg-slate-900/80 border border-slate-600 text-slate-400 hover:text-slate-200 hover:border-slate-400 transition-colors"
                    >
                        <X size={14} />
                    </button>
                )}
                {loading && (
                    <div className="absolute inset-0 bg-slate-950/70 flex flex-col items-center justify-center gap-3">
                        <Loader2 size={36} className="text-emerald-400 animate-spin" />
                        <p className="text-sm text-slate-300 font-medium">Analysing leaf…</p>
                        <p className="text-xs text-slate-500">CNN + VLM running in parallel</p>
                    </div>
                )}
            </div>
        );
    }

    return (
        <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            className={`
                cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center
                transition-all duration-200 flex flex-col items-center justify-center gap-4
                ${dragging
                    ? 'border-emerald-500 bg-emerald-500/10'
                    : 'border-slate-700 bg-slate-900/40 hover:border-slate-500 hover:bg-slate-900/70'}
            `}
        >
            <div className={`p-4 rounded-2xl transition-colors ${dragging ? 'bg-emerald-500/20' : 'bg-slate-800'}`}>
                <Upload size={32} className={dragging ? 'text-emerald-400' : 'text-slate-400'} />
            </div>
            <div>
                <p className="text-slate-300 font-semibold">Drop a leaf image here</p>
                <p className="text-slate-500 text-sm mt-1">or click to browse — JPEG / PNG, max 10 MB</p>
            </div>
            <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleChange}
            />
        </div>
    );
}

// ─── Main Component ─────────────────────────────────────────────────────────
export function DiagnosisEngine() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const resultsRef = useRef(null);

    const handleFileSelect = (f) => {
        setFile(f);
        setPreview(URL.createObjectURL(f));
        setResult(null);
        setError(null);
    };

    const handleClear = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
    };

    const handleSubmit = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);
        setResult(null);

        const form = new FormData();
        form.append('file', file);

        try {
            const res = await fetch(`${API_URL}/api/diagnose`, {
                method: 'POST',
                body: form,
            });

            if (!res.ok) {
                const body = await res.json().catch(() => ({}));
                throw new Error(body.detail ?? `Server error ${res.status}`);
            }

            const data = await res.json();
            setResult(data);

            // Scroll to results
            setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <section className="py-24 px-6" id="diagnose">
            <div className="max-w-7xl mx-auto">

                {/* Header */}
                <motion.div
                    className="text-center mb-14"
                    initial={{ opacity: 0, y: 24 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.6 }}
                >
                    <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 text-sm font-semibold mb-5">
                        <Sprout size={14} /> Live Diagnosis Engine
                    </div>
                    <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-emerald-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                        CNN + VLM Dual Diagnosis
                    </h2>
                    <p className="text-slate-400 max-w-2xl mx-auto text-lg leading-relaxed">
                        Upload a plant leaf photo. Our{' '}
                        <span className="text-blue-400 font-semibold">Custom CNN</span> and a{' '}
                        <span className="text-purple-400 font-semibold">Vision-Language Model</span>{' '}
                        independently analyse it, then a{' '}
                        <span className="text-emerald-400 font-semibold">confidence-weighted consensus</span>{' '}
                        gives the final verdict.
                    </p>
                </motion.div>

                {/* Architecture legend */}
                <motion.div
                    className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-10"
                    initial={{ opacity: 0, y: 16 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                >
                    {[
                        { icon: Brain, color: 'blue', label: 'CNN Classifier', desc: '97.30% accuracy · 38-class PlantVillage · 65% consensus weight' },
                        { icon: Microscope, color: 'purple', label: 'Vision-Language Model', desc: 'Claude / GPT-4o visual reasoning · clinical explanation · 35% weight' },
                        { icon: Zap, color: 'emerald', label: 'Weighted Consensus', desc: 'Probability-blended final verdict · agreement analysis · top-5 ranking' },
                    ].map(({ icon: Icon, color, label, desc }) => (
                        <div key={label} className={`rounded-2xl border p-4
                            ${color === 'blue'   ? 'border-blue-500/20   bg-blue-500/5'   :
                              color === 'purple' ? 'border-purple-500/20 bg-purple-500/5' :
                                                   'border-emerald-500/20 bg-emerald-500/5'}`}>
                            <div className="flex items-center gap-2 mb-2">
                                <Icon size={16} className={
                                    color === 'blue'   ? 'text-blue-400'   :
                                    color === 'purple' ? 'text-purple-400' : 'text-emerald-400'} />
                                <span className="text-sm font-semibold text-slate-200">{label}</span>
                            </div>
                            <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
                        </div>
                    ))}
                </motion.div>

                {/* Upload + button */}
                <motion.div
                    className="max-w-xl mx-auto mb-6"
                    initial={{ opacity: 0, y: 16 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                >
                    <UploadZone
                        onFileSelect={handleFileSelect}
                        preview={preview}
                        onClear={handleClear}
                        loading={loading}
                    />

                    {file && !loading && (
                        <motion.button
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            onClick={handleSubmit}
                            className="mt-4 w-full py-3.5 rounded-2xl font-bold text-sm
                                bg-gradient-to-r from-emerald-600 to-blue-600
                                hover:from-emerald-500 hover:to-blue-500
                                text-white transition-all duration-200 shadow-lg
                                shadow-emerald-900/30 hover:shadow-emerald-800/50"
                        >
                            Diagnose Leaf
                        </motion.button>
                    )}

                    {/* Error */}
                    <AnimatePresence>
                        {error && (
                            <motion.div
                                initial={{ opacity: 0, y: -8 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0 }}
                                className="mt-4 rounded-xl bg-red-500/10 border border-red-500/30 px-4 py-3 flex items-start gap-3"
                            >
                                <AlertTriangle size={16} className="text-red-400 mt-0.5 shrink-0" />
                                <div>
                                    <p className="text-sm font-semibold text-red-300">Diagnosis failed</p>
                                    <p className="text-xs text-red-400/80 mt-0.5">{error}</p>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </motion.div>

                {/* Results — three panels */}
                <AnimatePresence>
                    {result && (
                        <motion.div
                            ref={resultsRef}
                            initial={{ opacity: 0, y: 32 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.5 }}
                            className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8"
                        >
                            <CNNPanel data={result.cnn} />
                            <VLMPanel data={result.vlm} />
                            <ConsensusPanel data={result.consensus} />
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </section>
    );
}
