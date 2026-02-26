import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function Section({ children, className, id }) {
    return (
        <section id={id} className={twMerge("py-20 px-8 max-w-7xl mx-auto", className)}>
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-100px" }}
                transition={{ duration: 0.6 }}
            >
                {children}
            </motion.div>
        </section>
    );
}
