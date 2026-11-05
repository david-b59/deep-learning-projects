/**
 * Apparition en fondu avec léger glissement quand un élément entre dans l'écran.
 *
 * La préférence « moins d'animations » du système est respectée (voir `MotionConfig` dans App.tsx).
 */
import { motion } from "motion/react"
import type { ReactNode } from "react"

interface RevealProps {
  children: ReactNode
  /** Retard avant l'animation, en secondes (permet d'échelonner plusieurs éléments). */
  delay?: number
  className?: string
}

/** Enveloppe un contenu qui apparaît une seule fois au défilement. */
export function Reveal({ children, delay = 0, className }: RevealProps) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.5, delay, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  )
}
