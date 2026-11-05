/**
 * Apparition en fondu avec léger glissement quand un élément entre dans l'écran.
 *
 * Réalisé en CSS pur : un `IntersectionObserver` (API du navigateur) détecte l'arrivée à
 * l'écran, puis une transition CSS fait l'animation. Aucune bibliothèque d'animation n'est
 * nécessaire, ce qui garde le site léger. Si le visiteur préfère moins d'animations (réglage
 * du système), l'élément est affiché tout de suite, sans mouvement.
 */
import { useEffect, useRef, useState, type ReactNode } from "react"

import { cn } from "@/lib/utils"

interface RevealProps {
  children: ReactNode
  /** Retard avant l'animation, en secondes (permet d'échelonner plusieurs éléments). */
  delay?: number
  className?: string
}

/** Enveloppe un contenu qui apparaît une seule fois, au défilement. */
export function Reveal({ children, delay = 0, className }: RevealProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [shown, setShown] = useState(false)

  useEffect(() => {
    const element = ref.current
    if (!element) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShown(true)
          observer.disconnect() // l'animation ne se joue qu'une fois
        }
      },
      { rootMargin: "0px 0px -40px 0px" },
    )
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  return (
    <div
      ref={ref}
      className={cn(
        className,
        "transition duration-500 ease-out motion-reduce:transition-none",
        shown
          ? "translate-y-0 opacity-100"
          : "translate-y-4 opacity-0 motion-reduce:translate-y-0 motion-reduce:opacity-100",
      )}
      style={{ transitionDelay: `${delay}s` }}
    >
      {children}
    </div>
  )
}
