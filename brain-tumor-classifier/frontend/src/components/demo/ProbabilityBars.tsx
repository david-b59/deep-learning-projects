/**
 * Barres de probabilités animées : une barre par classe, la plus probable en couleur forte.
 */
import { useEffect, useState } from "react"

import { classLabel } from "@/lib/classes"
import { cn } from "@/lib/utils"

interface ProbabilityBarsProps {
  /** Probabilités par classe (somme = 1). */
  probabilities: Record<string, number>
  /** Classe prédite par le modèle. */
  predicted: string
}

/** Affiche les probabilités triées de la plus forte à la plus faible, avec animation d'entrée. */
export function ProbabilityBars({ probabilities, predicted }: ProbabilityBarsProps) {
  // Les barres partent de 0 puis grandissent : la transition CSS fait l'animation.
  const [shown, setShown] = useState(false)
  useEffect(() => {
    const frame = requestAnimationFrame(() => setShown(true))
    return () => cancelAnimationFrame(frame)
  }, [])

  const rows = Object.entries(probabilities).sort((a, b) => b[1] - a[1])
  return (
    <ul className="space-y-3" aria-label="Probabilités par catégorie">
      {rows.map(([name, value]) => (
        <li key={name}>
          <div className="mb-1 flex justify-between text-sm">
            <span className={cn(name === predicted && "font-semibold")}>{classLabel(name)}</span>
            <span className="tabular-nums text-muted-foreground">
              {(value * 100).toFixed(1).replace(".", ",")} %
            </span>
          </div>
          <div
            role="progressbar"
            aria-label={classLabel(name)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(value * 100)}
            className="h-2.5 overflow-hidden rounded-full bg-muted"
          >
            <div
              className={cn(
                "h-full rounded-full transition-[width] duration-700 ease-out motion-reduce:transition-none",
                name === predicted ? "bg-primary" : "bg-muted-foreground/40",
              )}
              style={{ width: shown ? `${value * 100}%` : "0%" }}
            />
          </div>
        </li>
      ))}
    </ul>
  )
}
