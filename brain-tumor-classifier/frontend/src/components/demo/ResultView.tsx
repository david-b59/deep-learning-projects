/**
 * Résultat d'une analyse : IRM + carte Grad-CAM (opacité réglable) et probabilités.
 */
import { useState } from "react"

import { ProbabilityBars } from "@/components/demo/ProbabilityBars"
import type { Prediction } from "@/lib/api"
import { CLASS_INFO, classLabel, type ClassName } from "@/lib/classes"

interface ResultViewProps {
  /** Réponse de l'API. */
  prediction: Prediction
  /** Adresse (URL locale) de l'IRM envoyée, affichée sous la carte. */
  imageUrl: string
  /** Vraie classe de l'image, connue seulement pour les exemples du site. */
  truth?: ClassName
}

/** Affiche la décision, les barres de probabilités et l'IRM avec la carte de chaleur. */
export function ResultView({ prediction, imageUrl, truth }: ResultViewProps) {
  // Opacité de la carte Grad-CAM (0 = IRM seule, 100 = IRM avec la chaleur au maximum).
  const [opacity, setOpacity] = useState(70)
  const { predicted_class: predicted, probabilities } = prediction
  const confidence = probabilities[predicted] ?? 0
  const description = CLASS_INFO[predicted as ClassName]?.description

  return (
    <div className="grid gap-8 md:grid-cols-2">
      <section aria-label="Image analysée" className="space-y-3">
        <div className="relative mx-auto aspect-square w-full max-w-md overflow-hidden rounded-xl border bg-black">
          <img
            src={imageUrl}
            alt="IRM envoyée"
            className="absolute inset-0 size-full object-fill"
          />
          <img
            src={`data:image/png;base64,${prediction.gradcam_png_base64}`}
            alt="Carte de chaleur Grad-CAM : les zones chaudes ont le plus compté dans la décision"
            className="absolute inset-0 size-full object-fill"
            style={{ opacity: opacity / 100 }}
          />
        </div>
        <label className="mx-auto flex max-w-md items-center gap-3 text-sm">
          <span className="whitespace-nowrap">Carte de chaleur</span>
          <input
            type="range"
            min={0}
            max={100}
            value={opacity}
            onChange={(event) => setOpacity(Number(event.target.value))}
            className="w-full accent-primary"
          />
          <span className="w-12 text-right whitespace-nowrap tabular-nums">{opacity} %</span>
        </label>
        <p className="mx-auto max-w-md text-xs text-muted-foreground">
          Les zones rouges et jaunes sont celles qui ont le plus influencé le modèle. Elles ne
          correspondent pas toujours à la tumeur : à interpréter avec prudence.
        </p>
      </section>

      <section aria-label="Résultat" className="space-y-6">
        <div aria-live="polite">
          <p className="text-sm text-muted-foreground">Le modèle propose</p>
          <h2 className="font-heading text-3xl font-semibold">{classLabel(predicted)}</h2>
          <p className="text-muted-foreground">
            avec {(confidence * 100).toFixed(1).replace(".", ",")} % de confiance
          </p>
          {description && <p className="mt-2 text-sm">{description}</p>}
          {truth && (
            <p className="mt-3 rounded-lg border p-3 text-sm">
              Vraie classe de cet exemple : <strong>{classLabel(truth)}</strong>.{" "}
              {truth === predicted
                ? "Le modèle a bien classé cette image."
                : "Le modèle s'est trompé sur cette image."}
            </p>
          )}
        </div>
        <ProbabilityBars probabilities={probabilities} predicted={predicted} />
        <p className="rounded-xl bg-muted p-4 text-sm">{prediction.disclaimer}</p>
      </section>
    </div>
  )
}
