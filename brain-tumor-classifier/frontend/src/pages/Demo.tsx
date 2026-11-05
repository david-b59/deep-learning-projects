/**
 * Page « Démo » : envoyer une IRM (ou choisir un exemple) et voir la décision du modèle.
 */
import { useEffect, useState } from "react"

import { Dropzone } from "@/components/demo/Dropzone"
import { ResultView } from "@/components/demo/ResultView"
import { PageTitle } from "@/components/PageTitle"
import { Button } from "@/components/ui/button"
import { ApiError, predict, type Prediction } from "@/lib/api"
import { classLabel, EXAMPLES } from "@/lib/classes"

/** État de la page : rien envoyé, analyse en cours, résultat, ou erreur. */
type State =
  | { step: "idle" }
  | { step: "loading" }
  | { step: "done"; prediction: Prediction; imageUrl: string }
  | { step: "error"; message: string }

/** Page de démonstration complète : dépôt d'image, exemples, analyse et résultat. */
export function Demo() {
  const [state, setState] = useState<State>({ step: "idle" })

  // Libère l'adresse locale de l'image quand elle n'est plus affichée.
  useEffect(() => {
    return () => {
      if (state.step === "done") URL.revokeObjectURL(state.imageUrl)
    }
  }, [state])

  /** Envoie le fichier à l'API et met à jour l'état selon la réponse. */
  const analyse = async (file: File) => {
    setState({ step: "loading" })
    try {
      const prediction = await predict(file)
      setState({ step: "done", prediction, imageUrl: URL.createObjectURL(file) })
    } catch (error) {
      const message = error instanceof ApiError ? error.message : "Une erreur est survenue."
      setState({ step: "error", message })
    }
  }

  /** Télécharge une image d'exemple du site puis l'analyse comme un fichier choisi. */
  const analyseExample = async (src: string) => {
    setState({ step: "loading" })
    try {
      const blob = await (await fetch(src)).blob()
      await analyse(new File([blob], src.split("/").pop() ?? "exemple.jpg", { type: blob.type }))
    } catch {
      setState({ step: "error", message: "Impossible de charger l'image d'exemple." })
    }
  }

  return (
    <>
      <PageTitle
        title="Démo"
        intro="Envoie une IRM cérébrale et découvre ce que le modèle en pense. Rien n'est enregistré."
      />

      {state.step === "done" ? (
        <div className="space-y-6">
          <ResultView prediction={state.prediction} imageUrl={state.imageUrl} />
          <Button variant="outline" onClick={() => setState({ step: "idle" })}>
            Analyser une autre image
          </Button>
        </div>
      ) : (
        <div className="space-y-8">
          <Dropzone onFile={analyse} disabled={state.step === "loading"} />

          <div aria-live="polite" className="min-h-6">
            {state.step === "loading" && <p className="text-muted-foreground">Analyse en cours…</p>}
            {state.step === "error" && (
              <p role="alert" className="text-destructive">
                {state.message}
              </p>
            )}
          </div>

          <section aria-label="Images d'exemple" className="space-y-3">
            <h2 className="font-heading text-lg font-semibold">Pas d'IRM sous la main ?</h2>
            <p className="text-sm text-muted-foreground">
              Clique sur un exemple tiré du jeu de test (images jamais vues pendant l'entraînement).
            </p>
            <ul className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              {EXAMPLES.map(({ name, src }) => (
                <li key={name}>
                  <button
                    type="button"
                    disabled={state.step === "loading"}
                    onClick={() => analyseExample(src)}
                    className="w-full overflow-hidden rounded-xl border text-left outline-none transition hover:border-primary focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50"
                  >
                    <img
                      src={src}
                      alt={`IRM d'exemple : ${classLabel(name)}`}
                      className="aspect-square w-full bg-black object-cover"
                    />
                    <span className="block p-2 text-sm">{classLabel(name)}</span>
                  </button>
                </li>
              ))}
            </ul>
          </section>

          <p className="text-xs text-muted-foreground">
            Images d'exemple : jeu de données « Brain Tumor MRI Dataset » de Masoud Nickparvar
            (licence CC BY 4.0).
          </p>
        </div>
      )}
    </>
  )
}
