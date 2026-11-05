/**
 * Page « Démo » : envoyer une IRM (ou choisir un exemple) et voir la décision du modèle.
 */
import { useEffect, useState } from "react"

import { Dropzone } from "@/components/demo/Dropzone"
import { ResultView } from "@/components/demo/ResultView"
import { PageTitle } from "@/components/PageTitle"
import { Button } from "@/components/ui/button"
import { ApiError, predict, type Prediction } from "@/lib/api"
import { CLASS_ORDER, classLabel, EXAMPLES, type ClassName, type Example } from "@/lib/classes"

/** État de la page : rien envoyé, analyse en cours, résultat, ou erreur. */
type State =
  | { step: "idle" }
  | { step: "loading" }
  | { step: "done"; prediction: Prediction; imageUrl: string; truth?: ClassName }
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
  const analyse = async (file: File, truth?: ClassName) => {
    setState({ step: "loading" })
    try {
      const prediction = await predict(file)
      setState({ step: "done", prediction, imageUrl: URL.createObjectURL(file), truth })
    } catch (error) {
      const message = error instanceof ApiError ? error.message : "Une erreur est survenue."
      setState({ step: "error", message })
    }
  }

  /** Télécharge une image d'exemple du site puis l'analyse comme un fichier choisi. */
  const analyseExample = async (example: Example) => {
    setState({ step: "loading" })
    try {
      const blob = await (await fetch(example.src)).blob()
      const name = example.src.split("/").pop() ?? "exemple.jpg"
      await analyse(new File([blob], name, { type: blob.type }), example.label)
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
          <ResultView
            prediction={state.prediction}
            imageUrl={state.imageUrl}
            truth={state.truth}
          />
          <Button variant="outline" onClick={() => setState({ step: "idle" })}>
            Analyser une autre image
          </Button>
        </div>
      ) : (
        <div className="space-y-8">
          <Dropzone onFile={(file) => analyse(file)} disabled={state.step === "loading"} />

          <p className="rounded-xl bg-muted p-4 text-sm">
            <strong>À savoir :</strong> le modèle a appris sur un seul jeu d'IRM. Sur des images
            d'une autre source (autre machine, autre type de séquence, annotations ou flèches
            dessinées), il peut se tromper même avec une forte confiance. Il faut des coupes
            cérébrales du même type que les exemples ci-dessous.
          </p>

          <div aria-live="polite" className="min-h-6">
            {state.step === "loading" && <p className="text-muted-foreground">Analyse en cours…</p>}
            {state.step === "error" && (
              <p role="alert" className="text-destructive">
                {state.message}
              </p>
            )}
          </div>

          {EXAMPLES.length > 0 && (
            <section aria-label="Images d'exemple" className="space-y-5">
              <div className="space-y-1">
                <h2 className="font-heading text-lg font-semibold">Pas d'IRM sous la main ?</h2>
                <p className="text-sm text-muted-foreground">
                  Ces images ont été tirées au hasard dans le jeu de test, sans regarder si le
                  modèle les classe bien : il peut donc se tromper. La vraie classe est indiquée
                  pour chaque groupe.
                </p>
              </div>
              {CLASS_ORDER.map((name) => (
                <div key={name} className="space-y-2">
                  <h3 className="text-sm font-medium">Vraie classe : {classLabel(name)}</h3>
                  <ul className="grid grid-cols-3 gap-3 sm:grid-cols-6">
                    {EXAMPLES
                      .filter((example) => example.label === name)
                      .map((example, index) => (
                        <li key={example.src}>
                          <button
                            type="button"
                            disabled={state.step === "loading"}
                            onClick={() => analyseExample(example)}
                            aria-label={`Analyser l'exemple ${index + 1} (${classLabel(name)})`}
                            className="w-full overflow-hidden rounded-lg border outline-none transition hover:border-primary focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50"
                          >
                            <img
                              src={example.src}
                              alt=""
                              loading="lazy"
                              decoding="async"
                              width={512}
                              height={512}
                              className="aspect-square w-full bg-black object-cover"
                            />
                          </button>
                        </li>
                      ))}
                  </ul>
                </div>
              ))}
            </section>
          )}

          <p className="text-xs text-muted-foreground">
            Images d'exemple : jeu de données « Brain Tumor MRI Dataset » de Masoud Nickparvar
            (licence CC BY 4.0).
          </p>
        </div>
      )}
    </>
  )
}
