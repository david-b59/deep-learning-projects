/**
 * Page « Résultats » : scores du modèle sur le jeu de test, matrice de confusion et limites.
 *
 * Les chiffres viennent de l'API (`/metrics-summary`), donc du modèle réellement publié.
 */
import { useEffect, useState } from "react"

import { PageTitle } from "@/components/PageTitle"
import { ApiError, fetchMetrics, type MetricsSummary } from "@/lib/api"
import { classLabel } from "@/lib/classes"
import { cn } from "@/lib/utils"

/** Limites connues du modèle, dites sans détour (reprises de la fiche du modèle). */
const LIMITS = [
  "Les gliomes sont les moins bien reconnus : 16 % ne sont pas retrouvés. 44 tumeurs (gliomes et méningiomes) sont classées « pas de tumeur », l'erreur la plus grave dans un contexte réel.",
  "Le modèle ne regarde pas toujours la tumeur : dans plusieurs erreurs, la zone chaude de la carte est ailleurs (crâne, orbite).",
  "La confiance peut tromper : certaines erreurs ont une probabilité de 100 %.",
  "Une seule source de données : le modèle n'a pas été testé sur d'autres hôpitaux ni d'autres machines.",
  "Aucune validation clinique, aucun avis médical.",
]

/** Formate une proportion (0 à 1) en pourcentage français, ex. 0,9438 devient « 94,4 % ». */
function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits).replace(".", ",")} %`
}

/** Formate un score entre 0 et 1 avec 3 décimales et la virgule française, ex. « 0,985 ». */
function score(value: number): string {
  return value.toFixed(3).replace(".", ",")
}

/** Tableau de la matrice de confusion : lignes = vraie classe, colonnes = classe prédite. */
function ConfusionMatrix({ metrics }: { metrics: MetricsSummary }) {
  const { class_names: names, confusion_matrix: matrix } = metrics
  return (
    <div className="space-y-2">
      <p className="text-sm text-muted-foreground">
        Chaque ligne est la vraie classe, chaque colonne la réponse du modèle. La diagonale (en
        couleur) compte les bonnes réponses ; les nombres en rouge sont des erreurs.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[28rem] border-collapse text-sm">
          <thead>
            <tr>
              <th scope="col" className="p-2 text-left font-medium">
                Vraie classe ↓ / Réponse →
              </th>
              {names.map((name) => (
                <th key={name} scope="col" className="p-2 text-right font-medium">
                  {classLabel(name)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => {
              const total = row.reduce((sum, value) => sum + value, 0)
              return (
                <tr key={names[i]} className="border-t">
                  <th scope="row" className="p-2 text-left font-medium">
                    {classLabel(names[i])}
                  </th>
                  {row.map((value, j) => (
                    <td
                      key={names[j]}
                      className={cn(
                        "p-2 text-right tabular-nums",
                        i === j && "bg-primary/15 font-semibold",
                        i !== j && value > 0 && "text-destructive",
                      )}
                      title={`${percent(value / total)} de la ligne`}
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/** Page des résultats : chiffre clé, scores par classe, matrice et limites. */
export function Results() {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Charge les scores une seule fois, à l'ouverture de la page.
  useEffect(() => {
    fetchMetrics()
      .then(setMetrics)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Une erreur est survenue."))
  }, [])

  return (
    <>
      <PageTitle
        title="Résultats"
        intro="Les performances du modèle sur 1 584 images de test, utilisées une seule fois, à la fin."
      />

      {error && (
        <p role="alert" className="mb-8 text-destructive">
          {error}
        </p>
      )}
      {!metrics && !error && <p className="mb-8 text-muted-foreground">Chargement…</p>}

      {metrics && (
        <div className="space-y-10">
          <section aria-label="Résultat global" className="rounded-xl border p-6">
            <p className="font-heading text-5xl font-semibold tabular-nums">
              {percent(metrics.accuracy)}
            </p>
            <p className="mt-1 text-muted-foreground">
              d'images de test bien classées. Sur les données de validation (qui ont servi à
              choisir le modèle), le score était de 97,7 % : le chiffre à retenir est celui du
              test.
            </p>
          </section>

          <section aria-label="Scores par catégorie" className="space-y-3">
            <h2 className="font-heading text-xl font-semibold">Par catégorie</h2>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[28rem] border-collapse text-sm">
                <thead>
                  <tr className="text-left">
                    <th scope="col" className="p-2 font-medium">Catégorie</th>
                    <th scope="col" className="p-2 text-right font-medium">Précision</th>
                    <th scope="col" className="p-2 text-right font-medium">Rappel</th>
                    <th scope="col" className="p-2 text-right font-medium">F1</th>
                    <th scope="col" className="p-2 text-right font-medium">Images</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.class_names.map((name) => {
                    const s = metrics.per_class[name]
                    return (
                      <tr key={name} className="border-t">
                        <th scope="row" className="p-2 text-left font-medium">
                          {classLabel(name)}
                        </th>
                        <td className="p-2 text-right tabular-nums">{score(s.precision)}</td>
                        <td className="p-2 text-right tabular-nums">{score(s.recall)}</td>
                        <td className="p-2 text-right tabular-nums">{score(s.f1)}</td>
                        <td className="p-2 text-right tabular-nums">{s.support}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="text-sm text-muted-foreground">
              La précision dit « quand le modèle annonce cette catégorie, a-t-il raison ? ». Le
              rappel dit « parmi les vraies images de cette catégorie, combien le modèle
              retrouve-t-il ? ». Le F1 les résume.
            </p>
          </section>

          <section aria-label="Matrice de confusion" className="space-y-3">
            <h2 className="font-heading text-xl font-semibold">Où le modèle se trompe</h2>
            <ConfusionMatrix metrics={metrics} />
          </section>
        </div>
      )}

      <section aria-label="Limites" className="mt-10 space-y-3">
        <h2 className="font-heading text-xl font-semibold">Les limites à connaître</h2>
        <ul className="list-disc space-y-2 pl-5 text-sm">
          {LIMITS.map((limit) => (
            <li key={limit}>{limit}</li>
          ))}
        </ul>
      </section>
    </>
  )
}
