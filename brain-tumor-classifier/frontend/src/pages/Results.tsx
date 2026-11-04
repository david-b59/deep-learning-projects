/**
 * Page « Résultats » (squelette : le contenu arrive dans une étape suivante).
 */
import { PageTitle } from "@/components/PageTitle"

/** Résultats. */
export function Results() {
  return (
    <>
      <PageTitle
        title="Résultats"
        intro="Les performances du modèle, avec ses points forts et ses limites."
      />
      <p className="rounded-xl border border-dashed p-6 text-sm text-muted-foreground">
        Bientôt : scores par catégorie et matrice de confusion (étape 8c).
      </p>
    </>
  )
}
