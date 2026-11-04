/**
 * Page « Démo » (squelette : le contenu arrive dans une étape suivante).
 */
import { PageTitle } from "@/components/PageTitle"

/** Démo. */
export function Demo() {
  return (
    <>
      <PageTitle
        title="Démo"
        intro="Envoie une IRM cérébrale et découvre ce que le modèle en pense."
      />
      <p className="rounded-xl border border-dashed p-6 text-sm text-muted-foreground">
        Bientôt : glisser-déposer d'une IRM, résultat et carte de chaleur (étape 8b).
      </p>
    </>
  )
}
