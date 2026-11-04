/**
 * Page « Comment ça marche » (squelette : le contenu arrive dans une étape suivante).
 */
import { PageTitle } from "@/components/PageTitle"

/** Comment ça marche. */
export function HowItWorks() {
  return (
    <>
      <PageTitle
        title="Comment ça marche"
        intro="Du pixel au diagnostic, expliqué simplement."
      />
      <p className="rounded-xl border border-dashed p-6 text-sm text-muted-foreground">
        Bientôt : schéma du fonctionnement du modèle et de la carte de chaleur (étape 8c).
      </p>
    </>
  )
}
