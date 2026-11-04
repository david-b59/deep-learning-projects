/**
 * Page « À propos » (squelette : le contenu arrive dans une étape suivante).
 */
import { PageTitle } from "@/components/PageTitle"

/** À propos. */
export function About() {
  return (
    <>
      <PageTitle
        title="À propos"
        intro="Le projet, les données et les limites à connaître."
      />
      <p className="rounded-xl border border-dashed p-6 text-sm text-muted-foreground">
        Bientôt : contexte du projet, source des données et licences (étape 8c).
      </p>
    </>
  )
}
