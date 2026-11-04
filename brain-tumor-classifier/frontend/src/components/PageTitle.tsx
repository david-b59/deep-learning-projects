/**
 * Titre de page réutilisable (titre + sous-titre), identique sur toutes les pages.
 */

/** Affiche le titre principal d'une page et une phrase d'introduction. */
export function PageTitle({ title, intro }: { title: string; intro: string }) {
  return (
    <div className="mb-8 space-y-2">
      <h1 className="font-heading text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h1>
      <p className="max-w-2xl text-muted-foreground">{intro}</p>
    </div>
  )
}
