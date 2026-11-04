/**
 * Pied de page : rappel permanent que l'outil n'est pas un dispositif médical.
 */

/** Bandeau d'avertissement et crédits, affiché sur toutes les pages. */
export function Footer() {
  return (
    <footer className="border-t">
      <div className="mx-auto max-w-6xl space-y-1 px-4 py-6 text-sm text-muted-foreground">
        <p className="font-medium text-foreground">
          Démonstration technique, pas un dispositif médical.
        </p>
        <p>
          Projet réalisé dans le cadre de mon Master Machine Learning Engineering, puis prolongé
          à titre personnel.
        </p>
      </div>
    </footer>
  )
}
