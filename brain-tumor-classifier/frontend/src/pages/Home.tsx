/**
 * Page d'accueil : en 30 secondes, à quoi sert le projet et où voir le résultat.
 */
import { Link } from "react-router-dom"

import { buttonVariants } from "@/components/ui/button"
import { cn } from "@/lib/utils"

/** Argument clé du projet, affiché sous forme de carte. */
const POINTS = [
  { title: "4 catégories", text: "Gliome, méningiome, tumeur de l'hypophyse, ou pas de tumeur." },
  { title: "94,4 % de réussite", text: "Mesurée sur des images que le modèle n'avait jamais vues." },
  { title: "Une IA qui explique", text: "Une carte de chaleur montre où le modèle a regardé." },
]

/** Page d'accueil avec accroche, bouton vers la démo et points clés. */
export function Home() {
  return (
    <div className="space-y-12">
      <section className="space-y-6 py-8">
        <h1 className="font-heading max-w-3xl text-4xl font-semibold tracking-tight sm:text-5xl">
          Une intelligence artificielle qui lit les IRM du cerveau.
        </h1>
        <p className="max-w-2xl text-lg text-muted-foreground">
          Envoie une IRM cérébrale : le modèle propose un diagnostic parmi quatre catégories et
          montre sur l'image les zones qui ont guidé sa décision.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link to="/demo" className={cn(buttonVariants({ size: "lg" }))}>
            Essayer la démo
          </Link>
          <Link
            to="/comment-ca-marche"
            className={cn(buttonVariants({ size: "lg", variant: "outline" }))}
          >
            Comment ça marche
          </Link>
        </div>
      </section>

      <section aria-label="Points clés" className="grid gap-4 sm:grid-cols-3">
        {POINTS.map(({ title, text }) => (
          <div key={title} className="rounded-xl border p-5">
            <h2 className="font-heading text-lg font-semibold">{title}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{text}</p>
          </div>
        ))}
      </section>

      <p className="rounded-xl bg-muted p-4 text-sm">
        <strong>Important :</strong> démonstration technique, pas un dispositif médical. Ne
        remplace jamais l'avis d'un médecin.
      </p>
    </div>
  )
}
