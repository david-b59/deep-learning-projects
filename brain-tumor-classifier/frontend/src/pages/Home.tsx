/**
 * Page d'accueil : en 30 secondes, à quoi sert le projet et où voir le résultat.
 */
import { lazy, Suspense, useState } from "react"
import { Link } from "react-router-dom"

import { ErrorBoundary } from "@/components/ErrorBoundary"
import { Reveal } from "@/components/Reveal"
import { buttonVariants } from "@/components/ui/button"
import { useTheme } from "@/lib/theme"
import { useReducedMotion } from "@/lib/use-reduced-motion"
import { cn } from "@/lib/utils"

// Three.js est lourd : le cerveau 3D n'est téléchargé que sur cette page, au moment voulu.
const BrainHero = lazy(() => import("@/components/BrainHero"))

/** Argument clé du projet, affiché sous forme de carte. */
const POINTS = [
  { title: "4 catégories", text: "Gliome, méningiome, tumeur de l'hypophyse, ou pas de tumeur." },
  { title: "94,4 % de réussite", text: "Mesurée sur des images de test, mises de côté jusqu'à la fin." },
  { title: "Une IA qui explique", text: "Une carte de chaleur montre où le modèle a regardé." },
]

/** Indique si le navigateur sait afficher de la 3D (WebGL). Sinon, on n'affiche pas le cerveau. */
function supportsWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas")
    return Boolean(canvas.getContext("webgl2") ?? canvas.getContext("webgl"))
  } catch {
    return false
  }
}

/** Page d'accueil avec accroche, cerveau 3D, bouton vers la démo et points clés. */
export function Home() {
  const { theme } = useTheme()
  const reduceMotion = useReducedMotion()
  // Calculé une seule fois : chaque test crée un contexte WebGL, qu'il ne faut pas multiplier.
  const [showBrain] = useState(supportsWebGL)

  return (
    <div className="space-y-12">
      <section className="grid items-center gap-8 py-4 md:grid-cols-2 md:py-8">
        <div className="min-w-0 space-y-6">
          <h1 className="font-heading text-4xl font-semibold tracking-tight sm:text-5xl">
            Une intelligence artificielle qui lit les IRM du cerveau.
          </h1>
          <p className="text-lg text-muted-foreground">
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
        </div>

        {/* Zone réservée même sans 3D : la page ne « saute » pas quand le cerveau arrive. */}
        <div className="mx-auto aspect-square w-full min-w-0 max-w-sm overflow-hidden md:max-w-none">
          {showBrain && (
            <ErrorBoundary>
              <Suspense fallback={null}>
                <BrainHero color={theme === "dark" ? "#a5b4fc" : "#4338ca"} still={reduceMotion} />
              </Suspense>
            </ErrorBoundary>
          )}
        </div>
      </section>

      <section aria-label="Points clés" className="grid gap-4 sm:grid-cols-3">
        {POINTS.map(({ title, text }, index) => (
          <Reveal key={title} delay={index * 0.1} className="rounded-xl border p-5">
            <h2 className="font-heading text-lg font-semibold">{title}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{text}</p>
          </Reveal>
        ))}
      </section>

      <p className="rounded-xl bg-muted p-4 text-sm">
        <strong>Important :</strong> démonstration technique, pas un dispositif médical. Ne
        remplace jamais l'avis d'un médecin.
      </p>
    </div>
  )
}
