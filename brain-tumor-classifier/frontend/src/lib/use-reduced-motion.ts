/**
 * Détecte si le visiteur a demandé moins d'animations (réglage du système).
 */
import { useSyncExternalStore } from "react"

const QUERY = "(prefers-reduced-motion: reduce)"

/** Souscrit aux changements de la préférence « moins d'animations ». */
function subscribe(callback: () => void): () => void {
  const media = window.matchMedia(QUERY)
  media.addEventListener("change", callback)
  return () => media.removeEventListener("change", callback)
}

/** Renvoie `true` si le visiteur préfère les animations réduites (et se met à jour si ça change). */
export function useReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(QUERY).matches,
    () => false,
  )
}
