/**
 * Barrière d'erreur : empêche un élément secondaire (ici le cerveau 3D) de faire tomber la page.
 *
 * React n'a pas d'équivalent « hook » pour attraper les erreurs d'affichage : il faut une classe.
 */
import { Component, type ReactNode } from "react"

interface ErrorBoundaryProps {
  children: ReactNode
  /** Ce qu'on affiche à la place en cas d'erreur (rien par défaut). */
  fallback?: ReactNode
}

interface ErrorBoundaryState {
  hasError: boolean
}

/** Affiche `fallback` si un composant enfant lève une erreur (chargement raté, WebGL cassé…). */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false }

  /** Passe en mode erreur dès qu'un enfant plante. */
  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true }
  }

  /** Garde une trace dans la console du navigateur pour le débogage. */
  componentDidCatch(error: Error): void {
    console.error("Élément décoratif indisponible :", error)
  }

  render(): ReactNode {
    return this.state.hasError ? (this.props.fallback ?? null) : this.props.children
  }
}
