/**
 * Mise en page commune : en-tête, contenu de la page courante, pied de page.
 */
import { useEffect } from "react"
import { Outlet, useLocation } from "react-router-dom"

import { Footer } from "@/components/layout/Footer"
import { Header } from "@/components/layout/Header"

/** Cadre de toutes les pages ; <Outlet /> accueille la page choisie par le routeur. */
export function Layout() {
  const { pathname } = useLocation()

  // Au changement de page, on remonte en haut (comportement attendu d'un site classique).
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])

  return (
    <div className="flex min-h-screen flex-col">
      <a
        href="#contenu"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded-md focus:bg-background focus:px-3 focus:py-2"
      >
        Aller au contenu
      </a>
      <Header />
      <main id="contenu" className="mx-auto w-full max-w-6xl flex-1 px-4 py-10">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}
