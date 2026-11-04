/**
 * En-tête du site : logo, navigation entre les pages et bouton de thème clair/sombre.
 */
import { Moon, Sun } from "lucide-react"
import { NavLink, Link } from "react-router-dom"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useTheme } from "@/lib/theme"

/** Pages du site, dans l'ordre d'affichage du menu. */
const LINKS = [
  { to: "/", label: "Accueil" },
  { to: "/demo", label: "Démo" },
  { to: "/comment-ca-marche", label: "Comment ça marche" },
  { to: "/resultats", label: "Résultats" },
  { to: "/a-propos", label: "À propos" },
]

/** Barre de navigation fixe en haut de page. */
export function Header() {
  const { theme, toggleTheme } = useTheme()
  return (
    <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4">
        <Link to="/" className="font-heading text-base font-semibold tracking-tight">
          Brain Tumor Classifier
        </Link>
        <nav aria-label="Navigation principale" className="flex items-center gap-1 overflow-x-auto">
          {LINKS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                cn(
                  "rounded-md px-3 py-1.5 text-sm whitespace-nowrap transition-colors hover:bg-muted",
                  isActive ? "bg-muted font-medium text-foreground" : "text-muted-foreground",
                )
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          aria-label={theme === "dark" ? "Passer au thème clair" : "Passer au thème sombre"}
        >
          {theme === "dark" ? <Sun /> : <Moon />}
        </Button>
      </div>
    </header>
  )
}
