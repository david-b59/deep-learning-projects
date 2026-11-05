/**
 * En-tête du site : logo, navigation entre les pages et bouton de thème clair/sombre.
 *
 * Sur grand écran, les liens sont alignés dans l'en-tête. Sur téléphone, ils sont rangés dans un
 * menu déroulant ouvert par un bouton.
 */
import { Menu, Moon, Sun, X } from "lucide-react"
import { useState } from "react"
import { Link, NavLink } from "react-router-dom"

import { Button } from "@/components/ui/button"
import { useTheme } from "@/lib/theme"
import { cn } from "@/lib/utils"

/** Pages du site, dans l'ordre d'affichage du menu. */
const LINKS = [
  { to: "/", label: "Accueil" },
  { to: "/demo", label: "Démo" },
  { to: "/comment-ca-marche", label: "Comment ça marche" },
  { to: "/resultats", label: "Résultats" },
  { to: "/a-propos", label: "À propos" },
]

/**
 * Liste des liens de navigation.
 *
 * `vertical` l'affiche en colonne (menu téléphone) ; `onNavigate` est appelé quand on clique un
 * lien (sert à refermer le menu téléphone).
 */
function NavLinks({ vertical = false, onNavigate }: { vertical?: boolean; onNavigate?: () => void }) {
  return (
    <>
      {LINKS.map(({ to, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              "rounded-md px-3 py-2 text-sm whitespace-nowrap transition-colors hover:bg-muted",
              vertical && "block",
              isActive ? "bg-muted font-medium text-foreground" : "text-muted-foreground",
            )
          }
        >
          {label}
        </NavLink>
      ))}
    </>
  )
}

/** Barre de navigation fixe en haut de page. */
export function Header() {
  const { theme, toggleTheme } = useTheme()
  const [open, setOpen] = useState(false)

  return (
    <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4">
        <Link
          to="/"
          className="font-heading text-base font-semibold tracking-tight whitespace-nowrap"
        >
          Brain Tumor Classifier
        </Link>

        <nav aria-label="Navigation principale" className="hidden items-center gap-1 md:flex">
          <NavLinks />
        </nav>

        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Passer au thème clair" : "Passer au thème sombre"}
          >
            {theme === "dark" ? <Sun /> : <Moon />}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setOpen((value) => !value)}
            aria-expanded={open}
            aria-controls="menu-mobile"
            aria-label={open ? "Fermer le menu" : "Ouvrir le menu"}
          >
            {open ? <X /> : <Menu />}
          </Button>
        </div>
      </div>

      {open && (
        <nav
          id="menu-mobile"
          aria-label="Navigation principale"
          className="flex flex-col gap-1 border-t px-4 py-3 md:hidden"
        >
          <NavLinks vertical onNavigate={() => setOpen(false)} />
        </nav>
      )}
    </header>
  )
}
