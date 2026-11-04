/**
 * Thème clair / sombre.
 *
 * Le choix est mémorisé dans le navigateur (localStorage). Au premier passage, le thème sombre
 * est utilisé (défaut du projet) ;
 * le visiteur peut basculer en clair. Le thème sombre s'active en ajoutant la classe « dark » sur <html>.
 */
import { createContext, useContext, useEffect, useState, type ReactNode } from "react"

export type Theme = "light" | "dark"

const STORAGE_KEY = "theme"

interface ThemeContextValue {
  theme: Theme
  toggleTheme: () => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)

/** Lit le thème mémorisé, sinon le thème sombre (défaut du projet). */
function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === "light" || saved === "dark") return saved
  } catch {
    // localStorage peut être indisponible (navigation privée) : on ignore.
  }
  return "dark" // thème sombre par défaut (choix de l'architecture du projet)
}

/** Fournit le thème courant à toute l'application et applique la classe sur <html>. */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initialTheme)

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark")
    try {
      localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      // Mémorisation impossible : le thème reste appliqué pour cette visite.
    }
  }, [theme])

  const toggleTheme = () => setTheme((current) => (current === "dark" ? "light" : "dark"))
  return <ThemeContext value={{ theme, toggleTheme }}>{children}</ThemeContext>
}

/** Donne accès au thème courant et à la bascule clair/sombre. */
export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext)
  if (!context) throw new Error("useTheme doit être utilisé dans un ThemeProvider")
  return context
}
