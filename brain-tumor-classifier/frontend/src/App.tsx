/**
 * Racine de l'application : thème, routeur et liste des pages.
 */
import { MotionConfig } from "motion/react"
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"

import { Layout } from "@/components/layout/Layout"
import { ThemeProvider } from "@/lib/theme"
import { About } from "@/pages/About"
import { Demo } from "@/pages/Demo"
import { Home } from "@/pages/Home"
import { HowItWorks } from "@/pages/HowItWorks"
import { Results } from "@/pages/Results"

/** Application complète : chaque adresse (URL) affiche une page dans la mise en page commune. */
export default function App() {
  return (
    <ThemeProvider>
      {/* reducedMotion="user" : les animations s'adaptent à la préférence du système. */}
      <MotionConfig reducedMotion="user">
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<Home />} />
              <Route path="demo" element={<Demo />} />
              <Route path="comment-ca-marche" element={<HowItWorks />} />
              <Route path="resultats" element={<Results />} />
              <Route path="a-propos" element={<About />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </MotionConfig>
    </ThemeProvider>
  )
}
