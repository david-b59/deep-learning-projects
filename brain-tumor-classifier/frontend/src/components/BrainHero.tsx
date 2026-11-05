/**
 * Cerveau 3D de l'accueil : un nuage de points qui tourne doucement et suit un peu la souris.
 *
 * On utilise Three.js directement (et seulement les classes nécessaires, pour que le navigateur
 * télécharge le moins possible). Ce composant est importé « à la demande » par la page d'accueil
 * (voir `lazy` dans Home.tsx).
 */
import { useEffect, useRef } from "react"
import {
  BufferAttribute,
  BufferGeometry,
  MathUtils,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Scene,
  WebGLRenderer,
} from "three"

import { buildBrainPoints } from "@/lib/brainShape"

interface BrainHeroProps {
  /** Couleur des points (change avec le thème clair/sombre). */
  color: string
  /** Si vrai, aucune animation (préférence « moins d'animations ») : une seule image fixe. */
  still: boolean
}

/** Vitesse de rotation, en radians par seconde. */
const ROTATION_SPEED = 0.25

/** Scène 3D complète. Décorative : masquée aux lecteurs d'écran. */
export default function BrainHero({ color, still }: BrainHeroProps) {
  const container = useRef<HTMLDivElement>(null)
  const material = useRef<PointsMaterial | null>(null)
  // Fonction qui redessine une image ; remplie par l'effet principal, utilisée par l'effet couleur.
  const redraw = useRef<() => void>(() => {})

  // Effet principal : crée la scène, lance l'animation, et nettoie tout à la fin.
  useEffect(() => {
    const host = container.current
    if (!host) return

    const renderer = new WebGLRenderer({ alpha: true, antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    host.appendChild(renderer.domElement)
    renderer.domElement.style.display = "block"

    const scene = new Scene()
    const camera = new PerspectiveCamera(45, 1, 0.1, 100)
    camera.position.set(0, 0.2, 3.1)

    const geometry = new BufferGeometry()
    geometry.setAttribute("position", new BufferAttribute(buildBrainPoints(), 3))
    const pointsMaterial = new PointsMaterial({
      color: "#ffffff", // la vraie couleur est posée par l'effet suivant
      size: 0.014,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    })
    material.current = pointsMaterial
    const cloud = new Points(geometry, pointsMaterial)
    scene.add(cloud)

    const draw = () => renderer.render(scene, camera)
    redraw.current = draw

    // Adapte la scène à la taille de la zone (téléphone, tablette, bureau).
    const resize = () => {
      const { clientWidth: width, clientHeight: height } = host
      if (width === 0 || height === 0) return
      renderer.setSize(width, height)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      draw()
    }
    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(host)
    resize()

    // Animation : seulement si le visiteur ne l'a pas désactivée, et seulement quand visible.
    const anim = { frame: 0, visible: true, pointerY: 0, last: performance.now() }
    const onPointerMove = (event: PointerEvent) => {
      anim.pointerY = (event.clientY / window.innerHeight) * 2 - 1 // -1 (haut) à 1 (bas)
    }
    const tick = (now: number) => {
      anim.frame = requestAnimationFrame(tick)
      if (!anim.visible) return
      cloud.rotation.y += ((now - anim.last) / 1000) * ROTATION_SPEED
      cloud.rotation.x = MathUtils.lerp(cloud.rotation.x, -anim.pointerY * 0.25, 0.05)
      draw()
      anim.last = now
    }
    let visibilityObserver: IntersectionObserver | null = null
    if (!still) {
      window.addEventListener("pointermove", onPointerMove)
      visibilityObserver = new IntersectionObserver(([entry]) => {
        anim.visible = entry.isIntersecting
        anim.last = performance.now() // évite un saut de rotation au retour à l'écran
      })
      visibilityObserver.observe(host)
      anim.frame = requestAnimationFrame(tick)
    }

    return () => {
      cancelAnimationFrame(anim.frame)
      window.removeEventListener("pointermove", onPointerMove)
      visibilityObserver?.disconnect()
      resizeObserver.disconnect()
      geometry.dispose()
      pointsMaterial.dispose()
      renderer.dispose()
      renderer.forceContextLoss() // libère tout de suite le contexte WebGL
      renderer.domElement.remove()
      material.current = null
      redraw.current = () => {}
    }
  }, [still])

  // Effet couleur : change la couleur sans recréer toute la scène.
  useEffect(() => {
    material.current?.color.set(color)
    redraw.current()
  }, [color, still])

  return <div ref={container} aria-hidden="true" className="size-full" />
}
