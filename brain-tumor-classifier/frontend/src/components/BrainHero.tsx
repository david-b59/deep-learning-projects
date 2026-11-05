/**
 * Cerveau 3D de l'accueil : un nuage de points qui tourne doucement et suit un peu la souris.
 *
 * Ce composant charge Three.js (lourd) : il est importé « à la demande » par la page d'accueil
 * (voir `lazy` dans Home.tsx), pour ne pas ralentir le reste du site.
 */
import { Canvas, useFrame } from "@react-three/fiber"
import { useEffect, useMemo, useRef } from "react"
import * as THREE from "three"

import { buildBrainPoints } from "@/lib/brainShape"

interface BrainCloudProps {
  /** Couleur des points (change avec le thème clair/sombre). */
  color: string
  /** Si vrai, le cerveau reste immobile. */
  still: boolean
}

/** Le nuage de points lui-même, animé à chaque image. */
function BrainCloud({ color, still }: BrainCloudProps) {
  const points = useRef<THREE.Points>(null)
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute("position", new THREE.BufferAttribute(buildBrainPoints(), 3))
    return g
  }, [])

  // Libère la mémoire graphique quand le composant disparaît.
  useEffect(() => () => geometry.dispose(), [geometry])

  useFrame((state, delta) => {
    const cloud = points.current
    if (!cloud || still) return
    cloud.rotation.y += delta * 0.25 // rotation lente
    // La souris incline légèrement le cerveau (effet de profondeur).
    cloud.rotation.x = THREE.MathUtils.lerp(cloud.rotation.x, -state.pointer.y * 0.25, 0.05)
  })

  return (
    <points ref={points} geometry={geometry}>
      <pointsMaterial
        color={color}
        size={0.014}
        sizeAttenuation
        transparent
        opacity={0.85}
        depthWrite={false}
      />
    </points>
  )
}

interface BrainHeroProps {
  /** Couleur des points. */
  color: string
  /** Si vrai, aucune animation (préférence « moins d'animations »). */
  still: boolean
}

/** Scène 3D complète. Décorative : masquée aux lecteurs d'écran. */
export default function BrainHero({ color, still }: BrainHeroProps) {
  return (
    <div aria-hidden="true" className="size-full">
      <Canvas
        camera={{ position: [0, 0.2, 3.1], fov: 45 }}
        dpr={[1, 2]}
        frameloop={still ? "demand" : "always"}
        gl={{ alpha: true, antialias: true }}
      >
        <BrainCloud color={color} still={still} />
      </Canvas>
    </div>
  )
}
