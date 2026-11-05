/**
 * Forme du cerveau 3D de l'accueil, dessinée par le code (aucun fichier 3D, donc aucune licence).
 *
 * Le cerveau est un nuage de points : deux hémisphères séparés par une scissure, des plis à la
 * surface (circonvolutions), un cervelet et un tronc cérébral. La forme est stylisée, pas
 * anatomiquement exacte.
 */

/** Nombre de points de chaque partie du cerveau. */
const HEMISPHERE_POINTS = 7600
const CEREBELLUM_POINTS = 1300
const BRAINSTEM_POINTS = 260

/** Angle d'or : répartit des points de façon régulière sur une sphère (spirale de Fibonacci). */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5))

/** Renvoie le point numéro `i` sur la sphère unité, parmi `count` points régulièrement répartis. */
function spherePoint(i: number, count: number): [number, number, number] {
  const y = 1 - (2 * (i + 0.5)) / count
  const radius = Math.sqrt(1 - y * y)
  const angle = GOLDEN_ANGLE * i
  return [Math.cos(angle) * radius, y, Math.sin(angle) * radius]
}

/**
 * Point du cerveau principal (les deux hémisphères) pour une direction donnée de la sphère unité.
 *
 * On déforme une sphère : ellipsoïde allongé, scissure au sommet, plis de surface, bombement
 * des lobes temporaux et dessous aplati.
 */
function cerebrumPoint(x: number, y: number, z: number): [number, number, number] {
  // Plis de surface : deux ondes qui se croisent donnent un relief de circonvolutions.
  const folds =
    0.085 * Math.sin(10 * x + 3 * z) * Math.sin(9 * y + 2 * x) +
    0.05 * Math.sin(15 * z + 2 * y) * Math.sin(12 * x) +
    0.03 * Math.sin(24 * y + 5 * z)
  const scale = 1 + folds

  let px = x * 0.78 * scale
  let py = (y < 0 ? y * 0.62 : y * 0.7) * scale // dessous plus plat que le dessus
  const pz = z * 1.0 * scale

  // Lobes temporaux : le bas des hémisphères s'élargit un peu.
  if (y < 0) px *= 1 + 0.18 * -y

  // Scissure interhémisphérique : un creux au sommet, le long de l'axe central.
  py -= 0.14 * Math.exp(-((px / 0.07) ** 2)) * Math.max(0, y)

  // Séparation des deux hémisphères.
  px += Math.sign(x) * 0.04
  return [px, py, pz]
}

/**
 * Construit tous les points du cerveau.
 *
 * @returns Un tableau plat [x0, y0, z0, x1, y1, z1, …], prêt pour une géométrie Three.js.
 */
export function buildBrainPoints(): Float32Array {
  const total = HEMISPHERE_POINTS + CEREBELLUM_POINTS + BRAINSTEM_POINTS
  const positions = new Float32Array(total * 3)
  let offset = 0
  const push = (x: number, y: number, z: number) => {
    positions[offset++] = x
    positions[offset++] = y
    positions[offset++] = z
  }

  // Cerveau principal.
  for (let i = 0; i < HEMISPHERE_POINTS; i++) {
    const [x, y, z] = spherePoint(i, HEMISPHERE_POINTS)
    push(...cerebrumPoint(x, y, z))
  }

  // Cervelet : petit ellipsoïde strié, à l'arrière et en dessous.
  for (let i = 0; i < CEREBELLUM_POINTS; i++) {
    const [x, y, z] = spherePoint(i, CEREBELLUM_POINTS)
    const ridges = 1 + 0.07 * Math.sin(34 * y)
    push(x * 0.46 * ridges, -0.5 + y * 0.2 * ridges, -0.72 + z * 0.3 * ridges)
  }

  // Tronc cérébral : un petit cylindre qui descend sous le cerveau.
  for (let i = 0; i < BRAINSTEM_POINTS; i++) {
    const t = i / BRAINSTEM_POINTS
    const angle = GOLDEN_ANGLE * i
    push(Math.cos(angle) * 0.09, -0.35 - t * 0.6, -0.2 + Math.sin(angle) * 0.09)
  }
  return positions
}
