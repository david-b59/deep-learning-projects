/**
 * Fonctions pures de l'inférence dans le navigateur.
 *
 * Elles reproduisent, côté navigateur, ce que fait le serveur Python (`ml/src/brain_mri/predict.py`
 * et `explain.py`) : prétraitement de l'IRM, choix de la classe, normalisation de la carte
 * Grad-CAM, agrandissement bilinéaire, palette « jet » et superposition sur l'image.
 *
 * Aucune de ces fonctions ne touche au DOM ni au modèle : elles se testent donc avec Vitest.
 */

/** Côté de l'image attendue par le modèle, en pixels. */
export const IMAGE_SIZE = 224

/** Moyenne ImageNet par canal (R, G, B), utilisée à l'entraînement. */
export const MEAN = [0.485, 0.456, 0.406] as const

/** Écart-type ImageNet par canal (R, G, B), utilisé à l'entraînement. */
export const STD = [0.229, 0.224, 0.225] as const

/** Opacité de la carte de chaleur sur l'IRM (même valeur que `OVERLAY_ALPHA` côté serveur). */
export const OVERLAY_ALPHA = 0.45

/**
 * Calcule, pour chaque pixel de sortie d'un axe, la plage de pixels d'entrée et leurs poids.
 *
 * Filtre « triangle » (bilinéaire) dont le support est élargi du facteur d'échelle quand on
 * réduit l'image (c'est l'antialias), coordonnées au centre des pixels, poids normalisés
 * (somme = 1). Même calcul que le noyau « aa » d'ATen utilisé par PyTorch.
 *
 * @param inSize - Taille d'entrée sur cet axe.
 * @param outSize - Taille de sortie sur cet axe.
 * @returns Pour chaque pixel de sortie : indice du premier pixel d'entrée et poids.
 */
function axisWeights(inSize: number, outSize: number): { start: number; weights: Float64Array }[] {
  const scale = inSize / outSize
  const filterScale = Math.max(scale, 1) // > 1 seulement quand on réduit
  const support = filterScale // support du triangle = 1, élargi du facteur d'échelle
  const result: { start: number; weights: Float64Array }[] = []
  for (let i = 0; i < outSize; i++) {
    const center = scale * (i + 0.5)
    const start = Math.max(Math.trunc(center - support + 0.5), 0)
    const end = Math.min(Math.trunc(center + support + 0.5), inSize)
    const weights = new Float64Array(end - start)
    let total = 0
    for (let j = 0; j < weights.length; j++) {
      const distance = Math.abs((j + start - center + 0.5) / filterScale)
      weights[j] = distance < 1 ? 1 - distance : 0
      total += weights[j]
    }
    for (let j = 0; j < weights.length; j++) weights[j] /= total // normalisation
    result.push({ start, weights })
  }
  return result
}

/**
 * Redimensionne une image RGBA en `size` × `size` comme `torchvision.transforms.v2.Resize(
 * antialias=True)` sur un tenseur uint8 (le prétraitement du serveur).
 *
 * Interpolation bilinéaire avec antialias : une passe horizontale, puis une passe verticale.
 * Comme dans PyTorch pour les images 8 bits, le résultat de chaque passe est arrondi à
 * l'entier le plus proche (0 à 255). L'alpha est ignoré, il est rendu à 255.
 *
 * @param rgba - Pixels RGBA de l'image d'origine (taille native).
 * @param width - Largeur d'origine.
 * @param height - Hauteur d'origine.
 * @param size - Côté de l'image de sortie.
 * @returns Les pixels RGBA de l'image `size` × `size`.
 */
export function resizeAntialias(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  size: number = IMAGE_SIZE,
): Uint8ClampedArray<ArrayBuffer> {
  const columns = axisWeights(width, size)
  const rows = axisWeights(height, size)
  // Passe horizontale : height lignes x size colonnes, valeurs arrondies à l'entier.
  const middle = new Uint8ClampedArray(height * size * 3)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < size; x++) {
      const { start, weights } = columns[x]
      for (let c = 0; c < 3; c++) {
        let sum = 0
        for (let j = 0; j < weights.length; j++) sum += weights[j] * rgba[(y * width + start + j) * 4 + c]
        middle[(y * size + x) * 3 + c] = Math.round(sum) // Uint8ClampedArray borne à 0-255
      }
    }
  }
  // Passe verticale sur le résultat de la première.
  const out = new Uint8ClampedArray(size * size * 4)
  for (let y = 0; y < size; y++) {
    const { start, weights } = rows[y]
    for (let x = 0; x < size; x++) {
      for (let c = 0; c < 3; c++) {
        let sum = 0
        for (let j = 0; j < weights.length; j++) sum += weights[j] * middle[((start + j) * size + x) * 3 + c]
        out[(y * size + x) * 4 + c] = Math.round(sum)
      }
      out[(y * size + x) * 4 + 3] = 255
    }
  }
  return out
}

/**
 * Prépare une image pour le modèle : canaux séparés (CHW), valeurs entre 0 et 1, puis
 * `(x - moyenne) / écart-type` pour chaque canal.
 *
 * L'image doit déjà avoir la taille attendue (le canvas la redimensionne en 224 × 224).
 *
 * @param rgba - Pixels RGBA (4 octets par pixel, ligne après ligne), comme `ImageData.data`.
 * @param width - Largeur de l'image, en pixels.
 * @param height - Hauteur de l'image, en pixels.
 * @returns Un tableau de `3 * height * width` flottants : tous les R, puis tous les V, puis tous les B.
 */
export function preprocess(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
): Float32Array {
  const plane = width * height // nombre de pixels d'un canal
  const out = new Float32Array(3 * plane)
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      // Octet 0-255 -> 0-1 -> centré et réduit avec la moyenne et l'écart-type du canal.
      out[c * plane + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c]
    }
  }
  return out
}

/**
 * Renvoie l'indice de la plus grande valeur (la classe la plus probable).
 *
 * @param values - Probabilités (ou scores) de chaque classe.
 * @returns L'indice du maximum (le premier en cas d'égalité).
 */
export function argmax(values: ArrayLike<number>): number {
  let best = 0
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i
  }
  return best
}

/**
 * Ramène une carte entre 0 et 1 en la divisant par son maximum (comme `_normalize` côté serveur).
 *
 * Une carte dont le maximum n'est pas strictement positif est renvoyée inchangée (la carte
 * du modèle passe par un ReLU : elle ne contient alors que des zéros).
 *
 * @param cam - Carte à valeurs positives ou nulles.
 * @returns Une nouvelle carte, de même taille.
 */
export function normalizeCam(cam: Float32Array): Float32Array {
  let peak = -Infinity
  for (const value of cam) if (value > peak) peak = value
  const out = new Float32Array(cam.length)
  for (let i = 0; i < cam.length; i++) out[i] = peak > 0 ? cam[i] / peak : cam[i]
  return out
}

/**
 * Agrandit une carte carrée (7 × 7) à `size` × `size` par interpolation bilinéaire.
 *
 * Même convention que PyTorch `F.interpolate(mode="bilinear")` utilisé par Captum côté serveur
 * (`align_corners=False` : le centre des pixels est aligné, pas les coins). Le résultat est
 * borné entre 0 et 1, comme `upsample` dans `explain.py`.
 *
 * @param cam - Carte carrée en une dimension (ligne après ligne), côté `sqrt(cam.length)`.
 * @param size - Côté de la carte agrandie, en pixels.
 * @returns La carte agrandie, `size * size` valeurs entre 0 et 1.
 */
export function upsampleBilinear(cam: Float32Array, size: number): Float32Array {
  const side = Math.round(Math.sqrt(cam.length)) // côté de la petite carte (7)
  const scale = side / size
  const out = new Float32Array(size * size)
  // Position source d'un pixel de sortie, comme PyTorch : centre aligné, jamais avant le bord.
  const source = (dst: number): number => Math.max((dst + 0.5) * scale - 0.5, 0)
  for (let y = 0; y < size; y++) {
    const sy = source(y)
    const y0 = Math.min(Math.floor(sy), side - 1)
    const y1 = Math.min(y0 + 1, side - 1)
    const fy = sy - y0
    for (let x = 0; x < size; x++) {
      const sx = source(x)
      const x0 = Math.min(Math.floor(sx), side - 1)
      const x1 = Math.min(x0 + 1, side - 1)
      const fx = sx - x0
      const top = cam[y0 * side + x0] * (1 - fx) + cam[y0 * side + x1] * fx
      const bottom = cam[y1 * side + x0] * (1 - fx) + cam[y1 * side + x1] * fx
      const value = top * (1 - fy) + bottom * fy
      out[y * size + x] = Math.min(Math.max(value, 0), 1) // bornée entre 0 et 1
    }
  }
  return out
}

// ---- Palette « jet » (recopie de `JET_TABLE` / `jet_colors` de predict.py) --------------

/** Nombre de couleurs de la palette. */
const JET_LEVELS = 256

/** Points de contrôle (positions, valeurs) du canal rouge, ceux de matplotlib. */
const JET_RED: [number[], number[]] = [
  [0.0, 0.35, 0.66, 0.89, 1.0],
  [0.0, 0.0, 1.0, 1.0, 0.5],
]
/** Points de contrôle du canal vert. */
const JET_GREEN: [number[], number[]] = [
  [0.0, 0.125, 0.375, 0.64, 0.91, 1.0],
  [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
]
/** Points de contrôle du canal bleu. */
const JET_BLUE: [number[], number[]] = [
  [0.0, 0.11, 0.34, 0.65, 1.0],
  [0.5, 1.0, 1.0, 0.0, 0.0],
]

/**
 * Interpolation linéaire par morceaux, équivalent de `numpy.interp` (valeurs fixes hors bornes).
 *
 * @param x - Position à évaluer.
 * @param xs - Positions des points de contrôle, croissantes.
 * @param ys - Valeurs aux points de contrôle.
 * @returns La valeur interpolée en `x`.
 */
function interp(x: number, xs: number[], ys: number[]): number {
  if (x <= xs[0]) return ys[0]
  const last = xs.length - 1
  if (x >= xs[last]) return ys[last]
  let k = 0
  while (x > xs[k + 1]) k++ // segment [xs[k], xs[k+1]] qui contient x
  return ys[k] + ((ys[k + 1] - ys[k]) * (x - xs[k])) / (xs[k + 1] - xs[k])
}

/** Table des 256 couleurs : 3 valeurs (R, V, B entre 0 et 1) par niveau, calculée une fois. */
const JET_TABLE: Float64Array = (() => {
  const table = new Float64Array(JET_LEVELS * 3)
  for (let i = 0; i < JET_LEVELS; i++) {
    const position = i / (JET_LEVELS - 1) // équivalent de np.linspace(0, 1, 256)
    table[i * 3] = interp(position, ...JET_RED)
    table[i * 3 + 1] = interp(position, ...JET_GREEN)
    table[i * 3 + 2] = interp(position, ...JET_BLUE)
  }
  return table
})()

/**
 * Convertit une valeur entre 0 et 1 en couleur de la palette « jet » (du bleu au rouge).
 *
 * @param value - Valeur entre 0 et 1 (hors de cet intervalle, elle est ramenée au bord).
 * @returns La couleur `[rouge, vert, bleu]`, chaque canal entre 0 et 1.
 */
export function jetColor(value: number): [number, number, number] {
  // Même découpage que le serveur : 256 niveaux, la valeur 1.0 tombe dans le dernier.
  const index = Math.min(Math.max(Math.trunc(value * JET_LEVELS), 0), JET_LEVELS - 1)
  return [JET_TABLE[index * 3], JET_TABLE[index * 3 + 1], JET_TABLE[index * 3 + 2]]
}

/**
 * Superpose la carte de chaleur à l'IRM : `(1 - alpha) * image + alpha * couleur_jet`.
 *
 * @param base - Pixels RGBA de l'IRM (seuls R, V, B sont lus), `heatmap.length * 4` octets.
 * @param heatmap - Carte entre 0 et 1, même nombre de pixels que l'image.
 * @param alpha - Opacité de la carte (0 = invisible, 1 = remplace l'image).
 * @returns Les pixels RGBA de l'image superposée (opaque), prêts pour un `ImageData`.
 */
export function overlay(
  base: Uint8ClampedArray | Uint8Array,
  heatmap: Float32Array,
  alpha: number = OVERLAY_ALPHA,
): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(heatmap.length * 4)
  for (let i = 0; i < heatmap.length; i++) {
    const color = jetColor(heatmap[i])
    for (let c = 0; c < 3; c++) {
      const blended = (1 - alpha) * (base[i * 4 + c] / 255) + alpha * color[c]
      // Comme le serveur : borné entre 0 et 1, puis tronqué (et non arrondi) en octet.
      out[i * 4 + c] = Math.trunc(Math.min(Math.max(blended, 0), 1) * 255)
    }
    out[i * 4 + 3] = 255 // pixel opaque
  }
  return out
}
