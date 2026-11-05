/**
 * Façade d'analyse d'une IRM : une seule fonction, `classify`, deux modes.
 *
 * - `VITE_API_URL` défini : l'image est envoyée à l'API FastAPI (mode serveur, Docker).
 * - Sinon : le modèle ONNX tourne dans le navigateur (mode public) ; l'image ne quitte jamais
 *   l'appareil du visiteur.
 *
 * Dans les deux cas, le résultat est le même type `Prediction`.
 */
import { API_URL, ApiError, MAX_UPLOAD_BYTES, predict, type Prediction } from "@/lib/api"
import { CLASS_ORDER } from "@/lib/classes"
import {
  IMAGE_SIZE,
  argmax,
  normalizeCam,
  overlay,
  preprocess,
  resizeAntialias,
  upsampleBilinear,
} from "@/lib/inference"
import { getSession, ModelError } from "@/lib/model"

/** Avertissement renvoyé avec chaque prédiction (même texte que l'API). */
export const DISCLAIMER = "Démonstration technique, pas un dispositif médical."

/** Vrai quand l'analyse se fait dans le navigateur (aucune API configurée). */
export const BROWSER_MODE = !API_URL

/** Nombre maximal de pixels d'une image (25 millions), comme l'API. */
const MAX_PIXELS = 25_000_000

/** Formats d'image acceptés. */
const ACCEPTED_TYPES = ["image/jpeg", "image/png"]

/**
 * Étape en cours, pour l'affichage : téléchargement du modèle (avec pourcentage, `null` si
 * inconnu) ou analyse de l'image.
 */
export type Status = { phase: "download"; percent: number | null } | { phase: "analysis" }

/**
 * Vérifie la taille et le format du fichier (mêmes limites dans les deux modes).
 *
 * @param file - Fichier choisi par le visiteur.
 * @throws ApiError si le fichier est trop gros ou n'est ni un JPG ni un PNG.
 */
function validate(file: File): void {
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new ApiError("Fichier trop volumineux (5 Mo maximum).")
  }
  if (!ACCEPTED_TYPES.includes(file.type)) {
    throw new ApiError("Format non pris en charge : envoie une image JPG ou PNG.")
  }
}

/**
 * Encode un canvas en PNG puis en base64, sans le préfixe `data:`.
 *
 * @param canvas - Canvas à encoder.
 * @returns Le PNG en base64.
 */
async function canvasToBase64(canvas: HTMLCanvasElement): Promise<string> {
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"))
  if (!blob) throw new ApiError("Impossible de produire l'image de la carte Grad-CAM.")
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(blob)
  })
  return dataUrl.slice(dataUrl.indexOf(",") + 1) // retire « data:image/png;base64, »
}

/**
 * Analyse une IRM entièrement dans le navigateur.
 *
 * Étapes : décodage à la taille native (fond noir), redimensionnement 224 × 224 antialias, prétraitement, modèle ONNX,
 * carte de la classe prédite (normalisée, agrandie), superposition, PNG.
 *
 * @param file - Image JPG ou PNG déjà validée.
 * @param onStatus - Appelée à chaque changement d'étape.
 */
async function classifyInBrowser(file: File, onStatus: (status: Status) => void): Promise<Prediction> {
  // Décodage et dessin dans un canvas 224 x 224, fond noir (les transparences deviennent noires).
  let bitmap: ImageBitmap
  try {
    bitmap = await createImageBitmap(file)
  } catch {
    throw new ApiError("Impossible de lire cette image. Essaie un autre fichier JPG ou PNG.")
  }
  // Plafond de sécurité (comme l'API) : une image géante ferait manquer de mémoire.
  if (bitmap.width * bitmap.height > MAX_PIXELS) {
    bitmap.close()
    throw new ApiError("Image trop grande (25 millions de pixels maximum).")
  }
  // 1) Décodage à la taille NATIVE : c'est notre code, pas le canvas, qui réduit l'image,
  //    pour reproduire exactement le redimensionnement antialias du serveur.
  const native = document.createElement("canvas")
  native.width = bitmap.width
  native.height = bitmap.height
  const nativeContext = native.getContext("2d", { willReadFrequently: true })
  if (!nativeContext) throw new ApiError("Ton navigateur ne permet pas de traiter l'image.")
  nativeContext.fillStyle = "#000" // les transparences deviennent noires
  nativeContext.fillRect(0, 0, bitmap.width, bitmap.height)
  nativeContext.drawImage(bitmap, 0, 0)
  const nativePixels = nativeContext.getImageData(0, 0, bitmap.width, bitmap.height).data
  const pixels = resizeAntialias(nativePixels, bitmap.width, bitmap.height, IMAGE_SIZE)
  bitmap.close()

  // 2) Canvas 224 x 224 qui reçoit l'image redimensionnée, puis la superposition.
  const canvas = document.createElement("canvas")
  canvas.width = canvas.height = IMAGE_SIZE
  const context = canvas.getContext("2d")
  if (!context) throw new ApiError("Ton navigateur ne permet pas de traiter l'image.")

  // Modèle : téléchargé la première fois (avec progression), puis réutilisé.
  // Le statut « téléchargement » n'apparaît que si le fichier est vraiment téléchargé (pas en cache).
  onStatus({ phase: "analysis" })
  const session = await getSession(({ loaded, total }) =>
    onStatus({ phase: "download", percent: total ? Math.min(100, (loaded / total) * 100) : null }),
  )
  onStatus({ phase: "analysis" })

  // Imports dynamiques : onnxruntime-web reste hors du paquet principal.
  const { Tensor } = await import("onnxruntime-web/wasm")
  let probs: Float32Array
  let cams: Float32Array
  try {
    const input = new Tensor("float32", preprocess(pixels, IMAGE_SIZE, IMAGE_SIZE), [
      1,
      3,
      IMAGE_SIZE,
      IMAGE_SIZE,
    ])
    const outputs = await session.run({ image: input })
    probs = outputs.probs.data as Float32Array
    cams = outputs.cams.data as Float32Array
  } catch {
    throw new ModelError("L'analyse a échoué dans ce navigateur. Réessaie ou change de navigateur.")
  }

  // Carte de la classe prédite : 7 x 7 -> normalisée -> agrandie -> superposée à l'IRM.
  const index = argmax(probs)
  const mapSize = cams.length / probs.length // 49 valeurs par classe
  const cam = cams.slice(index * mapSize, (index + 1) * mapSize)
  const heatmap = upsampleBilinear(normalizeCam(cam), IMAGE_SIZE)
  const blended = overlay(pixels, heatmap)
  context.putImageData(new ImageData(blended, IMAGE_SIZE, IMAGE_SIZE), 0, 0)

  return {
    predicted_class: CLASS_ORDER[index],
    probabilities: Object.fromEntries(CLASS_ORDER.map((name, i) => [name, probs[i]])),
    gradcam_png_base64: await canvasToBase64(canvas),
    disclaimer: DISCLAIMER,
  }
}

/**
 * Analyse une IRM : par l'API si `VITE_API_URL` est défini, sinon dans le navigateur.
 *
 * @param file - Image JPG ou PNG choisie par le visiteur.
 * @param onStatus - Appelée à chaque changement d'étape (téléchargement du modèle, analyse).
 * @returns La classe prédite, les probabilités et l'image Grad-CAM (PNG en base64).
 * @throws ApiError ou ModelError avec un message en français.
 */
export async function classify(file: File, onStatus: (status: Status) => void): Promise<Prediction> {
  validate(file)
  if (!BROWSER_MODE) {
    onStatus({ phase: "analysis" })
    return predict(file)
  }
  return classifyInBrowser(file, onStatus)
}
