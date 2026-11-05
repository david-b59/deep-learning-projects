/**
 * Chargement du modèle ONNX dans le navigateur.
 *
 * - Téléchargement avec progression (lecture en flux de la réponse).
 * - Cache du fichier dans le navigateur (Cache API) : les visites suivantes ne le retéléchargent pas.
 * - Création de la session onnxruntime-web une seule fois, mémorisée pour les analyses suivantes.
 *
 * onnxruntime-web est importé dynamiquement : il n'alourdit pas le paquet principal du site,
 * il n'est chargé qu'au premier clic sur « Analyser ».
 */
import { ApiError } from "@/lib/api"
import type { InferenceSession } from "onnxruntime-web/wasm"

/** Adresse du modèle : variable VITE_MODEL_URL, sinon le dépôt public Hugging Face du projet. */
export const MODEL_URL: string =
  import.meta.env.VITE_MODEL_URL || "https://huggingface.co/david-b59/brain-tumor-classifier/resolve/main/model.onnx"

/** Nom du cache du navigateur (à changer si le modèle publié change). */
export const CACHE_NAME = "brain-tumor-model-v1"

/** Erreur de chargement du modèle (message en français) : traitée comme les autres `ApiError`. */
export class ModelError extends ApiError {}

/**
 * Progression du téléchargement : octets reçus et total (`null` si le serveur ne l'annonce pas).
 */
export interface DownloadProgress {
  loaded: number
  total: number | null
}

/** Session ONNX déjà demandée (une seule création, même si plusieurs analyses se suivent). */
let sessionPromise: Promise<InferenceSession> | null = null

/**
 * Lit une réponse HTTP en flux et renvoie son contenu, en signalant la progression.
 *
 * @param response - Réponse déjà obtenue (statut vérifié par l'appelant).
 * @param onProgress - Appelée à chaque morceau reçu.
 * @returns Le contenu complet du fichier.
 */
async function readWithProgress(
  response: Response,
  onProgress: (progress: DownloadProgress) => void,
): Promise<Uint8Array> {
  const header = Number(response.headers.get("Content-Length"))
  const total = Number.isFinite(header) && header > 0 ? header : null // sinon : indéterminé
  if (!response.body) {
    // Navigateur sans lecture en flux : on lit d'un bloc, sans progression intermédiaire.
    const bytes = new Uint8Array(await response.arrayBuffer())
    onProgress({ loaded: bytes.length, total: bytes.length })
    return bytes
  }
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    onProgress({ loaded, total })
  }
  // Assemble les morceaux en un seul tableau.
  const bytes = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.length
  }
  return bytes
}

/**
 * Renvoie le fichier du modèle : depuis le cache du navigateur s'il y est, sinon téléchargé
 * (avec progression) puis mis en cache.
 *
 * @param onProgress - Appelée pendant le téléchargement seulement (jamais si le cache répond).
 * @throws ModelError si le réseau échoue ou si le serveur refuse la requête.
 */
export async function fetchModel(
  onProgress: (progress: DownloadProgress) => void,
): Promise<Uint8Array> {
  // La Cache API n'existe que dans un contexte sécurisé (https ou localhost) : sinon, pas de cache.
  const cache = typeof caches !== "undefined" ? await caches.open(CACHE_NAME).catch(() => null) : null

  const cached = await cache?.match(MODEL_URL)
  if (cached) return new Uint8Array(await cached.arrayBuffer())

  let response: Response
  try {
    response = await fetch(MODEL_URL)
  } catch {
    throw new ModelError(
      "Impossible de télécharger le modèle. Vérifie ta connexion Internet puis réessaie.",
    )
  }
  if (!response.ok) {
    throw new ModelError(`Le modèle est introuvable (erreur ${response.status}). Réessaie plus tard.`)
  }
  let bytes: Uint8Array
  try {
    bytes = await readWithProgress(response, onProgress)
  } catch {
    throw new ModelError("Le téléchargement du modèle a été interrompu. Réessaie.")
  }
  // Mise en cache : un échec (espace disque plein, navigation privée) ne doit pas bloquer l'analyse.
  await cache
    ?.put(MODEL_URL, new Response(bytes as BlobPart, { headers: { "Content-Type": "application/octet-stream" } }))
    .catch(() => undefined)
  return bytes
}

/**
 * Crée la session onnxruntime-web (moteur WebAssembly, processeur) à partir des octets du modèle.
 *
 * @param bytes - Contenu du fichier `.onnx`.
 * @throws ModelError si le navigateur est incompatible ou manque de mémoire.
 */
async function createSession(bytes: Uint8Array): Promise<InferenceSession> {
  if (typeof WebAssembly === "undefined") {
    throw new ModelError("Ton navigateur ne gère pas WebAssembly : utilise une version récente de Chrome, Firefox, Edge ou Safari.")
  }
  try {
    // Import dynamique : le moteur n'est chargé qu'ici. La variante « wasm » est le moteur
    // processeur seul (fichier .wasm de 15 Mo), bien plus léger que la variante WebGPU.
    const ort = await import("onnxruntime-web/wasm")
    // Adresse du fichier .wasm : Vite le copie dans le site avec un nom à empreinte.
    const { default: wasmUrl } = await import("onnxruntime-web/ort-wasm-simd-threaded.wasm?url")
    ort.env.wasm.wasmPaths = { wasm: wasmUrl }
    // Un seul fil : évite d'exiger les en-têtes spéciaux (isolation cross-origin) du multi-fil.
    ort.env.wasm.numThreads = 1
    return await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] })
  } catch (error) {
    if (error instanceof ModelError) throw error
    const text = String(error).toLowerCase()
    if (error instanceof RangeError || text.includes("memory") || text.includes("alloc")) {
      throw new ModelError(
        "Ton appareil n'a pas assez de mémoire pour charger le modèle. Ferme d'autres onglets puis réessaie.",
      )
    }
    throw new ModelError(
      "Impossible de démarrer le modèle dans ce navigateur. Essaie avec une version récente de Chrome, Firefox, Edge ou Safari.",
    )
  }
}

/**
 * Renvoie la session du modèle : la crée au premier appel (téléchargement compris), puis la réutilise.
 *
 * @param onProgress - Appelée pendant le téléchargement du fichier du modèle.
 * @throws ModelError avec un message en français ; après une erreur, l'appel suivant réessaie.
 */
export function getSession(
  onProgress: (progress: DownloadProgress) => void,
): Promise<InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = fetchModel(onProgress).then(createSession)
    // En cas d'échec, on oublie la promesse pour permettre un nouvel essai.
    sessionPromise.catch(() => {
      sessionPromise = null
    })
  }
  return sessionPromise
}
