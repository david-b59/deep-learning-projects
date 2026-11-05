/**
 * Appels à l'API FastAPI du projet (mode serveur, utilisé avec Docker) et types communs.
 */

/**
 * Adresse de l'API : variable d'environnement VITE_API_URL.
 * Non définie (ou vide), le site n'appelle aucune API : l'analyse se fait dans le navigateur.
 */
export const API_URL: string | undefined = import.meta.env.VITE_API_URL
  ? String(import.meta.env.VITE_API_URL).replace(/\/$/, "")
  : undefined

/** Taille maximale acceptée par l'API (5 Mo), vérifiée aussi côté navigateur. */
export const MAX_UPLOAD_BYTES = 5 * 1024 * 1024

/** Réponse de `POST /predict`. */
export interface Prediction {
  predicted_class: string
  probabilities: Record<string, number>
  gradcam_png_base64: string
  disclaimer: string
}

/** Scores d'une classe sur le jeu de test. */
export interface ClassScores {
  precision: number
  recall: number
  f1: number
  support: number
}

/** Scores du modèle sur le jeu de test (contenu de `metrics.json`, écrit à la compilation). */
export interface MetricsSummary {
  accuracy: number
  class_names: string[]
  per_class: Record<string, ClassScores>
  confusion_matrix: number[][]
}

/** Erreur affichable à l'utilisateur (message en français). */
export class ApiError extends Error {}

/**
 * Envoie une IRM à l'API et renvoie la prédiction.
 *
 * @param file - Image JPG ou PNG choisie par le visiteur.
 * @throws ApiError avec un message clair si l'image est refusée ou l'API injoignable.
 */
export async function predict(file: File): Promise<Prediction> {
  if (!API_URL) throw new ApiError("L'adresse de l'API n'est pas configurée.")
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new ApiError("Fichier trop volumineux (5 Mo maximum).")
  }
  const form = new FormData()
  form.append("file", file)

  let response: Response
  try {
    response = await fetch(`${API_URL}/predict`, { method: "POST", body: form })
  } catch {
    throw new ApiError("Impossible de joindre l'API. Est-elle démarrée ?")
  }
  if (!response.ok) {
    // L'API renvoie {"detail": "message"} pour les erreurs 400, 413 et 500.
    const body = await response.json().catch(() => null)
    const detail = typeof body?.detail === "string" ? body.detail : null
    throw new ApiError(detail ?? "Le serveur a rencontré un problème. Réessaie plus tard.")
  }
  return (await response.json()) as Prediction
}
