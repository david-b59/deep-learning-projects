/**
 * Les 4 classes du modèle : nom affiché, explication simple et images d'exemple.
 */

/** Nom technique d'une classe, tel que renvoyé par l'API. */
export type ClassName = "glioma" | "meningioma" | "notumor" | "pituitary"

/** Informations affichables pour une classe. */
export interface ClassInfo {
  label: string
  description: string
}

/** Textes en français pour chaque classe (vocabulaire simple, destiné au grand public). */
export const CLASS_INFO: Record<ClassName, ClassInfo> = {
  glioma: {
    label: "Gliome",
    description: "Tumeur qui se développe dans les cellules de soutien du cerveau.",
  },
  meningioma: {
    label: "Méningiome",
    description: "Tumeur des méninges, les membranes qui enveloppent le cerveau.",
  },
  notumor: {
    label: "Pas de tumeur",
    description: "Le modèle ne voit pas de tumeur sur cette image.",
  },
  pituitary: {
    label: "Tumeur de l'hypophyse",
    description: "Tumeur de l'hypophyse, petite glande à la base du cerveau.",
  },
}

/** Les classes dans l'ordre d'affichage. */
export const CLASS_ORDER: ClassName[] = ["glioma", "meningioma", "notumor", "pituitary"]

/** Une image d'exemple : sa vraie classe et son adresse. */
export interface Example {
  label: ClassName
  src: string
}

/**
 * Charge la liste des images d'exemple (`public/examples/manifest.json`).
 *
 * Cette liste est produite par `python -m brain_mri.examples` : images tirées au hasard dans le
 * jeu de test, sans regarder si le modèle les classe bien.
 */
export async function loadExamples(): Promise<Example[]> {
  const response = await fetch("/examples/manifest.json")
  if (!response.ok) throw new Error("Liste d'exemples introuvable")
  return (await response.json()) as Example[]
}

/** Renvoie le nom lisible d'une classe (ou le nom brut si elle est inconnue). */
export function classLabel(name: string): string {
  return name in CLASS_INFO ? CLASS_INFO[name as ClassName].label : name
}
