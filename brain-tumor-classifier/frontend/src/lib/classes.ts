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

/** Images d'exemple (tirées du jeu de test, donc jamais vues à l'entraînement). */
export const EXAMPLES: { name: ClassName; src: string }[] = [
  { name: "glioma", src: "/examples/glioma.jpg" },
  { name: "meningioma", src: "/examples/meningioma.jpg" },
  { name: "notumor", src: "/examples/notumor.jpg" },
  { name: "pituitary", src: "/examples/pituitary.jpg" },
]

/** Renvoie le nom lisible d'une classe (ou le nom brut si elle est inconnue). */
export function classLabel(name: string): string {
  return name in CLASS_INFO ? CLASS_INFO[name as ClassName].label : name
}
