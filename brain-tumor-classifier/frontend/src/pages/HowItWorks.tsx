/**
 * Page « Comment ça marche » : le trajet d'une IRM, du fichier envoyé à la carte de chaleur.
 */
import { PageTitle } from "@/components/PageTitle"

/** Étapes du fonctionnement, expliquées sans jargon. */
const STEPS = [
  {
    title: "1. L'IRM est envoyée",
    text: "Tu choisis une image (JPG ou PNG). Le site l'envoie à une API, un petit serveur qui héberge le modèle. L'image n'est pas enregistrée.",
  },
  {
    title: "2. L'image est préparée",
    text: "Elle est redimensionnée en 224 × 224 pixels et ses valeurs sont mises à la même échelle que celles vues à l'entraînement.",
  },
  {
    title: "3. Le réseau de neurones l'analyse",
    text: "Un ResNet-50, déjà entraîné à reconnaître des millions de photos, a été adapté à l'IRM cérébrale (transfert d'apprentissage) : d'abord la dernière couche, puis le dernier bloc.",
  },
  {
    title: "4. Il donne une probabilité par catégorie",
    text: "Gliome, méningiome, tumeur de l'hypophyse ou pas de tumeur : quatre pourcentages qui totalisent 100 %. La plus élevée devient la réponse du modèle.",
  },
  {
    title: "5. La carte de chaleur explique",
    text: "La méthode Grad-CAM colore les zones de l'image qui ont le plus pesé dans la décision : rouge et jaune pour les plus importantes.",
  },
]

/** Page pédagogique : étapes numérotées et mise en garde sur la lecture de la carte. */
export function HowItWorks() {
  return (
    <>
      <PageTitle
        title="Comment ça marche"
        intro="Du fichier envoyé à la carte de chaleur, en cinq étapes."
      />
      <ol className="grid gap-4 md:grid-cols-2">
        {STEPS.map(({ title, text }) => (
          <li key={title} className="rounded-xl border p-5">
            <h2 className="font-heading text-lg font-semibold">{title}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{text}</p>
          </li>
        ))}
      </ol>
      <p className="mt-8 rounded-xl bg-muted p-4 text-sm">
        <strong>Lire la carte avec prudence :</strong> elle est grossière (une grille de 7 × 7
        agrandie) et ne montre pas toujours la tumeur. C'est une aide pour comprendre, pas une
        preuve.
      </p>
    </>
  )
}
