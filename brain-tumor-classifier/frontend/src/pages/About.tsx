/**
 * Page « À propos » : contexte du projet, données, technologies, code et licences.
 */
import { PageTitle } from "@/components/PageTitle"

const GITHUB_URL = "https://github.com/david-b59/deep-learning-projects"
const HUGGING_FACE_URL = "https://huggingface.co/david-b59/brain-tumor-classifier"
const DATASET_URL = "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"

/** Technologies utilisées, par couche du projet. */
const STACK = [
  { part: "Modèle", tools: "PyTorch, ResNet-50 (transfert d'apprentissage), Grad-CAM" },
  { part: "API", tools: "FastAPI, Pydantic, Uvicorn" },
  { part: "Site", tools: "React, TypeScript, Vite, Tailwind CSS, shadcn/ui" },
  { part: "Publication", tools: "Hugging Face Hub (modèle au format safetensors)" },
]

/** Lien externe qui s'ouvre dans un nouvel onglet, de façon sûre. */
function ExternalLink({ href, children }: { href: string; children: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="underline underline-offset-4 hover:text-foreground"
    >
      {children}
    </a>
  )
}

/** Page de présentation du projet pour un public technique et non technique. */
export function About() {
  return (
    <>
      <PageTitle
        title="À propos"
        intro="Le projet, les données et ce qu'il faut savoir avant de s'y fier."
      />
      <div className="max-w-3xl space-y-8">
        <section className="space-y-2">
          <h2 className="font-heading text-xl font-semibold">Le projet</h2>
          <p className="text-muted-foreground">
            Projet réalisé dans le cadre de mon Master Machine Learning Engineering, puis
            prolongé à titre personnel : un modèle qui classe des IRM cérébrales en quatre
            catégories, une API, et ce site. Tout le code, les tests et la documentation sont
            publics sur <ExternalLink href={GITHUB_URL}>GitHub</ExternalLink>. Le modèle est
            publié sur <ExternalLink href={HUGGING_FACE_URL}>Hugging Face</ExternalLink> avec sa
            fiche.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="font-heading text-xl font-semibold">Les données</h2>
          <p className="text-muted-foreground">
            <ExternalLink href={DATASET_URL}>Brain Tumor MRI Dataset</ExternalLink> de Masoud
            Nickparvar (Kaggle), licence CC BY 4.0 : 7 200 images en quatre classes. Les 187
            copies identiques ont été retirées avant l'entraînement. Le jeu de test n'a été
            utilisé qu'une seule fois, à la fin.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="font-heading text-xl font-semibold">Les technologies</h2>
          <dl className="divide-y rounded-xl border">
            {STACK.map(({ part, tools }) => (
              <div key={part} className="grid gap-1 p-4 sm:grid-cols-[8rem_1fr]">
                <dt className="font-medium">{part}</dt>
                <dd className="text-sm text-muted-foreground">{tools}</dd>
              </div>
            ))}
          </dl>
        </section>

        <section className="space-y-2 rounded-xl bg-muted p-5">
          <h2 className="font-heading text-xl font-semibold">Avant de s'y fier</h2>
          <p className="text-sm">
            Démonstration technique, <strong>pas un dispositif médical</strong>. Aucune
            validation clinique. Ne jamais l'utiliser pour un diagnostic ni pour une décision de
            soin : seul un professionnel de santé le peut.
          </p>
        </section>

        <p className="text-sm text-muted-foreground">
          Code et poids du modèle : licence MIT.
        </p>
      </div>
    </>
  )
}
