/**
 * Zone de dépôt d'une IRM : glisser-déposer ou clic pour choisir un fichier.
 */
import { Upload } from "lucide-react"
import { useRef, useState, type DragEvent } from "react"

import { cn } from "@/lib/utils"

interface DropzoneProps {
  /** Appelé avec le fichier choisi par le visiteur. */
  onFile: (file: File) => void
  /** Désactive la zone pendant l'analyse. */
  disabled?: boolean
}

/** Grande zone cliquable et accessible au clavier, qui accepte une image JPG ou PNG. */
export function Dropzone({ onFile, disabled = false }: DropzoneProps) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  /** Récupère le premier fichier déposé. */
  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files[0]
    if (file && !disabled) onFile(file)
  }

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label="Choisir ou déposer une IRM (JPG ou PNG, 5 Mo maximum)"
      onClick={() => !disabled && input.current?.click()}
      onKeyDown={(event) => {
        if ((event.key === "Enter" || event.key === " ") && !disabled) {
          event.preventDefault()
          input.current?.click()
        }
      }}
      onDragOver={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-10 text-center transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
        dragging ? "border-primary bg-muted" : "border-border hover:bg-muted/50",
        disabled && "pointer-events-none opacity-50",
      )}
    >
      <Upload aria-hidden className="size-8 text-muted-foreground" />
      <p className="font-medium">Glisse une IRM ici, ou clique pour en choisir une</p>
      <p className="text-sm text-muted-foreground">JPG ou PNG, 5 Mo maximum</p>
      <input
        ref={input}
        type="file"
        accept="image/jpeg,image/png"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onFile(file)
          event.target.value = "" // permet de re-choisir le même fichier
        }}
      />
    </div>
  )
}
