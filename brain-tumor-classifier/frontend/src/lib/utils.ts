/**
 * Utilitaires partagés.
 */
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/**
 * Assemble des noms de classes CSS et résout les conflits Tailwind
 * (ex. « p-2 p-4 » devient « p-4 »).
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}
