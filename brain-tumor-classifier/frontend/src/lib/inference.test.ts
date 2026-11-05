/**
 * Tests Vitest des fonctions pures de `inference.ts`.
 *
 * Les valeurs de référence de la palette « jet » et de l'agrandissement bilinéaire viennent du
 * code Python du serveur (`jet_colors` de predict.py et `upsample` de explain.py).
 */
import { describe, expect, it } from "vitest"

import {
  argmax,
  jetColor,
  normalizeCam,
  overlay,
  preprocess,
  resizeAntialias,
  upsampleBilinear,
} from "./inference"

describe("preprocess", () => {
  it("normalise un pixel connu et range les canaux à plat (CHW)", () => {
    // Une image 1 x 2 : pixel 0 = (255, 0, 0), pixel 1 = (0, 255, 128), alpha ignoré.
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 128, 255])
    const out = preprocess(rgba, 2, 1)
    expect(out).toHaveLength(6) // 3 canaux x 2 pixels
    // Calcul à la main : (1 - 0,485) / 0,229 = 2,2489 ; (0 - 0,456) / 0,224 = -2,0357.
    expect(out[0]).toBeCloseTo(2.2489, 3) // canal R, pixel 0
    expect(out[1]).toBeCloseTo(-0.485 / 0.229, 4) // canal R, pixel 1
    expect(out[2]).toBeCloseTo(-2.0357, 3) // canal V, pixel 0
    expect(out[3]).toBeCloseTo((1 - 0.456) / 0.224, 4) // canal V, pixel 1
    expect(out[4]).toBeCloseTo(-0.406 / 0.225, 4) // canal B, pixel 0
    expect(out[5]).toBeCloseTo((128 / 255 - 0.406) / 0.225, 4) // canal B, pixel 1
  })
})

describe("argmax", () => {
  it("renvoie l'indice du maximum", () => {
    expect(argmax([0.1, 0.7, 0.15, 0.05])).toBe(1)
  })

  it("renvoie le premier indice en cas d'égalité", () => {
    expect(argmax([0.5, 0.5, 0])).toBe(0)
  })
})

describe("normalizeCam", () => {
  it("divise par le maximum : le plus grand devient 1", () => {
    const out = normalizeCam(new Float32Array([0, 2, 4, 1]))
    expect(Array.from(out)).toEqual([0, 0.5, 1, 0.25])
  })

  it("renvoie des zéros pour une carte nulle", () => {
    const out = normalizeCam(new Float32Array(49))
    expect(out.every((value) => value === 0)).toBe(true)
  })
})

describe("upsampleBilinear", () => {
  it("donne une carte de la bonne taille", () => {
    expect(upsampleBilinear(new Float32Array(49), 224)).toHaveLength(224 * 224)
  })

  it("garde une carte constante constante", () => {
    const out = upsampleBilinear(new Float32Array(49).fill(0.3), 20)
    for (const value of out) expect(value).toBeCloseTo(0.3, 6)
  })

  it("garde les valeurs des coins (convention align_corners=False de PyTorch)", () => {
    const cam = new Float32Array(49)
    cam[0] = 1 // coin haut-gauche
    cam[48] = 0.5 // coin bas-droite
    const out = upsampleBilinear(cam, 224)
    expect(out[0]).toBeCloseTo(1, 6)
    expect(out[224 * 224 - 1]).toBeCloseTo(0.5, 6)
  })

  it("reproduit PyTorch sur une rampe 7x7 agrandie en 14x14", () => {
    // Valeurs de référence obtenues avec `upsample` de explain.py (rampe i / 48).
    const ramp = Float32Array.from({ length: 49 }, (_, i) => i / 48)
    const out = upsampleBilinear(ramp, 14)
    const expected = [0.0, 0.005208, 0.015625, 0.026042, 0.036458, 0.046875]
    expected.forEach((value, x) => expect(out[x]).toBeCloseTo(value, 5))
    expect(out[5 * 14 + 3]).toBeCloseTo(0.354167, 5)
    expect(out[7 * 14 + 9]).toBeCloseTo(0.5625, 5)
    expect(out[13 * 14 + 13]).toBeCloseTo(1, 5)
  })
})

describe("jetColor", () => {
  // Valeurs de référence calculées avec `jet_colors` de predict.py.
  const references: [number, [number, number, number]][] = [
    [0.0, [0.0, 0.0, 0.5]],
    [0.1, [0.0, 0.0, 0.9456327985739751]],
    [0.25, [0.0, 0.503921568627451, 1.0]],
    [0.5, [0.4901960784313725, 1.0, 0.4775458570524985]],
    [0.75, [1.0, 0.5816993464052289, 0.0]],
    [0.9, [0.945632798573975, 0.02977487291212788, 0.0]],
    [1.0, [0.5, 0.0, 0.0]],
  ]

  it.each(references)("renvoie la couleur de référence pour %f", (value, expected) => {
    const color = jetColor(value)
    expected.forEach((channel, c) => expect(color[c]).toBeCloseTo(channel, 9))
  })

  it("ramène les valeurs hors de 0-1 au bord de la palette", () => {
    expect(jetColor(-3)).toEqual(jetColor(0))
    expect(jetColor(7)).toEqual(jetColor(1))
  })
})

describe("overlay", () => {
  /** Une image de 2 pixels gris (100 et 200) et une carte (0 et 1). */
  const base = new Uint8ClampedArray([100, 100, 100, 255, 200, 200, 200, 255])
  const heatmap = new Float32Array([0, 1])

  it("avec alpha = 0, rend l'image d'origine (opaque)", () => {
    const out = overlay(base, heatmap, 0)
    // La troncature de (100 / 255) * 255 peut perdre 1 niveau : on tolère cet écart.
    for (let i = 0; i < 8; i++) expect(Math.abs(out[i] - base[i])).toBeLessThanOrEqual(1)
    expect(out[3]).toBe(255)
    expect(out[7]).toBe(255)
  })

  it("avec alpha = 1, rend uniquement les couleurs de la palette", () => {
    const out = overlay(base, heatmap, 1)
    // Valeur 0 -> (0, 0, 0,5) ; valeur 1 -> (0,5, 0, 0) ; octets tronqués comme côté serveur.
    expect(Array.from(out.slice(0, 4))).toEqual([0, 0, 127, 255])
    expect(Array.from(out.slice(4, 8))).toEqual([127, 0, 0, 255])
  })

  it("mélange avec alpha = 0,45 par défaut", () => {
    const out = overlay(base, heatmap)
    // Pixel 0 : bleu = 0,55 * 100/255 + 0,45 * 0,5 = 0,44069 -> 112 ; rouge = 0,55 * 100/255 = 0,2157 -> 55.
    expect(out[0]).toBe(55)
    expect(out[2]).toBe(112)
  })
})

describe("resizeAntialias", () => {
  /**
   * Image de test déterministe, identique à celle du script Python de référence :
   * valeur = (x*53 + y*97 + canal*29 + x*y*7) % 256, alpha = 255.
   */
  function testImage(width: number, height: number): Uint8ClampedArray {
    const data = new Uint8ClampedArray(width * height * 4)
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        for (let c = 0; c < 3; c++) data[(y * width + x) * 4 + c] = (x * 53 + y * 97 + c * 29 + x * y * 7) % 256
        data[(y * width + x) * 4 + 3] = 255
      }
    }
    return data
  }

  // Références : torchvision v2.Resize((224, 224), antialias=True) sur un tenseur uint8.
  // Chaque entrée : [ligne, colonne, [R, V, B]] pour l'image 53 x 37 (largeur x hauteur).
  const reductions: [number, number, number[]][] = [
    [0, 0, [0, 29, 58]],
    [0, 223, [196, 225, 254]],
    [223, 0, [164, 193, 222]],
    [223, 223, [152, 181, 210]],
    [112, 112, [23, 52, 79]],
    [1, 2, [5, 34, 63]],
    [50, 170, [54, 83, 112]],
    [100, 30, [162, 125, 154]],
    [150, 200, [142, 171, 190]],
    [200, 100, [11, 40, 69]],
    [17, 89, [121, 67, 96]],
    [130, 45, [177, 206, 235]],
    [75, 75, [123, 152, 176]],
    [222, 3, [180, 209, 154]],
  ]

  it("reproduit torchvision (antialias) à 1 niveau près sur une image impaire 53 x 37", () => {
    const out = resizeAntialias(testImage(53, 37), 53, 37)
    expect(out).toHaveLength(224 * 224 * 4)
    for (const [y, x, rgb] of reductions) {
      rgb.forEach((expected, c) => {
        expect(Math.abs(out[(y * 224 + x) * 4 + c] - expected)).toBeLessThanOrEqual(1)
      })
      expect(out[(y * 224 + x) * 4 + 3]).toBe(255)
    }
  })

  it("agrandit une image 10 x 10 comme torchvision (à 1 niveau près)", () => {
    // Références pour l'image 10 x 10 : [ligne, colonne, [R, V, B]].
    const references: [number, number, number[]][] = [
      [0, 0, [0, 29, 58]],
      [0, 223, [221, 250, 23]],
      [223, 0, [105, 134, 163]],
      [223, 223, [125, 154, 183]],
      [112, 112, [112, 141, 111]],
      [1, 2, [0, 29, 58]],
    ]
    const out = resizeAntialias(testImage(10, 10), 10, 10)
    for (const [y, x, rgb] of references) {
      rgb.forEach((expected, c) => {
        expect(Math.abs(out[(y * 224 + x) * 4 + c] - expected)).toBeLessThanOrEqual(1)
      })
    }
  })

  it("laisse une image déjà 224 x 224 identique", () => {
    const image = testImage(224, 224)
    expect(Array.from(resizeAntialias(image, 224, 224))).toEqual(Array.from(image))
  })
})
