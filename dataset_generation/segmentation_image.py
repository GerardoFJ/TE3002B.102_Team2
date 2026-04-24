# Daniel Eduardo Hinojosa Alvarado
# A00838156
import os
import numpy as np
from PIL import Image
import torch
import cv2
from transformers import Sam3Processor, Sam3Model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "Test_images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "segmented")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

SAM3_MODEL_ID = "facebook/sam3"
print(f"Cargando SAM3 desde HuggingFace ({SAM3_MODEL_ID})...")
processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
model = Sam3Model.from_pretrained(SAM3_MODEL_ID, torch_dtype=torch.bfloat16)
model.eval().to(device)
print("Modelo listo.\n")

# Prompt por imagen — suficientemente específico para SAM3
PROMPTS = {
    "stop":      "stop sign",
    "direction": "road sign",
    "straight":  "road sign",
    "workers":   "road sign",
    "away":      "road sign",
}
DEFAULT_PROMPT = "traffic sign"


def segment_sign(img_pil: Image.Image, text_prompt: str) -> np.ndarray | None:
    """
    Usa SAM3 con un text prompt para segmentar la señal.
    Retorna la máscara binaria uint8 [H, W] en el tamaño original,
    o None si SAM3 no encontró ningún objeto.
    """
    orig_w, orig_h = img_pil.size

    inputs = processor(
        images=img_pil,
        text=text_prompt,
        return_tensors="pt",
    ).to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=[(orig_h, orig_w)],
    )[0]

    if len(results["masks"]) == 0:
        return None

    # Si hay varias máscaras tomar la de mayor score
    scores = results["scores"].cpu()
    best_idx = int(scores.argmax())
    mask = results["masks"][best_idx].cpu().numpy().astype(np.uint8) * 255

    # Limpieza morfológica ligera
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def save_results(img_pil: Image.Image, mask: np.ndarray, base_name: str) -> None:
    img_np = np.array(img_pil)

    # PNG con canal alpha (fondo transparente)
    rgba = np.dstack([img_np, mask]).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(
        os.path.join(OUTPUT_DIR, f"{base_name}.png")
    )

    # Máscara binaria
    Image.fromarray(mask).save(
        os.path.join(OUTPUT_DIR, f"{base_name}_mask.png")
    )

    # Imagen de comparación: original | máscara overlay
    overlay = img_np.copy()
    overlay[mask == 0] = (overlay[mask == 0] * 0.25).astype(np.uint8)
    comparison = np.hstack([img_np, overlay])
    Image.fromarray(comparison).save(
        os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
    )


# ── Procesar todas las imágenes ──────────────────────────────────────────────
extensions = (".png", ".jpg", ".jpeg")
image_files = sorted(
    f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(extensions)
)

print(f"Procesando {len(image_files)} imágenes...\n")

for fname in image_files:
    base_name = os.path.splitext(fname)[0]
    prompt    = PROMPTS.get(base_name, DEFAULT_PROMPT)
    print(f"  [{base_name}] prompt='{prompt}'", end=" ", flush=True)

    img  = Image.open(os.path.join(IMAGES_DIR, fname)).convert("RGB")
    mask = segment_sign(img, prompt)

    if mask is None:
        print("⚠ SAM3 no encontró objetos — prueba con otro prompt")
        continue

    save_results(img, mask, base_name)
    fg_pct = np.count_nonzero(mask) / mask.size * 100
    print(f"foreground={fg_pct:.1f}%  ✓")

print(f"\nResultados guardados en: {OUTPUT_DIR}/")
print("  <nombre>.png             → imagen con fondo transparente (RGBA)")
print("  <nombre>_mask.png        → máscara binaria")
print("  <nombre>_comparison.png  → comparación visual")
