# Daniel Eduardo Hinojosa Alvarado
# A00838156
import os
import numpy as np
from PIL import Image
import torch
import cv2
from transformers import Sam3Processor, Sam3Model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "dataset_cleaned")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset_segmented")

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
    )
    # Mover a device: tensores float → bfloat16, enteros (text tokens) → sin castear
    inputs = {
        k: v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

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


def save_results(folder: str, img_pil: Image.Image, mask: np.ndarray, base_name: str) -> None:
    out_folder = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(out_folder, exist_ok=True)

    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    img_np = np.array(img_pil)
    rgba = np.dstack([img_np, mask]).astype(np.uint8)
    cropped = rgba[y0:y1, x0:x1]
    Image.fromarray(cropped, "RGBA").save(os.path.join(out_folder, f"{base_name}.png"))

def check_blur(img: Image.Image, threshold: float = 100.0) -> bool:
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold


# ── Procesar todas las imágenes ──────────────────────────────────────────────

folders = [f for f in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR,f))]
print(f"Labels to be segmented {folders}")

for folder in folders:
    print(f"====SEGMENTING {folder} ====")
    extensions = (".png", ".jpg", ".jpeg")
    image_files = sorted(
        f for f in os.listdir(IMAGES_DIR+f'/{folder}') if f.lower().endswith(extensions)
    )

    print(f"Procesando {len(image_files)} imágenes...\n")

    for fname in image_files:
        base_name = os.path.splitext(fname)[0]
        # Usar el nombre del folder (clase) como prompt, no el nombre del archivo
        prompt    = PROMPTS.get(folder, DEFAULT_PROMPT)
        print(f"  [{base_name}] prompt='{prompt}'", end=" ", flush=True)

        img  = Image.open(os.path.join(IMAGES_DIR, folder ,fname)).convert("RGB")
        if check_blur(img):
            print("Image blurred continuing with another")
            continue

        mask = segment_sign(img, prompt)

        if mask is None:
            print("⚠ SAM3 no encontró objetos — prueba con otro prompt")
            continue

        save_results(folder,img, mask, base_name)
        fg_pct = np.count_nonzero(mask) / mask.size * 100
        print(f"foreground={fg_pct:.1f}%  ✓")

print(f"\nResultados guardados en: {OUTPUT_DIR}/")
