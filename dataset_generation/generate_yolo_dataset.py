import os
import random
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
BACKGROUNDS_DIR = SCRIPT_DIR / "backgrounds"
SEGMENTED_DIR = SCRIPT_DIR / "dataset_segmented"
OUTPUT_DIR = SCRIPT_DIR / "dataset_ultra"

# ── Configuración del dataset ────────────────────────────────────────────────
DATASET_SIZE = 10_000           # número total de imágenes a generar
TRAIN_RATIO = 0.8               # 80% train / 20% eval
MIN_OBJECTS_PER_IMAGE = 1
MAX_OBJECTS_PER_IMAGE = 3
MIN_SCALE = 0.08                # tamaño mínimo relativo al lado corto del fondo
MAX_SCALE = 0.30                # tamaño máximo relativo al lado corto del fondo
MAX_PLACEMENT_TRIES = 20        # reintentos para evitar solapamiento
SEED = 42

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_dataset() -> tuple[list[Path], list[str], dict[str, list[Path]]]:
    backgrounds = list_images(BACKGROUNDS_DIR)
    if not backgrounds:
        raise RuntimeError(f"No hay fondos en {BACKGROUNDS_DIR}")

    classes = sorted(p.name for p in SEGMENTED_DIR.iterdir() if p.is_dir())
    if not classes:
        raise RuntimeError(f"No hay clases en {SEGMENTED_DIR}")

    segments = {cls: list_images(SEGMENTED_DIR / cls) for cls in classes}
    for cls, imgs in segments.items():
        if not imgs:
            raise RuntimeError(f"No hay imágenes segmentadas para la clase '{cls}'")

    return backgrounds, classes, segments


def bbox_overlap(a: tuple, b: tuple) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def resize_segment(seg: Image.Image, bg_w: int, bg_h: int) -> Image.Image:
    scale = random.uniform(MIN_SCALE, MAX_SCALE)
    target = int(min(bg_w, bg_h) * scale)
    s_w, s_h = seg.size
    if s_w >= s_h:
        new_w = target
        new_h = max(1, int(s_h * target / s_w))
    else:
        new_h = target
        new_w = max(1, int(s_w * target / s_h))
    new_w = max(1, min(new_w, bg_w))
    new_h = max(1, min(new_h, bg_h))
    return seg.resize((new_w, new_h), Image.LANCZOS)


def try_place(seg: Image.Image, bg_w: int, bg_h: int, existing: list[tuple]) -> tuple | None:
    s_w, s_h = seg.size
    if s_w > bg_w or s_h > bg_h:
        return None
    for _ in range(MAX_PLACEMENT_TRIES):
        x = random.randint(0, bg_w - s_w)
        y = random.randint(0, bg_h - s_h)
        box = (x, y, x + s_w, y + s_h)
        if not any(bbox_overlap(box, b) for b in existing):
            return box
    return None


def tight_bbox_from_alpha(seg: Image.Image, offset_x: int, offset_y: int) -> tuple | None:
    alpha = seg.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        return None
    ax0, ay0, ax1, ay1 = bbox
    return (offset_x + ax0, offset_y + ay0, offset_x + ax1, offset_y + ay1)


def yolo_line(class_id: int, bbox: tuple, img_w: int, img_h: int) -> str:
    x0, y0, x1, y1 = bbox
    cx = ((x0 + x1) / 2) / img_w
    cy = ((y0 + y1) / 2) / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def pick_balanced_class(counts: dict[str, int], classes: list[str]) -> str:
    min_count = min(counts[c] for c in classes)
    candidates = [c for c in classes if counts[c] == min_count]
    return random.choice(candidates)


def setup_output_dirs() -> None:
    for split in ("train", "eval"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_metadata(classes: list[str]) -> None:
    with open(OUTPUT_DIR / "classes.txt", "w") as f:
        for c in classes:
            f.write(c + "\n")

    data_yaml = (
        f"path: {OUTPUT_DIR}\n"
        f"train: images/train\n"
        f"val: images/eval\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(data_yaml)


def generate() -> None:
    random.seed(SEED)
    backgrounds, classes, segments = load_dataset()
    class_to_id = {c: i for i, c in enumerate(classes)}
    counts = {c: 0 for c in classes}

    setup_output_dirs()

    n_train = int(DATASET_SIZE * TRAIN_RATIO)
    print(f"Clases detectadas: {classes}")
    print(f"Generando {DATASET_SIZE} imágenes ({n_train} train / {DATASET_SIZE - n_train} eval)\n")

    generated = 0
    idx = 0
    while generated < DATASET_SIZE:
        split = "train" if generated < n_train else "eval"

        bg_path = random.choice(backgrounds)
        bg = Image.open(bg_path).convert("RGB").copy()
        bg_w, bg_h = bg.size

        n_objects = random.randint(MIN_OBJECTS_PER_IMAGE, MAX_OBJECTS_PER_IMAGE)
        placed_boxes: list[tuple] = []
        annotations: list[str] = []

        for _ in range(n_objects):
            cls = pick_balanced_class(counts, classes)
            seg_path = random.choice(segments[cls])
            seg = Image.open(seg_path).convert("RGBA")
            seg = resize_segment(seg, bg_w, bg_h)

            placement = try_place(seg, bg_w, bg_h, placed_boxes)
            if placement is None:
                continue

            bg.paste(seg, (placement[0], placement[1]), seg)
            tight = tight_bbox_from_alpha(seg, placement[0], placement[1])
            if tight is None:
                continue

            placed_boxes.append(tight)
            annotations.append(yolo_line(class_to_id[cls], tight, bg_w, bg_h))
            counts[cls] += 1

        idx += 1
        if not annotations:
            continue

        stem = f"img_{generated:06d}"
        bg.save(OUTPUT_DIR / "images" / split / f"{stem}.jpg", quality=90)
        with open(OUTPUT_DIR / "labels" / split / f"{stem}.txt", "w") as f:
            f.write("\n".join(annotations) + "\n")

        generated += 1
        if generated % 200 == 0:
            print(f"  [{generated}/{DATASET_SIZE}] counts={counts}")

    write_metadata(classes)
    print(f"\nDataset listo en: {OUTPUT_DIR}")
    print(f"Conteo final por clase: {counts}")


if __name__ == "__main__":
    generate()
