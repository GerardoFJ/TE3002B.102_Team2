"""
TE3002B - Reto Semana 2: Clasificación de Flechas con Regresión Logística
Equipo 2

Dataset:
  L (*.png)   → Izquierda (y=0) — imágenes digitales limpias
  s3 (*.jpg)  → Izquierda (y=0) — fotos reales
  s11 (*.jpg) → Izquierda (y=0) — fotos reales
  s1 (*.jpg)  → Derecha   (y=1) — fotos reales
  s2 (*.jpg)  → Derecha   (y=1) — fotos reales
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # backend sin display (guarda en archivo)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
warnings.filterwarnings("ignore")


# ─── Implementación manual de Regresión Logística con Gradient Descent ────────
class LogisticRegressionGD:
    """
    Regresión Logística entrenada con Gradient Descent.

    Implementa exactamente la regla de la slide:
        β ← β - η · ∇J(β)

    donde el gradiente de la binary cross-entropy es:
        ∇J(β) = (1/m) · Xᵀ · (σ(Xβ) - y)

    y la función sigmoide es:
        σ(z) = 1 / (1 + e^{-z})

    Incluye regularización L2 (ridge) para evitar overfitting:
        ∇J(β) += (λ/m) · β    (no se regulariza el bias b)
    """

    def __init__(self, learning_rate=0.1, n_iter=1000, lambda_reg=0.01):
        self.lr         = learning_rate
        self.n_iter     = n_iter
        self.lambda_reg = lambda_reg
        self.beta       = None   # pesos (sin bias)
        self.b          = None   # bias (intercepto)
        self.history    = []     # historial del costo por época

    @staticmethod
    def _sigmoid(z):
        # Clipping para estabilidad numérica (evita overflow en exp)
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _cost(self, X, y):
        """Binary cross-entropy con regularización L2."""
        m   = len(y)
        p   = self._sigmoid(X @ self.beta + self.b)
        p   = np.clip(p, 1e-9, 1 - 1e-9)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = (self.lambda_reg / (2 * m)) * np.sum(self.beta ** 2)
        return bce + reg

    def fit(self, X, y):
        m, n = X.shape
        self.beta    = np.zeros(n)
        self.b       = 0.0
        self.history = []

        for _ in range(self.n_iter):
            # Predicción: σ(Xβ + b)
            p = self._sigmoid(X @ self.beta + self.b)

            # Gradientes: ∇J(β) = (1/m) Xᵀ(p - y) + (λ/m)β
            error      = p - y
            grad_beta  = (X.T @ error) / m + (self.lambda_reg / m) * self.beta
            grad_b     = error.mean()

            # Actualización: β ← β - η·∇J(β)
            self.beta -= self.lr * grad_beta
            self.b    -= self.lr * grad_b

            self.history.append(self._cost(X, y))

        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.beta + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# ─── Configuración ────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "Data_Set", "Arrow")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "Results", "model_memory.pkl")
IMG_SIZE     = 64        # píxeles (NxN)
TEST_SIZE    = 0.2
RANDOM_STATE = 42
FORCE_RETRAIN = False    # True: reentrenar aunque exista modelo guardado

# Mapeo prefijo → etiqueta
LABEL_MAP = {
    "L":   0,   # Izquierda
    "s3":  0,   # Izquierda
    "s11": 0,   # Izquierda
    "s1":  1,   # Derecha
    "s2":  1,   # Derecha
}

# ─── 1. Carga del dataset ─────────────────────────────────────────────────────
def get_label(filename):
    """Determina la etiqueta a partir del prefijo del nombre de archivo."""
    # Probar prefijos de mayor a menor longitud para evitar ambigüedad (s11 vs s1)
    for prefix in sorted(LABEL_MAP.keys(), key=len, reverse=True):
        if filename.startswith(prefix):
            return LABEL_MAP[prefix]
    return None


def load_images(dataset_path, img_size=IMG_SIZE):
    """Carga y preprocesa todas las imágenes del dataset."""
    images, labels, filenames = [], [], []

    for fname in sorted(os.listdir(dataset_path)):
        if not (fname.endswith(".png") or fname.endswith(".jpg")):
            continue

        label = get_label(fname)
        if label is None:
            print(f"  [SKIP] Sin etiqueta para: {fname}")
            continue

        path = os.path.join(dataset_path, fname)
        img  = cv2.imread(path)
        if img is None:
            print(f"  [ERROR] No se pudo leer: {fname}")
            continue

        # Preprocesamiento
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Recortar barras negras del top/bottom (artefacto de cámara)
        row_means = gray.mean(axis=1)
        valid_rows = np.where(row_means > 10)[0]
        if len(valid_rows) > 10:
            gray = gray[valid_rows[0]:valid_rows[-1]+1, :]

        # Binarización: Otsu encuentra automáticamente el umbral óptimo
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Operaciones morfológicas para limpiar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Crop al bounding box del contorno más grande (= la flecha)
        # Esto normaliza la posición independientemente de dónde esté en el frame
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y_c, w, h = cv2.boundingRect(largest)
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y_c - pad)
            x2 = min(binary.shape[1], x + w + pad)
            y2 = min(binary.shape[0], y_c + h + pad)
            binary = binary[y1:y2, x1:x2]

        resized = cv2.resize(binary, (img_size, img_size))
        norm = resized.astype(np.float32) / 255.0

        images.append(norm)
        labels.append(label)
        filenames.append(fname)

    return np.array(images), np.array(labels), filenames


print("=" * 60)
print("  TE3002B — Clasificador de Flechas (Regresión Logística)")
print("=" * 60)
print(f"\n[1] Cargando dataset desde: {DATASET_PATH}")
X_imgs, y, filenames = load_images(DATASET_PATH)
print(f"    Total imágenes: {len(y)}")
print(f"    Izquierda (y=0): {np.sum(y == 0)}")
print(f"    Derecha   (y=1): {np.sum(y == 1)}")


# ─── 2. Extracción de características ────────────────────────────────────────
def extract_features(images):
    """
    Extrae un vector de características para cada imagen combinando:

    a) Proyección de columnas (64 valores):
       Suma vertical de píxeles oscuros por columna.
       Una flecha izquierda concentra masa en columnas izquierdas;
       una flecha derecha, en columnas derechas.

    b) Proyección de filas (64 valores):
       Suma horizontal de píxeles oscuros por fila.

    c) Asimetría lateral (1 valor escalar):
       (suma mitad derecha − suma mitad izquierda) / total
       Valor positivo → flecha derecha; negativo → flecha izquierda.

    d) Asimetría vertical (1 valor escalar):
       (suma mitad inferior − superior) / total

    Total: 64 + 64 + 1 + 1 = 130 características.
    """
    feats = []
    for img in images:
        dark = 1.0 - img            # invertir: píxeles oscuros = alto valor

        col_proj = dark.sum(axis=0)          # (64,)
        row_proj = dark.sum(axis=1)          # (64,)

        total     = dark.sum() + 1e-6
        h         = img.shape[1]
        left_sum  = dark[:, :h//2].sum()
        right_sum = dark[:, h//2:].sum()
        asym_lr   = (right_sum - left_sum) / total

        v         = img.shape[0]
        top_sum   = dark[:v//2, :].sum()
        bot_sum   = dark[v//2:, :].sum()
        asym_tb   = (bot_sum - top_sum) / total

        feat = np.concatenate([col_proj, row_proj, [asym_lr], [asym_tb]])
        feats.append(feat)

    return np.array(feats)


# ─── 1b. Data Augmentation ───────────────────────────────────────────────────
def augment(images, labels, filenames):
    """
    Genera muestras adicionales volteando cada imagen horizontalmente.

    Una flecha IZQUIERDA volteada = DERECHA (y viceversa).
    Esto duplica el dataset y balancea perfectamente las clases.
    También agrega versiones con pequeña rotación (±8°) para robustez.
    """
    aug_imgs, aug_labels, aug_fns = list(images), list(labels), list(filenames)
    h, w = images[0].shape

    for img, label, fn in zip(images, labels, filenames):
        # Volteo horizontal: invierte la dirección de la flecha
        flipped = np.fliplr(img)
        aug_imgs.append(flipped)
        aug_labels.append(1 - label)       # 0→1, 1→0
        aug_fns.append(fn + "_flip")

        # Rotación leve para robustez (±8°) — mantiene la misma etiqueta
        for angle in [-8, 8]:
            M   = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rot = cv2.warpAffine(img, M, (w, h))
            aug_imgs.append(rot)
            aug_labels.append(label)
            aug_fns.append(fn + f"_rot{angle}")

    return np.array(aug_imgs), np.array(aug_labels), aug_fns


# ─── 2. Extracción de características ────────────────────────────────────────
def hog_features(img, cell_size=8, n_bins=9):
    """
    HOG (Histogram of Oriented Gradients) simplificado.

    Divide la imagen en celdas de cell_size×cell_size píxeles.
    En cada celda calcula un histograma de orientaciones del gradiente.
    Mucho más robusto a posición e iluminación que proyecciones crudas.
    """
    # Gradientes en x e y
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle     = np.arctan2(gy, gx) * (180 / np.pi) % 180  # 0–180°

    h, w   = img.shape
    n_cx   = w // cell_size
    n_cy   = h // cell_size
    hist   = []

    for cy in range(n_cy):
        for cx in range(n_cx):
            r0, r1 = cy * cell_size, (cy + 1) * cell_size
            c0, c1 = cx * cell_size, (cx + 1) * cell_size
            mag_cell = magnitude[r0:r1, c0:c1]
            ang_cell = angle[r0:r1, c0:c1]
            cell_hist, _ = np.histogram(ang_cell, bins=n_bins,
                                        range=(0, 180), weights=mag_cell)
            hist.append(cell_hist)

    return np.concatenate(hist)


def extract_features(images):
    """
    Extrae un vector de características por imagen combinando:

    a) Proyección de columnas (IMG_SIZE valores) — masa oscura por columna
    b) Proyección de filas    (IMG_SIZE valores) — masa oscura por fila
    c) Asimetría lateral (1 escalar): (mitad der − izq) / total
    d) Asimetría vertical (1 escalar): (mitad inf − sup) / total
    e) HOG features: gradientes locales en celdas 8×8, 9 bins
       → (IMG_SIZE/8)² × 9 = 64 × 9 / 8² = 8×8×9 = 576 valores

    Total: 64 + 64 + 1 + 1 + 576 = 706 características.
    """
    feats = []
    for img in images:
        dark = 1.0 - img

        col_proj = dark.sum(axis=0)
        row_proj = dark.sum(axis=1)

        total    = dark.sum() + 1e-6
        h        = img.shape[1]
        asym_lr  = (dark[:, h//2:].sum() - dark[:, :h//2].sum()) / total
        v        = img.shape[0]
        asym_tb  = (dark[v//2:, :].sum() - dark[:v//2, :].sum()) / total

        hog = hog_features(img)

        feat = np.concatenate([col_proj, row_proj, [asym_lr], [asym_tb], hog])
        feats.append(feat)

    return np.array(feats)


print("\n[2] Extrayendo características...")
X = extract_features(X_imgs)
print(f"    Dimensión del vector de características: {X.shape[1]}")

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ─── 3. División train/test (ANTES de augmentation) ──────────────────────────
# Importante: dividir primero, luego aumentar solo el train
# para que el test refleje condiciones reales sin imágenes artificiales
X_train_raw, X_test, y_train_raw, y_test, fn_train, fn_test = train_test_split(
    X_scaled, y, filenames,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Augmentation sobre el conjunto de entrenamiento
X_imgs_train = X_imgs[[i for i, fn in enumerate(filenames) if fn in set(fn_train)]]
y_imgs_train  = y[[i for i, fn in enumerate(filenames) if fn in set(fn_train)]]
fn_imgs_train = [fn for fn in filenames if fn in set(fn_train)]

X_aug_imgs, y_aug, fn_aug = augment(X_imgs_train, y_imgs_train, fn_imgs_train)
X_aug_feats  = extract_features(X_aug_imgs)
X_train      = scaler.transform(X_aug_feats)
y_train      = y_aug

print(f"\n[3] División del dataset (con augmentation en train):")
print(f"    Original  — train: {len(y_train_raw)}, test: {len(y_test)}")
print(f"    Augmented — train: {len(y_train)} ({np.sum(y_train==0)} izq / {np.sum(y_train==1)} der)")
print(f"    Prueba    — sin augmentation: {len(y_test)} ({np.sum(y_test==0)} izq / {np.sum(y_test==1)} der)")


# ─── 4. Entrenamiento con Gradient Descent (con memoria) ──────────────────────
LR      = 0.1    # η: tasa de aprendizaje
N_ITER  = 2000   # épocas de GD
LAM     = 0.01   # λ: regularización L2

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not FORCE_RETRAIN and os.path.exists(MODEL_PATH):
    # ── MODO MEMORIA: cargar pesos ya entrenados ──────────────────────────────
    # X_train y X_test ya están correctamente escalados desde el paso [2].
    # Solo necesitamos cargar el modelo (β y b); no hay que re-escalar nada.
    print(f"\n[4] Cargando modelo desde memoria: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    model = checkpoint["model"]
    print(f"    β cargado  — costo final guardado: {model.history[-1]:.4f}")
    cv_scores = np.array([float("nan")])
    print(f"    (Validación cruzada omitida al cargar desde memoria)")

else:
    # ── MODO ENTRENAMIENTO: Gradient Descent desde cero ───────────────────────
    print(f"\n[4] Entrenando con Gradient Descent (η={LR}, épocas={N_ITER}, λ={LAM})...")
    model = LogisticRegressionGD(learning_rate=LR, n_iter=N_ITER, lambda_reg=LAM)
    model.fit(X_train, y_train)
    print(f"    Costo inicial : {model.history[0]:.4f}")
    print(f"    Costo final   : {model.history[-1]:.4f}")

    # Validación cruzada 5-fold manual
    print("    Ejecutando validación cruzada 5-fold...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for train_idx, val_idx in kf.split(X_train):
        m_cv = LogisticRegressionGD(learning_rate=LR, n_iter=N_ITER, lambda_reg=LAM)
        m_cv.fit(X_train[train_idx], y_train[train_idx])
        preds = m_cv.predict(X_train[val_idx])
        cv_scores.append(accuracy_score(y_train[val_idx], preds))
    cv_scores = np.array(cv_scores)
    print(f"    Validación cruzada 5-fold: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── GUARDAR MEMORIA ───────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"    Modelo guardado en: {MODEL_PATH}")


# ─── 5. Evaluación ───────────────────────────────────────────────────────────
print("\n[5] Evaluación en conjunto de prueba:")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)   # ya devuelve P(y=1|x) directamente

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"    Accuracy  : {acc:.4f}")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")
print(f"    F1-score  : {f1:.4f}")
print(f"    AUC-ROC   : {roc_auc:.4f}")


# ─── 6. Visualizaciones ──────────────────────────────────────────────────────
print("\n[6] Generando visualizaciones...")
os.makedirs(os.path.join(os.path.dirname(__file__), "Results"), exist_ok=True)
results_dir = os.path.join(os.path.dirname(__file__), "Results")

fig = plt.figure(figsize=(18, 12))
fig.suptitle("TE3002B — Clasificador de Flechas (Regresión Logística + Gradient Descent)", fontsize=14, fontweight="bold")
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.38)

# 6a. Matriz de confusión
ax1 = fig.add_subplot(gs[0, 0])
cm  = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Izquierda", "Derecha"])
disp.plot(ax=ax1, colorbar=False)
ax1.set_title("Matriz de Confusión")

# 6b. Curva ROC
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
ax2.plot([0, 1], [0, 1], "k--", lw=1)
ax2.set_xlabel("FPR (False Positive Rate)")
ax2.set_ylabel("TPR (True Positive Rate)")
ax2.set_title("Curva ROC")
ax2.legend()

# 6c. Distribución del dataset
ax3 = fig.add_subplot(gs[0, 2])
counts = [np.sum(y == 0), np.sum(y == 1)]
ax3.bar(["Izquierda\n(y=0)", "Derecha\n(y=1)"], counts, color=["#4878d0", "#ee854a"])
ax3.set_title("Balance del Dataset")
ax3.set_ylabel("Número de imágenes")
for i, c in enumerate(counts):
    ax3.text(i, c + 1, str(c), ha="center")

# 6d. Curva de convergencia del Gradient Descent
ax_conv = fig.add_subplot(gs[0, 3])
ax_conv.plot(model.history, color="steelblue", lw=1.5)
ax_conv.set_xlabel("Época")
ax_conv.set_ylabel("Costo J(β)")
ax_conv.set_title(f"Convergencia GD\n(η={LR}, λ={LAM})")
ax_conv.set_yscale("log")

# 6e. Proyección de columnas promedio por clase
ax4 = fig.add_subplot(gs[1, 0:2])
left_feats  = X_imgs[y == 0]
right_feats = X_imgs[y == 1]
left_proj   = (1 - left_feats).mean(axis=0).sum(axis=0)   # columna promedio
right_proj  = (1 - right_feats).mean(axis=0).sum(axis=0)

cols = np.arange(IMG_SIZE)
ax4.plot(cols, left_proj,  label="Izquierda (y=0)", color="#4878d0", lw=2)
ax4.plot(cols, right_proj, label="Derecha   (y=1)", color="#ee854a", lw=2)
ax4.axvline(IMG_SIZE // 2, color="gray", linestyle="--", lw=1, label="Centro")
ax4.set_xlabel("Columna (píxel)")
ax4.set_ylabel("Intensidad oscura promedio")
ax4.set_title("Proyección de Columnas (promedio por clase)")
ax4.legend()

# 6f. Métricas resumen
ax5 = fig.add_subplot(gs[1, 2])
metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": roc_auc}
bars = ax5.bar(metrics.keys(), metrics.values(), color=["#4878d0","#6acc65","#ee854a","#d65f5f","#956cb4"])
ax5.set_ylim(0, 1.1)
ax5.set_title("Resumen de Métricas")
ax5.set_ylabel("Valor")
for bar in bars:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.3f}", ha="center", fontsize=9)

out_path = os.path.join(results_dir, "resultados.png")
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"    Guardado en: {out_path}")


# ─── 7. Ejemplos de aciertos y errores ───────────────────────────────────────
errors_idx   = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt != yp]
correct_idx  = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt == yp]

n_show = min(4, len(errors_idx))
n_ok   = min(4, len(correct_idx))

fig2, axes = plt.subplots(2, max(n_show, n_ok, 1), figsize=(14, 6))
fig2.suptitle("Ejemplos de predicciones", fontsize=12, fontweight="bold")

label_str = {0: "Izq", 1: "Der"}

for col in range(max(n_show, n_ok)):
    # Fila 0: aciertos
    if col < n_ok:
        idx  = correct_idx[col]
        orig = fn_test[idx]
        img  = cv2.imread(os.path.join(DATASET_PATH, orig))
        img  = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB)
        ax   = axes[0, col] if max(n_show, n_ok) > 1 else axes[0]
        ax.imshow(img)
        ax.set_title(f"Real:{label_str[y_test[idx]]} Pred:{label_str[y_pred[idx]]}", fontsize=8, color="green")
        ax.axis("off")
    else:
        ax = axes[0, col]
        ax.axis("off")

    # Fila 1: errores
    if col < n_show:
        idx  = errors_idx[col]
        orig = fn_test[idx]
        img  = cv2.imread(os.path.join(DATASET_PATH, orig))
        img  = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB)
        ax   = axes[1, col] if max(n_show, n_ok) > 1 else axes[1]
        ax.imshow(img)
        ax.set_title(f"Real:{label_str[y_test[idx]]} Pred:{label_str[y_pred[idx]]}", fontsize=8, color="red")
        ax.axis("off")
    else:
        ax = axes[1, col] if max(n_show, n_ok) > 1 else axes[1]
        ax.axis("off")

axes[0, 0].set_ylabel("Aciertos", fontsize=10)
axes[1, 0].set_ylabel(f"Errores ({len(errors_idx)} total)", fontsize=10)

out2 = os.path.join(results_dir, "ejemplos.png")
plt.savefig(out2, dpi=100, bbox_inches="tight")
plt.close()
print(f"    Guardado en: {out2}")

print(f"\n{'='*60}")
print(f"  Errores en prueba: {len(errors_idx)} / {len(y_test)}")
if errors_idx:
    print("  Archivos con error:")
    for i in errors_idx:
        print(f"    {fn_test[i]}  → Real: {label_str[y_test[i]]}, Pred: {label_str[y_pred[i]]}")
print(f"{'='*60}\n")
