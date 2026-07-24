import os
import cv2
import numpy as np
from tensorflow import keras

# ============================================================
# CAI PROIECT
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_manual_cnn_best.h5")

CUSTOM_DATASET_DIR = os.path.join(BASE_DIR, "..", "DATASET_MNIST_PROCESAT")

OUT_CSV = os.path.join(BASE_DIR, "results_test_1_quantizare_globala.csv")

# ============================================================
# FORMATE TESTATE
# ============================================================

FORMATS = [
    ("Float32", None, None),

    # Variante realiste
    ("ap_fixed<16,6>", 16, 6),
    ("ap_fixed<14,5>", 14, 5),
    ("ap_fixed<12,5>", 12, 5),
    ("ap_fixed<12,4>", 12, 4),
    ("ap_fixed<10,4>", 10, 4),
    ("ap_fixed<8,3>", 8, 3),

    # Variante agresive, ca sa se vada degradarea
    ("ap_fixed<7,3>", 7, 3),
    ("ap_fixed<6,2>", 6, 2),
    ("ap_fixed<5,2>", 5, 2),
    ("ap_fixed<4,2>", 4, 2),
]

# ============================================================
# CUANTIZARE AP_FIXED
# ============================================================

def quantize_ap_fixed(arr, total_bits, int_bits):
    frac_bits = total_bits - int_bits
    scale = 2 ** frac_bits

    min_val = -(2 ** (int_bits - 1))
    max_val = (2 ** (int_bits - 1)) - (1.0 / scale)

    arr_q = np.round(arr * scale) / scale
    arr_q = np.clip(arr_q, min_val, max_val)

    return arr_q.astype(np.float32)

# ============================================================
# DATASET CUSTOM
# ============================================================

def load_custom_dataset(dataset_dir):
    images = []
    labels = []

    if not os.path.exists(dataset_dir):
        print(f"[INFO] Dataset custom nu exista: {dataset_dir}")
        return None, None

    print()
    print("======================================")
    print("INCARC DATASET CUSTOM")
    print("======================================")

    for label in range(10):
        folder = os.path.join(dataset_dir, str(label))

        if not os.path.exists(folder):
            print(f"[WARN] Lipseste folderul pentru cifra {label}: {folder}")
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

        print(f"Cifra {label}: {len(files)} imagini")

        for name in files:
            path = os.path.join(folder, name)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            img = img.astype("float32") / 255.0

            # vrem cifra alba pe fundal negru
            if np.mean(img) > 0.5:
                img = 1.0 - img

            images.append(img)
            labels.append(label)

    if len(images) == 0:
        print("[INFO] Nu am gasit imagini custom.")
        return None, None

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    images = np.expand_dims(images, axis=-1)

    print("--------------------------------------")
    print("Total imagini custom:", len(images))
    print("Shape:", images.shape)
    print("======================================")
    print()

    return images, labels

# ============================================================
# MODEL CU TOATE GREUTATILE CUANTIZATE
# ============================================================

def make_quantized_model(model_float, total_bits, int_bits):
    model_q = keras.models.clone_model(model_float)
    model_q.set_weights(model_float.get_weights())

    weights_q = []

    for w in model_float.get_weights():
        w_q = quantize_ap_fixed(w, total_bits, int_bits)
        weights_q.append(w_q)

    model_q.set_weights(weights_q)

    model_q.compile(
        optimizer="adam",
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model_q

# ============================================================
# MAIN
# ============================================================

def main():
    print("======================================")
    print("TEST 1 - CUANTIZARE GLOBALA")
    print("======================================")
    print("Acest test cuantizeaza TOATE greutatile si bias-urile.")
    print("Calculele interne Keras raman in float32.")
    print()
    print("Model:", MODEL_PATH)
    print("Dataset custom:", CUSTOM_DATASET_DIR)
    print()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Nu gasesc modelul: {MODEL_PATH}")

    model_float = keras.models.load_model(MODEL_PATH)

    model_float.compile(
        optimizer="adam",
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    print("Arhitectura model:")
    model_float.summary()
    print()

    # ========================================================
    # MNIST
    # ========================================================

    print("Incarc MNIST test...")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)

    y_test_cat = keras.utils.to_categorical(y_test, 10)

    print("MNIST test shape:", x_test.shape)
    print()

    # ========================================================
    # CUSTOM
    # ========================================================

    x_custom, y_custom = load_custom_dataset(CUSTOM_DATASET_DIR)

    if x_custom is not None:
        y_custom_cat = keras.utils.to_categorical(y_custom, 10)
    else:
        y_custom_cat = None

    # ========================================================
    # EVALUARE
    # ========================================================

    results = []

    print()
    print("======================================")
    print("REZULTATE")
    print("======================================")

    for name, total_bits, int_bits in FORMATS:
        print()
        print("--------------------------------------")
        print(name)
        print("--------------------------------------")

        if name == "Float32":
            model_eval = model_float
        else:
            model_eval = make_quantized_model(model_float, total_bits, int_bits)

        mnist_loss, mnist_acc = model_eval.evaluate(
            x_test,
            y_test_cat,
            verbose=0
        )

        print(f"MNIST accuracy:  {mnist_acc:.6f}")
        print(f"MNIST loss:      {mnist_loss:.6f}")

        if x_custom is not None:
            custom_loss, custom_acc = model_eval.evaluate(
                x_custom,
                y_custom_cat,
                verbose=0
            )

            print(f"Custom accuracy: {custom_acc:.6f}")
            print(f"Custom loss:     {custom_loss:.6f}")
        else:
            custom_acc = None
            custom_loss = None

            print("Custom accuracy: N/A")
            print("Custom loss:     N/A")

        results.append({
            "format": name,
            "mnist_acc": mnist_acc,
            "mnist_loss": mnist_loss,
            "custom_acc": custom_acc,
            "custom_loss": custom_loss,
        })

    # ========================================================
    # TABEL FINAL
    # ========================================================

    print()
    print("======================================")
    print("TABEL FINAL - TEST 1")
    print("======================================")
    print(f"{'Format':<18} {'MNIST acc':<12} {'Custom acc':<12} {'MNIST loss':<12} {'Custom loss':<12}")

    for r in results:
        custom_acc_str = "N/A" if r["custom_acc"] is None else f"{r['custom_acc']:.6f}"
        custom_loss_str = "N/A" if r["custom_loss"] is None else f"{r['custom_loss']:.6f}"

        print(
            f"{r['format']:<18} "
            f"{r['mnist_acc']:<12.6f} "
            f"{custom_acc_str:<12} "
            f"{r['mnist_loss']:<12.6f} "
            f"{custom_loss_str:<12}"
        )

    # ========================================================
    # CSV
    # ========================================================

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("format,mnist_accuracy,custom_accuracy,mnist_loss,custom_loss\n")

        for r in results:
            custom_acc_str = "" if r["custom_acc"] is None else f"{r['custom_acc']:.6f}"
            custom_loss_str = "" if r["custom_loss"] is None else f"{r['custom_loss']:.6f}"

            f.write(
                f"{r['format']},"
                f"{r['mnist_acc']:.6f},"
                f"{custom_acc_str},"
                f"{r['mnist_loss']:.6f},"
                f"{custom_loss_str}\n"
            )

    print()
    print("Rezultatele au fost salvate in:")
    print(OUT_CSV)
    print("======================================")


if __name__ == "__main__":
    main()