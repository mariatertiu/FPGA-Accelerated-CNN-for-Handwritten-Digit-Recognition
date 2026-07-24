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

OUT_CSV = os.path.join(BASE_DIR, "results_test_2_sensibilitate_straturi.csv")

# ============================================================
# SETARI TEST
# ============================================================

LAYER_NAMES = ["conv1", "conv2", "fc1", "fc2"]

FORMATS = [
    ("ap_fixed<16,6>", 16, 6),
    ("ap_fixed<12,4>", 12, 4),
    ("ap_fixed<10,4>", 10, 4),
    ("ap_fixed<8,3>", 8, 3),
    ("ap_fixed<6,2>", 6, 2),
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

            # Cifra alba pe fundal negru
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
# MODEL CU ANUMITE STRATURI CUANTIZATE
# ============================================================

def make_model_with_quantized_layers(model_float, total_bits, int_bits, layers_to_quantize):
    model_q = keras.models.clone_model(model_float)
    model_q.set_weights(model_float.get_weights())

    for layer_name in layers_to_quantize:
        layer_float = model_float.get_layer(layer_name)
        layer_q = model_q.get_layer(layer_name)

        weights = layer_float.get_weights()

        if len(weights) == 0:
            continue

        weights_q = [
            quantize_ap_fixed(w, total_bits, int_bits)
            for w in weights
        ]

        layer_q.set_weights(weights_q)

    model_q.compile(
        optimizer="adam",
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model_q

# ============================================================
# EVALUARE MODEL
# ============================================================

def evaluate_model(model, x_test, y_test_cat, x_custom, y_custom_cat):
    mnist_loss, mnist_acc = model.evaluate(
        x_test,
        y_test_cat,
        verbose=0
    )

    if x_custom is not None:
        custom_loss, custom_acc = model.evaluate(
            x_custom,
            y_custom_cat,
            verbose=0
        )
    else:
        custom_loss, custom_acc = None, None

    return mnist_acc, mnist_loss, custom_acc, custom_loss

# ============================================================
# MAIN
# ============================================================

def main():
    print("======================================")
    print("TEST 2 - SENSIBILITATE PE STRATURI")
    print("======================================")
    print("Acest test cuantizeaza pe rand fiecare strat.")
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

    print("Verific layerele testate:")
    for name in LAYER_NAMES:
        layer = model_float.get_layer(name)
        weights = layer.get_weights()
        shapes = [w.shape for w in weights]
        print(f"{name}: {shapes}")
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
    # TEST FLOAT32
    # ========================================================

    results = []

    print()
    print("======================================")
    print("FLOAT32 ORIGINAL")
    print("======================================")

    mnist_acc, mnist_loss, custom_acc, custom_loss = evaluate_model(
        model_float,
        x_test,
        y_test_cat,
        x_custom,
        y_custom_cat
    )

    print(f"MNIST accuracy:  {mnist_acc:.6f}")
    print(f"MNIST loss:      {mnist_loss:.6f}")

    if custom_acc is not None:
        print(f"Custom accuracy: {custom_acc:.6f}")
        print(f"Custom loss:     {custom_loss:.6f}")
    else:
        print("Custom accuracy: N/A")
        print("Custom loss:     N/A")

    results.append({
        "format": "Float32",
        "test_type": "original",
        "layers": "none",
        "mnist_acc": mnist_acc,
        "mnist_loss": mnist_loss,
        "custom_acc": custom_acc,
        "custom_loss": custom_loss,
    })

    # ========================================================
    # TESTE PE FORMATE
    # ========================================================

    for format_name, total_bits, int_bits in FORMATS:
        print()
        print("======================================")
        print(format_name)
        print("======================================")

        tests = []

        # Toate straturile cuantizate
        tests.append(("all_layers", LAYER_NAMES))

        # Cate un strat pe rand
        for layer_name in LAYER_NAMES:
            tests.append((f"only_{layer_name}", [layer_name]))

        for test_type, layers_to_quantize in tests:
            print()
            print("--------------------------------------")
            print(f"Test: {test_type}")
            print(f"Layere cuantizate: {layers_to_quantize}")
            print("--------------------------------------")

            model_q = make_model_with_quantized_layers(
                model_float,
                total_bits,
                int_bits,
                layers_to_quantize
            )

            mnist_acc, mnist_loss, custom_acc, custom_loss = evaluate_model(
                model_q,
                x_test,
                y_test_cat,
                x_custom,
                y_custom_cat
            )

            print(f"MNIST accuracy:  {mnist_acc:.6f}")
            print(f"MNIST loss:      {mnist_loss:.6f}")

            if custom_acc is not None:
                print(f"Custom accuracy: {custom_acc:.6f}")
                print(f"Custom loss:     {custom_loss:.6f}")
            else:
                print("Custom accuracy: N/A")
                print("Custom loss:     N/A")

            results.append({
                "format": format_name,
                "test_type": test_type,
                "layers": "+".join(layers_to_quantize),
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
    print("TABEL FINAL - TEST 2")
    print("======================================")
    print(
        f"{'Format':<18} "
        f"{'Test':<14} "
        f"{'Layers':<22} "
        f"{'MNIST acc':<12} "
        f"{'Custom acc':<12} "
        f"{'MNIST loss':<12} "
        f"{'Custom loss':<12}"
    )

    for r in results:
        custom_acc_str = "N/A" if r["custom_acc"] is None else f"{r['custom_acc']:.6f}"
        custom_loss_str = "N/A" if r["custom_loss"] is None else f"{r['custom_loss']:.6f}"

        print(
            f"{r['format']:<18} "
            f"{r['test_type']:<14} "
            f"{r['layers']:<22} "
            f"{r['mnist_acc']:<12.6f} "
            f"{custom_acc_str:<12} "
            f"{r['mnist_loss']:<12.6f} "
            f"{custom_loss_str:<12}"
        )

    # ========================================================
    # CSV
    # ========================================================

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("format,test_type,layers,mnist_accuracy,custom_accuracy,mnist_loss,custom_loss\n")

        for r in results:
            custom_acc_str = "" if r["custom_acc"] is None else f"{r['custom_acc']:.6f}"
            custom_loss_str = "" if r["custom_loss"] is None else f"{r['custom_loss']:.6f}"

            f.write(
                f"{r['format']},"
                f"{r['test_type']},"
                f"{r['layers']},"
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