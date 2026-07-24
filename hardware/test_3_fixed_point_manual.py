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

OUT_CSV = os.path.join(BASE_DIR, "results_test_3_fixed_point_manual.csv")

# ============================================================
# SETARI TEST
# ============================================================

# Pentru test rapid pune 100.
# Pentru documentatie pune 1000.
# Pentru mai mult poate dura foarte mult.
NUM_MNIST_TEST = 100

# Ruleaza pe rand. Intai 16,6.
# Dupa ce merge, schimbi lista pentru 12,4, apoi optional 10,4.
FORMATS = [
    ("ap_fixed<16,6>", 16, 6),
    # ("ap_fixed<12,4>", 12, 4),
    # ("ap_fixed<10,4>", 10, 4),
]

# Daca datasetul custom este foarte mare si dureaza mult,
# poti limita numarul de imagini custom. None = toate.
MAX_CUSTOM_TEST = None
# MAX_CUSTOM_TEST = 1000

# ============================================================
# CUANTIZARE AP_FIXED
# ============================================================

def quantize_ap_fixed(x, total_bits, int_bits):
    frac_bits = total_bits - int_bits
    scale = 2 ** frac_bits

    min_val = -(2 ** (int_bits - 1))
    max_val = (2 ** (int_bits - 1)) - (1.0 / scale)

    x_q = np.round(x * scale) / scale
    x_q = np.clip(x_q, min_val, max_val)

    return x_q.astype(np.float32)


def relu(x):
    return np.maximum(x, 0.0).astype(np.float32)

# ============================================================
# LAYERE MANUALE
# ============================================================

def conv2d_valid(x, w, b, total_bits, int_bits):
    h, w_in, in_c = x.shape
    kh, kw, _, out_c = w.shape

    out_h = h - kh + 1
    out_w = w_in - kw + 1

    out = np.zeros((out_h, out_w, out_c), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            for oc in range(out_c):
                acc = quantize_ap_fixed(b[oc], total_bits, int_bits)

                for r in range(kh):
                    for c in range(kw):
                        for ic in range(in_c):
                            mul = x[i + r, j + c, ic] * w[r, c, ic, oc]
                            mul = quantize_ap_fixed(mul, total_bits, int_bits)
                            acc = quantize_ap_fixed(acc + mul, total_bits, int_bits)

                out[i, j, oc] = acc

    return out


def maxpool2x2(x):
    h, w, c = x.shape

    out_h = h // 2
    out_w = w // 2

    out = np.zeros((out_h, out_w, c), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            for ch in range(c):
                patch = x[i * 2:i * 2 + 2, j * 2:j * 2 + 2, ch]
                out[i, j, ch] = np.max(patch)

    return out


def dense(x, w, b, total_bits, int_bits):
    out_dim = b.shape[0]
    out = np.zeros((out_dim,), dtype=np.float32)

    for o in range(out_dim):
        acc = quantize_ap_fixed(b[o], total_bits, int_bits)

        for i in range(x.shape[0]):
            mul = x[i] * w[i, o]
            mul = quantize_ap_fixed(mul, total_bits, int_bits)
            acc = quantize_ap_fixed(acc + mul, total_bits, int_bits)

        out[o] = acc

    return out

# ============================================================
# FORWARD MANUAL FIXED-POINT
# ============================================================

def forward_fixed_point(img, weights_q, total_bits, int_bits):
    conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b = weights_q

    # input: 28x28 -> 28x28x1
    x = img.astype(np.float32)
    x = quantize_ap_fixed(x, total_bits, int_bits)
    x = np.expand_dims(x, axis=-1)

    # Conv1 + ReLU
    x = conv2d_valid(x, conv1_w, conv1_b, total_bits, int_bits)
    x = relu(x)
    x = quantize_ap_fixed(x, total_bits, int_bits)

    # Pool1
    x = maxpool2x2(x)
    x = quantize_ap_fixed(x, total_bits, int_bits)

    # Conv2 + ReLU
    x = conv2d_valid(x, conv2_w, conv2_b, total_bits, int_bits)
    x = relu(x)
    x = quantize_ap_fixed(x, total_bits, int_bits)

    # Pool2
    x = maxpool2x2(x)
    x = quantize_ap_fixed(x, total_bits, int_bits)

    # Flatten
    x = x.reshape(-1)
    x = quantize_ap_fixed(x, total_bits, int_bits)

    # FC1 + ReLU
    x = dense(x, fc1_w, fc1_b, total_bits, int_bits)
    x = relu(x)
    x = quantize_ap_fixed(x, total_bits, int_bits)

    # FC2
    x = dense(x, fc2_w, fc2_b, total_bits, int_bits)
    x = quantize_ap_fixed(x, total_bits, int_bits)

    return x

# ============================================================
# DATASET CUSTOM
# ============================================================

def load_custom_dataset(dataset_dir, max_samples=None):
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
            if max_samples is not None and len(images) >= max_samples:
                break

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

        if max_samples is not None and len(images) >= max_samples:
            break

    if len(images) == 0:
        print("[INFO] Nu am gasit imagini custom.")
        return None, None

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print("--------------------------------------")
    print("Total imagini custom folosite:", len(images))
    print("Shape:", images.shape)
    print("======================================")
    print()

    return images, labels

# ============================================================
# EVALUARE MANUALA
# ============================================================

def evaluate_manual(images, labels, weights_float, total_bits, int_bits, name):
    # Cuantizam greutatile si bias-urile
    weights_q = [
        quantize_ap_fixed(w, total_bits, int_bits)
        for w in weights_float
    ]

    correct = 0
    total = len(images)

    for n in range(total):
        logits = forward_fixed_point(
            images[n],
            weights_q,
            total_bits,
            int_bits
        )

        pred = int(np.argmax(logits))

        if pred == int(labels[n]):
            correct += 1

        if n < 10:
            print(f"{name} sample {n}: label={labels[n]}, pred={pred}")

        if (n + 1) % 100 == 0:
            print(f"{name}: procesate {n + 1}/{total}")

    acc = correct / total

    print()
    print(f"{name} accuracy: {acc:.6f}")
    print(f"{name} correct: {correct} / {total}")
    print()

    return acc, correct, total

# ============================================================
# MAIN
# ============================================================

def main():
    print("======================================")
    print("TEST 3 - SIMULARE FIXED-POINT MANUALA")
    print("======================================")
    print("Acest test cuantizeaza greutati, bias-uri, input, produse, acumulari si activari.")
    print("Este mai apropiat de HLS decat testele Keras, dar este mult mai lent.")
    print()
    print("Model:", MODEL_PATH)
    print("Dataset custom:", CUSTOM_DATASET_DIR)
    print("NUM_MNIST_TEST:", NUM_MNIST_TEST)
    print("MAX_CUSTOM_TEST:", MAX_CUSTOM_TEST)
    print()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Nu gasesc modelul: {MODEL_PATH}")

    model = keras.models.load_model(MODEL_PATH)

    conv1_w, conv1_b = model.get_layer("conv1").get_weights()
    conv2_w, conv2_b = model.get_layer("conv2").get_weights()
    fc1_w, fc1_b = model.get_layer("fc1").get_weights()
    fc2_w, fc2_b = model.get_layer("fc2").get_weights()

    weights_float = [
        conv1_w, conv1_b,
        conv2_w, conv2_b,
        fc1_w, fc1_b,
        fc2_w, fc2_b
    ]

    print("Shapes:")
    print("conv1_w:", conv1_w.shape, "conv1_b:", conv1_b.shape)
    print("conv2_w:", conv2_w.shape, "conv2_b:", conv2_b.shape)
    print("fc1_w:", fc1_w.shape, "fc1_b:", fc1_b.shape)
    print("fc2_w:", fc2_w.shape, "fc2_b:", fc2_b.shape)
    print()

    # ========================================================
    # MNIST
    # ========================================================

    print("Incarc MNIST...")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_test = x_test.astype("float32") / 255.0
    x_test = x_test[:NUM_MNIST_TEST]
    y_test = y_test[:NUM_MNIST_TEST]

    print("MNIST folosit:", x_test.shape)
    print()

    # ========================================================
    # CUSTOM
    # ========================================================

    x_custom, y_custom = load_custom_dataset(
        CUSTOM_DATASET_DIR,
        max_samples=MAX_CUSTOM_TEST
    )

    # ========================================================
    # TESTE
    # ========================================================

    results = []

    for format_name, total_bits, int_bits in FORMATS:
        print()
        print("======================================")
        print(format_name)
        print("======================================")

        mnist_acc, mnist_correct, mnist_total = evaluate_manual(
            x_test,
            y_test,
            weights_float,
            total_bits,
            int_bits,
            "MNIST"
        )

        if x_custom is not None:
            custom_acc, custom_correct, custom_total = evaluate_manual(
                x_custom,
                y_custom,
                weights_float,
                total_bits,
                int_bits,
                "CUSTOM"
            )
        else:
            custom_acc = None
            custom_correct = None
            custom_total = None

        results.append({
            "format": format_name,
            "mnist_acc": mnist_acc,
            "mnist_correct": mnist_correct,
            "mnist_total": mnist_total,
            "custom_acc": custom_acc,
            "custom_correct": custom_correct,
            "custom_total": custom_total,
        })

    # ========================================================
    # TABEL FINAL
    # ========================================================

    print()
    print("======================================")
    print("TABEL FINAL - TEST 3")
    print("======================================")
    print(
        f"{'Format':<18} "
        f"{'MNIST acc':<12} "
        f"{'MNIST correct':<15} "
        f"{'Custom acc':<12} "
        f"{'Custom correct':<15}"
    )

    for r in results:
        mnist_correct_str = f"{r['mnist_correct']}/{r['mnist_total']}"

        if r["custom_acc"] is None:
            custom_acc_str = "N/A"
            custom_correct_str = "N/A"
        else:
            custom_acc_str = f"{r['custom_acc']:.6f}"
            custom_correct_str = f"{r['custom_correct']}/{r['custom_total']}"

        print(
            f"{r['format']:<18} "
            f"{r['mnist_acc']:<12.6f} "
            f"{mnist_correct_str:<15} "
            f"{custom_acc_str:<12} "
            f"{custom_correct_str:<15}"
        )

    # ========================================================
    # CSV
    # ========================================================

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("format,mnist_accuracy,mnist_correct,mnist_total,custom_accuracy,custom_correct,custom_total\n")

        for r in results:
            if r["custom_acc"] is None:
                custom_acc_str = ""
                custom_correct_str = ""
                custom_total_str = ""
            else:
                custom_acc_str = f"{r['custom_acc']:.6f}"
                custom_correct_str = str(r["custom_correct"])
                custom_total_str = str(r["custom_total"])

            f.write(
                f"{r['format']},"
                f"{r['mnist_acc']:.6f},"
                f"{r['mnist_correct']},"
                f"{r['mnist_total']},"
                f"{custom_acc_str},"
                f"{custom_correct_str},"
                f"{custom_total_str}\n"
            )

    print()
    print("Rezultatele au fost salvate in:")
    print(OUT_CSV)
    print("======================================")


if __name__ == "__main__":
    main()