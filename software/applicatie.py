import cv2
import numpy as np
import os
import socket
from PIL import Image, ImageDraw, ImageFont


# Setări generale pentru conexiunea cu placa FPGA și pentru camera folosită

# Adresa IP a plăcii ZedBoard.
BOARD_IP = "192.168.1.10"
# Portul pe care rulează serverul fpga_server pe ZedBoard.
BOARD_PORT = 12345

CAMERA_INDEX = 1  # 0 = camera laptop, 1/2 = telefon ca webcam, dupa caz

# Folderul în care se salvează imaginile de debug.
CAPTURE_DIR = "captures_multi"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Dacă este True, imaginea camerei este oglindită pe orizontală.
MIRROR_CAMERA = False

# Dimensiunea maximă a zonei analizate din imagine.
ROI_SIZE = 420

# Aria minimă pentru o componentă ca să fie considerată cifră.
MIN_COMPONENT_AREA_DEFAULT = 120
MAX_COMPONENT_AREA_RATIO_DEFAULT = 18  # procent, adica 18%

# Dimensiunea la care este redusă cifra înainte de plasarea în imaginea 28x28.
DIGIT_SIZE = 18

# Numele ferestrei principale afișate de OpenCV.
PAGE_WINDOW = "PREZENTARE LICENTA - CNN PE FPGA "

# Fereastra mare, redimensionabila. Pentru prezentare, maximizeaza fereastra.
PAGE_W = 1500
PAGE_H = 760

# Imaginea de fundal memorată la calibrare. Inițial nu există calibrare.
BACKGROUND_CROP = None


# Funcții pentru trackbar-urile din fereastra OpenCV

# Funcție goală folosită obligatoriu de trackbar-urile OpenCV.
def nothing(x):
    pass


# Creează fereastra principală și trackbar-urile pentru reglarea pragurilor.
def setup_controls():
    cv2.namedWindow(PAGE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PAGE_WINDOW, PAGE_W, PAGE_H)
    cv2.moveWindow(PAGE_WINDOW, 8, 20)

    cv2.createTrackbar("DIFF_THR", PAGE_WINDOW, 25, 100, nothing)
    cv2.createTrackbar("MIN_AREA", PAGE_WINDOW, MIN_COMPONENT_AREA_DEFAULT, 1000, nothing)
    cv2.createTrackbar("MAX_AREA_%", PAGE_WINDOW, MAX_COMPONENT_AREA_RATIO_DEFAULT, 50, nothing)


# Citește valorile curente ale trackbar-urilor și le limitează la valori sigure.
def get_control_values():
    diff_thr = cv2.getTrackbarPos("DIFF_THR", PAGE_WINDOW)
    # Citește pragul folosit pentru diferența față de fundalul calibrat.
    min_area = cv2.getTrackbarPos("MIN_AREA", PAGE_WINDOW)
    # Citește aria minimă acceptată pentru o cifră.
    max_area_percent = cv2.getTrackbarPos("MAX_AREA_%", PAGE_WINDOW)
    # Citește procentul maxim permis pentru aria unei componente.

    if diff_thr < 5:
        diff_thr = 5

    if min_area < 20:
        min_area = 20

    if max_area_percent < 3:
        max_area_percent = 3

    max_area_ratio = max_area_percent / 100.0

    return diff_thr, min_area, max_area_ratio


# Funcții pentru comunicarea TCP/IP cu ZedBoard

# Primește exact n bytes de la socket, nu doar cât vine într-un singur recv.
def recv_exact(sock, n):
    data = b""

    while len(data) < n:
        chunk = sock.recv(n - len(data))

        if not chunk:
            raise RuntimeError("Conexiunea s-a inchis prematur")

        data += chunk

    return data


# Transformă cifra 28x28 în 784 bytes și o trimite la ZedBoard.
def send_to_board(digit28):
    # Trimitem exact formatul asteptat de server:
    # 28x28 = 784 valori uint8.
    # Pentru ap_fixed<8,4>, valoarea 1.0 este reprezentata raw ca 16,
    # deoarece sunt 4 biti fractionari.
    # Fundal = 0, cifra = 16.
    x = (digit28 > 0).astype(np.uint8) * 16
    # Pixelii albi devin 16, adică 1.0 în format ap_fixed<8,4>.
    x = x.reshape(-1)

    if x.size != 784:
        raise ValueError(f"Dimensiune gresita pentru input: {x.size}, trebuia 784")

    payload = x.tobytes()
    # Vectorul de 784 valori este transformat în bytes pentru trimitere prin TCP.

    if len(payload) != 784:
        raise ValueError(f"Payload gresit: {len(payload)} bytes, trebuia 784")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect((BOARD_IP, BOARD_PORT))
        # Se conectează la serverul TCP de pe ZedBoard.
        s.sendall(payload)
        # Trimite toată imaginea 28x28 către placa FPGA.

        data = recv_exact(s, 4)
        # Așteaptă răspunsul de 4 bytes, adică cifra prezisă.

    result = int.from_bytes(data, byteorder="little", signed=True)
    # Interpretează răspunsul ca întreg semnat în format little-endian.

    return result


# Funcții pentru extragerea ROI-ului și preprocesarea imaginii

# Calculează coordonatele ROI-ului central din imaginea camerei.
def get_center_roi(frame):
    h, w = frame.shape[:2]

    roi_size = min(ROI_SIZE, h, w)
    # ROI-ul nu poate fi mai mare decât imaginea camerei.

    x1 = (w - roi_size) // 2
    y1 = (h - roi_size) // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    return x1, y1, x2, y2


# Recentrează cifra în imaginea 28x28 folosind centrul de masă al pixelilor albi.
def center_by_mass(img):
    m = cv2.moments(img)
    # Momentele sunt folosite pentru calcularea centrului de masă.

    if abs(m["m00"]) < 1e-5:
    # Dacă nu există pixeli albi suficienți, imaginea nu se poate recentra.
        return img

    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])

    M = np.float32([
        [1, 0, 14 - cx],
        [0, 1, 14 - cy]
    ])

    centered = cv2.warpAffine(
        img,
        M,
        (28, 28),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return centered.astype(np.uint8)


# Transformă o componentă detectată într-o imagine standard 28x28 pentru CNN.
def component_to_digit28(component_img):
    ys, xs = np.where(component_img > 0)
    # Găsește coordonatele pixelilor albi ai componentei.

    if len(xs) < 20:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    digit = component_img[y0:y1 + 1, x0:x1 + 1]
    # Decupează doar zona în care se află cifra.

    digit_h, digit_w = digit.shape[:2]

    if digit_h < 8 or digit_w < 3:
        return None

    size = max(digit_h, digit_w)

    square = np.zeros((size, size), dtype=np.uint8)
    # Creează o imagine pătrată ca cifra să nu fie deformată la redimensionare.

    y_off = (size - digit_h) // 2
    x_off = (size - digit_w) // 2

    square[
        y_off:y_off + digit_h,
        x_off:x_off + digit_w
    ] = digit

    digit18 = cv2.resize(
    # Redimensionează cifra la 18x18, lăsând margine liberă în imaginea finală 28x28.
        square,
        (DIGIT_SIZE, DIGIT_SIZE),
        interpolation=cv2.INTER_AREA
    )

    _, digit18 = cv2.threshold(
        digit18,
        20,
        255,
        cv2.THRESH_BINARY
    )

    digit28 = np.zeros((28, 28), dtype=np.uint8)
    # Creează imaginea finală de intrare pentru CNN.

    start = (28 - DIGIT_SIZE) // 2
    end = start + DIGIT_SIZE

    digit28[start:end, start:end] = digit18

    digit28 = center_by_mass(digit28)
    # Recentrează cifra după distribuția pixelilor activi.

    return digit28


# Convertește ROI-ul în gri, îmbunătățește contrastul și taie marginile.
def prepare_gray_crop(roi):
    gray_original = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Elimină informația de culoare și păstrează intensitatea.

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray_eq = clahe.apply(gray_original)
    # Îmbunătățește contrastul local.

    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    # Reduce zgomotul mic din imagine.

    h, w = blur.shape
    margin = int(0.05 * w)

    crop = blur[margin:h - margin, margin:w - margin]
    # Taie marginile ROI-ului ca să evite detecții false la margine.

    return gray_eq, crop, margin


# Alege imaginea binară folosită la detecție: Otsu/adaptiv sau diferență față de fundal.
def choose_threshold(crop, background_crop, diff_thr):
    _, thresh_otsu = cv2.threshold(
    # Creează o imagine binară folosind metoda Otsu.
        crop,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    thresh_adapt = cv2.adaptiveThreshold(
    # Creează o imagine binară folosind threshold adaptiv.
        crop,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )

    white_otsu = np.count_nonzero(thresh_otsu)
    # Numără pixelii albi din varianta Otsu.
    white_adapt = np.count_nonzero(thresh_adapt)
    # Numără pixelii albi din varianta adaptivă.

    if white_adapt > white_otsu * 2.5:
        thresh_base = thresh_otsu
    elif white_adapt < white_otsu * 0.25:
        thresh_base = thresh_otsu
    else:
        thresh_base = thresh_adapt

    if background_crop is not None and background_crop.shape == crop.shape:
        diff = cv2.subtract(background_crop, crop)
        # Calculează diferența dintre fundalul calibrat și imaginea curentă.

        _, thresh_bg = cv2.threshold(
        # Binarizează diferența față de fundal.
            diff,
            diff_thr,
            255,
            cv2.THRESH_BINARY
        )

        white_bg = np.count_nonzero(thresh_bg)
        area = crop.shape[0] * crop.shape[1]

        if area * 0.001 < white_bg < area * 0.35:
            return thresh_bg, thresh_base

    return thresh_base, thresh_base


# Curăță imaginea binară prin operații morfologice de deschidere și închidere.
def clean_threshold(thresh):
    kernel_open = np.ones((2, 2), np.uint8)
    thresh_clean = cv2.morphologyEx(
    # Aplică o operație morfologică pentru curățarea imaginii binare.
        thresh,
        cv2.MORPH_OPEN,
        kernel_open
    )

    kernel_close = np.ones((3, 3), np.uint8)
    thresh_clean = cv2.morphologyEx(
    # Aplică o operație morfologică pentru curățarea imaginii binare.
        thresh_clean,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    return thresh_clean


# Preprocesează ROI-ul și extrage toate componentele care pot fi cifre.
def preprocess_multi_digits(
    roi,
    background_crop,
    diff_thr,
    min_component_area,
    max_component_area_ratio
):
    gray_eq, crop, margin = prepare_gray_crop(roi)

    thresh_raw, thresh_base = choose_threshold(
        crop,
        background_crop,
        diff_thr
    )

    thresh_clean = clean_threshold(thresh_raw)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_clean)
    # Găsește toate componentele albe din imaginea binară.

    crop_h, crop_w = thresh_clean.shape
    crop_area = crop_h * crop_w

    components = []

    for i in range(1, num_labels):
    # Se pornește de la 1 deoarece eticheta 0 este fundalul.
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_component_area:
        # Elimină punctele mici sau zgomotul.
            continue

        if area > max_component_area_ratio * crop_area:
        # Elimină zonele prea mari ca să fie cifre.
            continue

        if bw < 5 or bh < 18:
        # Elimină componentele prea înguste sau prea joase.
            continue

        aspect = bh / max(bw, 1)
        # Calculează raportul înălțime/lățime.

        if aspect < 0.45:
        # Elimină formele prea late.
            continue

        if aspect > 8.0:
        # Elimină formele exagerat de înalte.
            continue

        border = 4

        if x <= border or y <= border:
        # Elimină componentele lipite de marginea ROI-ului.
            continue

        if x + bw >= crop_w - border:
            continue

        if y + bh >= crop_h - border:
            continue

        component = np.zeros_like(thresh_clean)
        component[labels == i] = 255
        # Construiește imaginea binară doar pentru componenta curentă.

        digit28 = component_to_digit28(component)
        # Normalizează componenta la formatul 28x28.

        if digit28 is None:
            continue

        if np.count_nonzero(digit28) < 20:
            continue

        bbox = (x + margin, y + margin, bw, bh)

        components.append({
            "bbox": bbox,
            "digit28": digit28,
            "area": area
        })

    components.sort(key=lambda item: item["bbox"][0])
    # Sortează cifrele de la stânga la dreapta.

    return gray_eq, thresh_base, thresh_raw, thresh_clean, components


# Funcție care trimite fiecare cifră detectată către FPGA

# Trimite pe rând fiecare cifră detectată către FPGA și salvează predicțiile.
def predict_components_fpga(components):
    results = []

    for comp in components:
    # Parcurge fiecare cifră detectată.
        digit28 = comp["digit28"]

        try:
            pred = send_to_board(digit28)
            # Trimite cifra curentă la FPGA și primește predicția.

            results.append({
                "bbox": comp["bbox"],
                "digit28": digit28,
                "pred": pred,
                "safe": True,
                "error": ""
            })

        except Exception as e:
            results.append({
                "bbox": comp["bbox"],
                "digit28": digit28,
                "pred": -1,
                "safe": False,
                "error": str(e)
            })

    return results


# Funcție pentru salvarea imaginilor și datelor de test

# Salvează ROI-ul, threshold-ul și cifrele 28x28 pentru verificări ulterioare.
def save_detection_debug(roi, thresh_clean, results, idx):
    folder = os.path.join(CAPTURE_DIR, f"cap_{idx:04d}")
    # Creează un folder separat pentru fiecare salvare de debug.
    os.makedirs(folder, exist_ok=True)

    cv2.imwrite(os.path.join(folder, "roi.png"), roi)
    # Salvează ROI-ul original.
    cv2.imwrite(os.path.join(folder, "threshold_clean.png"), thresh_clean)
    # Salvează imaginea binară curățată.

    for i, result in enumerate(results):
        digit28 = result["digit28"]
        pred = result["pred"]

        cv2.imwrite(
            os.path.join(folder, f"digit_{i}_pred_{pred}.png"),
            digit28
        )

        x = ((digit28 > 0).astype(np.uint8) * 16)

        np.savetxt(
        # Salvează vectorul de intrare 28x28 în format text.
            os.path.join(folder, f"digit_{i}_input28.txt"),
            x.reshape(-1),
            fmt="%d"
        )

    print(f"[OK] Debug salvat in {folder}")


# Funcții pentru construirea interfeței grafice principale și a ferestrei de debug

# Ideea interfetei:
#   - Fereastra principala contine doar zonele importante pentru prezentare:
#       1) Camera live
#       2) ROI cu detectii
#       3) Control + rezultat FPGA
#       4) Cifre 28x28 normalizate
#   - Threshold-urile se pot afisa separat cu tasta D, ca sa nu aglomereze pagina.
#   - Textul este desenat cu umbra + antialiasing pentru claritate.

SHOW_DEBUG_WINDOWS = False

# Dimensiune pagina principala. Pentru prezentare, maximizeaza fereastra.
PAGE_W = 1500
PAGE_H = 760

# Paleta simpla, cu contrast bun. OpenCV foloseste BGR.
COL_BG = (18, 18, 18)
COL_SURFACE = (255, 255, 255)
COL_SURFACE_2 = (245, 248, 252)
COL_HEADER = (0, 0, 0)
COL_PANEL_HEAD = (0, 0, 0)
COL_BORDER = (20, 20, 20)
# Textul de pe panourile deschise este negru real, nu gri.
COL_TEXT_DARK = (0, 0, 0)
COL_TEXT_MUTED = (0, 0, 0)
COL_TEXT_LIGHT = (255, 255, 255)
# Culori pastrate ca directie, dar facute mai intense. OpenCV foloseste BGR.
COL_OK = (0, 0, 0)
COL_ERR = (0, 0, 0)
COL_ACCENT = (255, 105, 0)
COL_ACCENT_2 = (0, 200, 255)
COL_YELLOW = (0, 245, 255)
COL_BLUE_TEXT = (0, 0, 0)

FONT_MAIN = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIMPLE = cv2.FONT_HERSHEY_SIMPLEX

# Funcții pentru afișarea textului clar în interfață folosind fonturi prin Pillow
# cv2.putText foloseste fonturi Hershey, care arata colturoase.
# Pentru text clar de tip Windows folosim Arial prin Pillow.

FONT_CACHE = {}


# Încarcă un font clar pentru textul din interfață.
def get_arial_font(size, bold=False):
    """Incarca Arial din Windows. Daca nu exista, foloseste fontul implicit PIL."""
    key = (int(size), bool(bold))
    if key in FONT_CACHE:
        return FONT_CACHE[key]

    possible_paths = []

    if bold:
        possible_paths += [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/Arialbd.ttf",
            "/mnt/c/Windows/Fonts/arialbd.ttf",
        ]

    possible_paths += [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/mnt/c/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, int(size))
                FONT_CACHE[key] = font
                return font
            except Exception:
                pass

    font = ImageFont.load_default()
    FONT_CACHE[key] = font
    return font


# Convertește o culoare din format BGR OpenCV în format RGB Pillow.
def bgr_to_rgb(color):
    return (int(color[2]), int(color[1]), int(color[0]))


# Scrie text pe imagine folosind Pillow, ca să fie mai clar decât cv2.putText.
def draw_text(
    img,
    text,
    org,
    scale=0.62,
    color=(255, 255, 255),
    thickness=1,
    shadow=False,
    font=FONT_MAIN
):
    """
    Text clar folosind Arial prin Pillow.
    Coordonata org ramane ca la OpenCV: y reprezinta linia de baza a textului.
    """
    text = str(text).upper()
    x, y = org

    # Conversie aproximativa din scale OpenCV in dimensiune font PIL.
    font_size = max(10, int(scale * 34))
    bold = thickness >= 2
    pil_font = get_arial_font(font_size, bold=bold)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # La PIL, y este partea de sus. La OpenCV, y este baza textului.
    text_x = int(x)
    text_y = int(y - font_size)

    if shadow:
        draw.text((text_x + 1, text_y + 1), text, font=pil_font, fill=(0, 0, 0))

    draw.text((text_x, text_y), text, font=pil_font, fill=bgr_to_rgb(color))

    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# Calculează dimensiunea textului ca să poată fi poziționat corect.
def text_size(text, scale=0.62, thickness=1, font=FONT_MAIN):
    font_size = max(10, int(scale * 34))
    pil_font = get_arial_font(font_size, bold=thickness >= 2)
    bbox = pil_font.getbbox(str(text).upper())
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# Desenează o etichetă mică, de tip buton, cu text în interior.
def draw_label_box(img, text, x, y, scale=0.56, fg=COL_TEXT_LIGHT, bg=COL_ACCENT):
    """Buton mic, curat, cu text Arial."""
    thickness = 1
    tw, th = text_size(str(text), scale, thickness)
    pad_x = 10
    pad_y = 7
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + tw + 2 * pad_x
    y2 = y1 + th + 2 * pad_y + 4
    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    draw_text(img, text, (x1 + pad_x, y1 + pad_y + th + 2), scale, fg, 1, shadow=False)
    return x2


# Scrie text într-un chenar și îl micșorează automat dacă este prea lung.
def draw_fitted_text(img, text, box, max_scale, min_scale, color, thickness=2, font=FONT_SIMPLE):
    """
    Deseneaza text Arial in interiorul unui dreptunghi.
    Folosit pentru REZULTAT FPGA, ca numerele cu mai multe cifre sa nu iasa din zona.
    """
    x, y, w, h = box
    text = str(text).upper()

    scale = max_scale
    while scale >= min_scale:
        tw, th = text_size(text, scale, thickness)
        if tw <= w and th <= h:
            break
        scale -= 0.05

    if scale < min_scale:
        scale = min_scale
        tw, th = text_size(text, scale, thickness)

    font_size = max(10, int(scale * 34))
    tx = x + max(0, (w - tw) // 2)
    # draw_text primeste y ca linie de baza, deci adaugam font_size.
    ty = y + max(font_size, (h + th) // 2 + font_size // 2 - 3)

    draw_text(img, text, (tx, ty), scale, color, thickness, shadow=False)


# Transformă o imagine grayscale în BGR pentru a putea fi afișată în panouri color.
def to_bgr(img):
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img.copy()


# Redimensionează o imagine ca să încapă într-un panou fără deformare.
def fit_to_box(img, box_w, box_h, bg_color=COL_SURFACE):
    img = to_bgr(img)
    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        return np.full((box_h, box_w, 3), bg_color, dtype=np.uint8)

    scale = min(box_w / w, box_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.full((box_h, box_w, 3), bg_color, dtype=np.uint8)

    x = (box_w - new_w) // 2
    y = (box_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


# Creează un panou cu titlu și conținut redimensionat.
def make_panel(img, title, w, h, title_h=34, bg_color=COL_SURFACE):
    """
    Panou clar: titlul este separat de imagine, deci nu se suprapune peste conținut.
    Chenarul este subțire, iar marginea albă este minimă.
    """
    panel = np.full((h, w, 3), bg_color, dtype=np.uint8)

    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), COL_BORDER, 1)
    cv2.rectangle(panel, (0, 0), (w, title_h), COL_PANEL_HEAD, -1)

    draw_text(
        panel,
        title,
        (12, 24),
        0.56,
        COL_TEXT_LIGHT,
        2,
        shadow=False,
        font=FONT_SIMPLE
    )

    content = fit_to_box(img, w - 6, h - title_h - 6, bg_color=bg_color)
    ch, cw = content.shape[:2]
    panel[title_h + 3:title_h + 3 + ch, 3:3 + cw] = content

    return panel


# Creează o casetă informativă cu etichetă și valoare.
def make_info_tile(label, value, w, h, value_scale=1.0, value_color=COL_TEXT_DARK):
    tile = np.full((h, w, 3), COL_SURFACE_2, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)
    draw_text(tile, label, (16, 36), 0.66, COL_TEXT_MUTED, 1, shadow=False)

    value = str(value)
    # Daca statusul este lung, folosim font mai mic ca sa ramana citibil.
    scale = value_scale
    if len(value) > 18:
        scale = min(scale, 0.72)
    if len(value) > 30:
        scale = min(scale, 0.56)

    draw_text(tile, value, (16, h - 24), scale, value_color, 1, shadow=False)
    return tile


# Construiește panoul cu rezultatul FPGA, statusul, calibrarea și comenzile.
def make_control_result_panel(
    last_number,
    last_status,
    components_count,
    diff_thr,
    min_component_area,
    max_component_area_ratio
):
    panel = np.full((175, 885, 3), COL_SURFACE, dtype=np.uint8)

    calib_text = "DA" if BACKGROUND_CROP is not None else "NU"

    result_text = last_number if last_number != "" else "-"
    status_text = last_status if last_status != "" else "ASTEAPTA COMANDA"

    # Coloana rezultat FPGA.
    draw_text(panel, "REZULTAT FPGA", (18, 30), 0.52, COL_TEXT_DARK, 2, shadow=False, font=FONT_SIMPLE)

    # Zona fixa pentru rezultat. Textul se micsoreaza automat ca sa incapa.
    result_box = (18, 48, 200, 96)
    cv2.rectangle(panel, (result_box[0], result_box[1]), (result_box[0] + result_box[2], result_box[1] + result_box[3]), (245, 245, 245), -1)
    cv2.rectangle(panel, (result_box[0], result_box[1]), (result_box[0] + result_box[2], result_box[1] + result_box[3]), (80, 80, 80), 1)
    draw_fitted_text(
        panel,
        result_text,
        result_box,
        max_scale=2.35,
        min_scale=0.62,
        color=(0, 0, 0),
        thickness=2,
        font=FONT_SIMPLE
    )

    cv2.line(panel, (230, 18), (230, 150), (120, 120, 120), 1)

    # Coloana status.
    draw_text(panel, "STATUS", (255, 30), 0.52, COL_TEXT_DARK, 2, shadow=False, font=FONT_SIMPLE)
    draw_fitted_text(
        panel,
        status_text,
        (255, 48, 245, 60),
        max_scale=0.78,
        min_scale=0.42,
        color=(0, 0, 0),
        thickness=2,
        font=FONT_SIMPLE
    )

    cv2.line(panel, (520, 18), (520, 150), (120, 120, 120), 1)

    # Coloana informatii.
    draw_text(panel, f"CALIBRARE: {calib_text}", (545, 34), 0.54, COL_TEXT_DARK, 1, shadow=False, font=FONT_SIMPLE)
    draw_text(panel, f"CIFRE DETECTATE: {components_count}", (545, 66), 0.54, COL_TEXT_DARK, 1, shadow=False, font=FONT_SIMPLE)
    draw_text(panel, f"PRAGURI: DIFF={diff_thr}   MIN={min_component_area}   MAX={int(max_component_area_ratio * 100)}%", (545, 98), 0.44, COL_TEXT_DARK, 1, shadow=False, font=FONT_SIMPLE)

    # Comenzi pe doua randuri, ca textul TRIMITERE sa nu fie acoperit de butonul D.
    # Toate etichetele sunt desenate cu negru real.
    draw_text(panel, "COMENZI:", (545, 132), 0.46, (0, 0, 0), 2, shadow=False, font=FONT_SIMPLE)

    y1 = 112
    x = 640
    x = draw_label_box(panel, "B", x, y1, 0.48, COL_TEXT_LIGHT, (35, 110, 220)) + 7
    draw_text(panel, "CALIB.", (x, y1 + 20), 0.39, (0, 0, 0), 1, shadow=False, font=FONT_SIMPLE)

    x = 725
    x = draw_label_box(panel, "C", x, y1, 0.48, COL_TEXT_LIGHT, (45, 165, 60)) + 7
    draw_text(panel, "TRIMITERE", (x, y1 + 20), 0.37, (0, 0, 0), 1, shadow=False, font=FONT_SIMPLE)

    y2 = 142
    x = 640
    x = draw_label_box(panel, "D", x, y2, 0.48, COL_TEXT_LIGHT, (150, 70, 190)) + 7
    draw_text(panel, "DEBUG", (x, y2 + 20), 0.39, (0, 0, 0), 1, shadow=False, font=FONT_SIMPLE)

    return panel


# Construiește panoul cu cifrele normalizate 28x28 și predicțiile primite.
def make_digits_preview(components, last_results):
    img = np.full((175, 580, 3), COL_SURFACE, dtype=np.uint8)

    if len(components) == 0:
        draw_text(img, "Nu sunt cifre detectate", (35, 78), 0.72, COL_TEXT_DARK, 2, shadow=False, font=FONT_SIMPLE)
        draw_text(img, "Scrie in chenar si apasa C", (35, 120), 0.52, COL_TEXT_DARK, 2, shadow=False, font=FONT_SIMPLE)
        return img

    x = 18
    y = 18
    step = 112

    for i, comp in enumerate(components[:5]):
        digit = comp["digit28"]

        big = cv2.resize(digit, (86, 86), interpolation=cv2.INTER_NEAREST)
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(img, (x - 4, y - 4), (x + 90, y + 90), (255, 255, 255), -1)
        img[y:y + 86, x:x + 86] = big
        cv2.rectangle(img, (x - 4, y - 4), (x + 90, y + 90), (0, 0, 0), 1)

        pred_text = "-"
        pred_color = COL_TEXT_DARK

        if len(last_results) > i:
            if last_results[i]["safe"]:
                pred_text = str(last_results[i]["pred"])
                pred_color = COL_TEXT_DARK
            else:
                pred_text = "ERR"
                pred_color = COL_TEXT_DARK

        draw_text(img, f"cifra {i}", (x - 3, y + 115), 0.42, COL_TEXT_DARK, 1, shadow=False, font=FONT_SIMPLE)
        draw_text(img, f"FPGA {pred_text}", (x - 3, y + 145), 0.48, pred_color, 2, shadow=False, font=FONT_SIMPLE)

        x += step

    return img


# Creează antetul negru din partea de sus a interfeței.
def make_header():
    header = np.full((48, PAGE_W, 3), COL_HEADER, dtype=np.uint8)

    draw_text(
        header,
        "RECUNOASTERE CIFRE MANUSCRISE CU CNN ACCELERAT PE FPGA",
        (16, 32),
        0.68,
        COL_TEXT_LIGHT,
        2,
        shadow=False,
        font=FONT_SIMPLE
    )

    return header


# Copiază un panou în pagina principală fără să depășească marginile.
def put_panel_safe(page, panel, x, y):
    """Copiaza un panou in pagina fara crash daca panoul depaseste pagina."""
    ph, pw = page.shape[:2]
    h, w = panel.shape[:2]

    if x >= pw or y >= ph:
        return

    copy_w = min(w, pw - x)
    copy_h = min(h, ph - y)

    if copy_w <= 0 or copy_h <= 0:
        return

    page[y:y + copy_h, x:x + copy_w] = panel[:copy_h, :copy_w]


# Asamblează toată pagina principală din panourile camerei, ROI-ului, controlului și cifrelor.
def build_page(
    display,
    roi_vis,
    components,
    last_number,
    last_status,
    last_results,
    diff_thr,
    min_component_area,
    max_component_area_ratio
):
    page = np.full((PAGE_H, PAGE_W, 3), COL_BG, dtype=np.uint8)
    page[0:48, 0:PAGE_W] = make_header()

    camera_panel = make_panel(display, "CAMERA LIVE", 900, 500, title_h=34, bg_color=(0, 0, 0))
    roi_panel = make_panel(roi_vis, "ROI CU DETECTII SI PREDICTII", 585, 500, title_h=34, bg_color=(0, 0, 0))

    control_panel = make_panel(
        make_control_result_panel(
            last_number,
            last_status,
            len(components),
            diff_thr,
            min_component_area,
            max_component_area_ratio
        ),
        "CONTROL SI REZULTAT FPGA",
        900,
        215,
        title_h=34,
        bg_color=COL_SURFACE
    )

    digits_panel = make_panel(
        make_digits_preview(components, last_results),
        "CIFRE 28X28 NORMALIZATE",
        585,
        215,
        title_h=34,
        bg_color=COL_SURFACE
    )

    put_panel_safe(page, camera_panel, 5, 54)
    put_panel_safe(page, roi_panel, 910, 54)
    put_panel_safe(page, control_panel, 5, 560)
    put_panel_safe(page, digits_panel, 910, 560)

    return page


# Construiește fereastra de debug cu imaginile intermediare de preprocesare.
def make_debug_grid(thresh_base, thresh_raw, thresh_clean, gray=None):
    debug_w = 1200
    debug_h = 700
    page = np.full((debug_h, debug_w, 3), COL_BG, dtype=np.uint8)

    if gray is None:
        gray = np.zeros_like(thresh_clean)

    p1 = make_panel(gray, "GRI EGALIZAT", 565, 310)
    p2 = make_panel(thresh_base, "THRESHOLD BAZA", 565, 310)
    p3 = make_panel(thresh_raw, "THRESHOLD FOLOSIT", 565, 310)
    p4 = make_panel(thresh_clean, "THRESHOLD CURATAT", 565, 310)

    page[20:330, 25:590] = p1
    page[20:330, 610:1175] = p2
    page[350:660, 25:590] = p3
    page[350:660, 610:1175] = p4

    return page


# Creează fereastra principală și trackbar-urile pentru reglarea pragurilor.
def setup_controls():
    cv2.namedWindow(PAGE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PAGE_WINDOW, PAGE_W, PAGE_H)
    cv2.moveWindow(PAGE_WINDOW, 8, 20)

    cv2.createTrackbar("DIFF_THR", PAGE_WINDOW, 25, 100, nothing)
    cv2.createTrackbar("MIN_AREA", PAGE_WINDOW, MIN_COMPONENT_AREA_DEFAULT, 1000, nothing)
    cv2.createTrackbar("MAX_AREA_%", PAGE_WINDOW, MAX_COMPONENT_AREA_RATIO_DEFAULT, 50, nothing)


# Desenează chenarul ROI peste imaginea live.
def draw_camera_overlay(display, x1, y1, x2, y2, components, last_number, last_status):
    # Fara text peste imagine, ca pagina sa fie curata.
    # Chenar principal ROI.
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 0), 3)

    # Zona recomandata pentru scris.
    cv2.rectangle(display, (x1 + 30, y1 + 60), (x2 - 30, y2 - 60), COL_ACCENT_2, 2)


# Desenează dreptunghiurile componentelor detectate și predicțiile peste ROI.
def draw_roi_overlay(roi_vis, components, last_results):
    cv2.rectangle(
        roi_vis,
        (30, 60),
        (roi_vis.shape[1] - 30, roi_vis.shape[0] - 60),
        COL_ACCENT_2,
        2
    )

    for idx_comp, comp in enumerate(components):
        x, y, w, h = comp["bbox"]

        cv2.rectangle(roi_vis, (x, y), (x + w, y + h), COL_YELLOW, 2)
        # fara eticheta text pentru fiecare componenta, ca sa nu se incarce imaginea

    if len(last_results) > 0:
        for r in last_results:
            x, y, w, h = r["bbox"]
            color = (0, 220, 0) if r["safe"] else (0, 0, 255)
            label_text = str(r["pred"]) if r["safe"] else "ERR"

            cv2.rectangle(roi_vis, (x, y), (x + w, y + h), color, 3)
            draw_label_box(
                roi_vis,
                label_text,
                x,
                min(roi_vis.shape[0] - 10, y + h + 28),
                0.90,
                color,
                (0, 0, 0)
            )


# Funcția principală: pornește camera, procesează cadrele și gestionează tastele.
def main():
    print("=== APLICATIE FPGA UI V12 CLAR - FISIER CORECT ===")
    global BACKGROUND_CROP, SHOW_DEBUG_WINDOWS

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    # Deschide camera folosind backend-ul DirectShow pe Windows.

    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Nu pot deschide camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    setup_controls()

    last_number = ""
    last_status = ""
    last_results = []
    idx = 0

    print("Comenzi aplicatie:")
    print("  B = calibrare fundal")
    print("  C = trimite cifrele detectate la FPGA")
    print("  S = salveaza debug pentru ultima detectie FPGA")
    print("  D = afiseaza/ascunde fereastra separata de debug")
    print("  Q sau ESC = iesire")
    print()
    print("Important: tastele merg dupa ce dai click pe fereastra OpenCV.")
    print(f"Camera index: {CAMERA_INDEX}")
    print(f"FPGA: {BOARD_IP}:{BOARD_PORT}")
    print()

    current_roi = None
    current_thresh_clean = None
    current_components = []
    current_gray = None
    current_thresh_base = None
    current_thresh_raw = None

    while True:
        ret, frame = cap.read()
        # Citește un cadru nou de la cameră.

        if not ret:
            print("Nu pot citi frame-ul")
            break

        if MIRROR_CAMERA:
        # Oglindește imaginea dacă această opțiune este activată.
            frame = cv2.flip(frame, 1)

        diff_thr, min_component_area, max_component_area_ratio = get_control_values()
        # Ia pragurile curente din trackbar-uri.

        x1, y1, x2, y2 = get_center_roi(frame)
        # Calculează zona centrală în care se caută cifrele.
        roi = frame[y1:y2, x1:x2].copy()
        # Extrage ROI-ul din imaginea live.
        current_roi = roi.copy()

        gray, thresh_base, thresh_raw, thresh_clean, components = preprocess_multi_digits(
        # Aplică preprocesarea și detectează cifrele din ROI.
            roi,
            BACKGROUND_CROP,
            diff_thr,
            min_component_area,
            max_component_area_ratio
        )

        current_gray = gray
        current_thresh_base = thresh_base
        current_thresh_raw = thresh_raw
        current_thresh_clean = thresh_clean
        current_components = components

        display = frame.copy()
        draw_camera_overlay(display, x1, y1, x2, y2, components, last_number, last_status)

        roi_vis = roi.copy()
        draw_roi_overlay(roi_vis, components, last_results)

        page = build_page(
            display,
            roi_vis,
            components,
            last_number,
            last_status,
            last_results,
            diff_thr,
            min_component_area,
            max_component_area_ratio
        )

        cv2.imshow(PAGE_WINDOW, page)
        # Afișează pagina principală.

        if SHOW_DEBUG_WINDOWS:
            debug_page = make_debug_grid(thresh_base, thresh_raw, thresh_clean, gray)
            cv2.imshow("Debug preprocesare - threshold", debug_page)
        else:
            try:
                cv2.destroyWindow("Debug preprocesare - threshold")
            except cv2.error:
                pass

        key = cv2.waitKeyEx(30)
        # Așteaptă o tastă și menține fereastra actualizată.

        if key != -1:
            print("Tasta apasata, cod =", key)

        if key in [ord("q"), ord("Q"), 27]:
        # Q sau ESC închide aplicația.
            print("[INFO] Iesire din aplicatie.")
            break

        elif key in [ord("d"), ord("D")]:
        # D afișează sau ascunde fereastra de debug.
            SHOW_DEBUG_WINDOWS = not SHOW_DEBUG_WINDOWS
            if SHOW_DEBUG_WINDOWS:
                print("[INFO] Debug preprocesare afisat separat.")
            else:
                print("[INFO] Debug preprocesare ascuns.")

        elif key in [ord("b"), ord("B")]:
        # B salvează fundalul curent pentru calibrare.
            _, crop_for_background, _ = prepare_gray_crop(roi)
            BACKGROUND_CROP = crop_for_background.copy()
            last_status = "Fundal calibrat"
            last_number = ""
            last_results = []
            print("[OK] Fundal calibrat. Acum arata cifrele si apasa C.")

        elif key in [ord("c"), ord("C")]:
        # C trimite cifrele detectate către FPGA.
            if len(current_components) == 0:
                print("[INFO] Nu am detectat nicio cifra.")
                last_number = ""
                last_status = "Nu vad cifre"
                last_results = []
                continue

            print()
            print("======================================")
            print("       TRIMIT CIFRELE LA FPGA         ")
            print("======================================")
            print(f"Numar componente: {len(current_components)}")
            print()

            results = predict_components_fpga(current_components)
            # Trimite toate cifrele detectate către FPGA.

            number_text = ""
            all_safe = True

            for i, r in enumerate(results):
                if r["safe"]:
                    number_text += str(r["pred"])
                    # Adaugă predicția la numărul final afișat.
                    print(f"Cifra {i}: FPGA pred={r['pred']} | OK")
                else:
                    number_text += "?"
                    all_safe = False
                    print(f"Cifra {i}: EROARE FPGA/TCP | {r['error']}")

            print("--------------------------------------")
            print(f"Numar detectat FPGA: {number_text}")

            if all_safe:
                last_status = "OK"
                print("Status: OK")
            else:
                last_status = "Eroare FPGA"
                print("Status: Eroare FPGA")

            print("======================================")
            print()

            last_number = number_text
            last_results = results
            idx += 1

        elif key in [ord("s"), ord("S")]:
        # S salvează datele de debug pentru ultima detecție.
            if current_roi is None or len(last_results) == 0:
                print("[INFO] Nu am ce salva. Apasa intai C.")
                continue

            save_detection_debug(current_roi, current_thresh_clean, last_results, idx)
            # Salvează imaginile și vectorii ultimei rulări.
            idx += 1

    cap.release()
    # Eliberează camera la închiderea aplicației.
    cv2.destroyAllWindows()
    # Închide toate ferestrele OpenCV.


if __name__ == "__main__":
    main()
