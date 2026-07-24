#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include <sys/mman.h>
#include <sys/socket.h>

#include <arpa/inet.h>
#include <netinet/in.h>

// Setări pentru serverul TCP care primește datele de la aplicația Python

#define SERVER_PORT 12345
// Portul trebuie să fie același cu cel folosit în aplicația Python.

// Dimensiunile imaginii de intrare și ale vectorului de ieșire CNN

#define INPUT_H      28
#define INPUT_W      28
#define INPUT_SIZE   (INPUT_H * INPUT_W)   /* 784 */
// Imaginea 28x28 are 784 pixeli.
#define OUTPUT_SIZE  10
// Rețeaua returnează 10 scoruri, câte unul pentru fiecare cifră 0-9.

/*
 * Python trimite 784 bytes:
 *   uint8 raw ap_fixed<8,4>
 *
 * Acceleratorul are AXI Stream pe 32 biti:
 *   fiecare pixel trebuie pus intr-un uint32_t
 *
 * Deci:
 *   TCP input  = 784 bytes
 *   DMA input  = 784 * 4 = 3136 bytes
 *   DMA output = 10 * 4 = 40 bytes
 */

#define INPUT_NET_BYTES   (INPUT_SIZE * sizeof(uint8_t))
// Prin TCP intrarea este primită ca 784 valori pe 8 biți.
#define INPUT_DMA_BYTES   (INPUT_SIZE * sizeof(uint32_t))
// Pentru DMA, fiecare valoare este pusă pe 32 biți.
#define OUTPUT_DMA_BYTES  (OUTPUT_SIZE * sizeof(uint32_t))
// Ieșirea DMA conține 10 cuvinte de 32 biți.

// Adresele fizice ale perifericelor hardware mapate în Vivado

#define DMA_BASE_PHYS   0x41E00000
// Adresa de bază a registrelor AXI DMA.
#define HLS_BASE_PHYS   0x40000000
// Adresa de bază a registrului de control al acceleratorului CNN.

#define REG_MAP_SIZE    0x10000

// Adresele bufferelor din DDR folosite de DMA pentru intrare și ieșire

#define IN_BUF_PHYS     0x1E000000
// Bufferul din DDR din care DMA citește imaginea.
#define OUT_BUF_PHYS    0x1E001000
// Bufferul din DDR în care DMA scrie scorurile.

// Dimensiunea mapată pentru fiecare buffer este suficientă pentru intrare și ieșire

#define BUF_MAP_SIZE    0x1000

// Offset-urile registrelor AXI DMA folosite în modul simplu

/* MM2S: Memory Map to Stream */
#define MM2S_DMACR      0x00
#define MM2S_DMASR      0x04
#define MM2S_SA         0x18
#define MM2S_LENGTH     0x28

/* S2MM: Stream to Memory Map */
#define S2MM_DMACR      0x30
#define S2MM_DMASR      0x34
#define S2MM_DA         0x48
#define S2MM_LENGTH     0x58

// Biții de control ai registrelor DMACR
#define DMACR_RS        (1u << 0)
#define DMACR_RESET     (1u << 2)

// Biții de stare ai registrelor DMASR
#define DMASR_HALTED        (1u << 0)
#define DMASR_IDLE          (1u << 1)
#define DMASR_INTERNAL_ERR  (1u << 4)
#define DMASR_SLAVE_ERR     (1u << 5)
#define DMASR_DECODE_ERR    (1u << 6)
#define DMASR_IOC_IRQ       (1u << 12)
#define DMASR_DLY_IRQ       (1u << 13)
#define DMASR_ERR_IRQ       (1u << 14)

// Masca folosită pentru curățarea biților de întrerupere DMA
#define DMASR_IRQ_ALL       (DMASR_IOC_IRQ | DMASR_DLY_IRQ | DMASR_ERR_IRQ)

// Registrul de control AXI-Lite al acceleratorului HLS

#define HLS_CTRL        0x00

#define HLS_AP_START    (1u << 0)
#define HLS_AP_DONE     (1u << 1)
#define HLS_AP_IDLE     (1u << 2)
#define HLS_AP_READY    (1u << 3)

// Pointeri către registrele hardware și bufferele mapate cu mmap

static volatile uint32_t *dma_regs = NULL;
// Pointer către registrele AXI DMA.
static volatile uint32_t *hls_regs = NULL;
// Pointer către registrele AXI-Lite ale acceleratorului.

static volatile uint32_t *in_buf = NULL;
// Pointer către bufferul de intrare din DDR.
static volatile uint32_t *out_buf = NULL;
// Pointer către bufferul de ieșire din DDR.

// Bufferul în care sunt primiți cei 784 bytes trimiși de aplicația Python
static uint8_t net_input[INPUT_NET_BYTES];
// Aici se salvează imaginea primită de la aplicația Python.

// Funcții scurte pentru scrierea și citirea registrelor hardware

// Scrie o valoare într-un registru hardware.
static inline void reg_write(volatile uint32_t *base, uint32_t offset, uint32_t value)
{
    base[offset / 4] = value;
    // Offsetul este în bytes, iar accesul se face pe cuvinte de 32 biți.
}

// Citește o valoare dintr-un registru hardware.
static inline uint32_t reg_read(volatile uint32_t *base, uint32_t offset)
{
    return base[offset / 4];
    // Citește registrul corespunzător offsetului cerut.
}

// Funcții care citesc sau trimit exact numărul de bytes cerut

// Citește din socket până când primește exact numărul de bytes cerut.
static int recv_exact(int sock, void *buf, size_t len)
{
    uint8_t *p = (uint8_t *)buf;
    // Pointer folosit pentru a umple bufferul byte cu byte.
    size_t received = 0;

    while (received < len) {
        ssize_t r = recv(sock, p + received, len - received, 0);
        // recv poate primi mai puțini bytes, deci se repetă până se primește tot.

        if (r == 0) {
            fprintf(stderr, "Clientul a inchis conexiunea prematur.\n");
            return -1;
        }

        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }

            perror("recv");
            return -1;
        }

        received += (size_t)r;
    }

    return 0;
}

// Trimite prin socket până când sunt trimiși toți bytes ceruți.
static int send_exact(int sock, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t sent = 0;

    while (sent < len) {
        ssize_t r = send(sock, p + sent, len - sent, 0);
        // send poate trimite parțial, deci se repetă până se trimite tot.

        if (r <= 0) {
            if (r < 0 && errno == EINTR) {
                continue;
            }

            perror("send");
            return -1;
        }

        sent += (size_t)r;
    }

    return 0;
}

// Funcții pentru afișarea și verificarea stării DMA

// Afișează în terminal starea unui canal DMA.
static void print_dma_status(const char *name, uint32_t sr)
{
    printf("%s DMASR = 0x%08X", name, sr);

    if (sr & DMASR_HALTED) {
        printf(" [HALTED]");
    }

    if (sr & DMASR_IDLE) {
        printf(" [IDLE]");
    }

    if (sr & DMASR_IOC_IRQ) {
        printf(" [IOC_IRQ]");
    }

    if (sr & DMASR_DLY_IRQ) {
        printf(" [DLY_IRQ]");
    }

    if (sr & DMASR_ERR_IRQ) {
        printf(" [ERR_IRQ]");
    }

    if (sr & DMASR_INTERNAL_ERR) {
        printf(" [INTERNAL_ERR]");
    }

    if (sr & DMASR_SLAVE_ERR) {
        printf(" [SLAVE_ERR]");
    }

    if (sr & DMASR_DECODE_ERR) {
        printf(" [DECODE_ERR]");
    }

    printf("\n");
}

// Verifică dacă registrul de stare DMA conține erori.
static int dma_has_error(uint32_t sr)
{
    return (sr & (DMASR_INTERNAL_ERR | DMASR_SLAVE_ERR | DMASR_DECODE_ERR | DMASR_ERR_IRQ)) != 0;
    // Returnează 1 dacă există orice eroare DMA importantă.
}

// Resetarea modulului AXI DMA înainte de o inferență

// Resetează canalele DMA și așteaptă terminarea resetării.
static int dma_reset(void)
{
    reg_write(dma_regs, MM2S_DMACR, DMACR_RESET);
    // Resetează canalul care trimite datele către accelerator.
    reg_write(dma_regs, S2MM_DMACR, DMACR_RESET);
    // Resetează canalul care primește rezultatele de la accelerator.

    int timeout = 100000;

    while (timeout > 0) {
        uint32_t mm2s_cr = reg_read(dma_regs, MM2S_DMACR);
        // Citește starea resetului pentru canalul MM2S.
        uint32_t s2mm_cr = reg_read(dma_regs, S2MM_DMACR);
        // Citește starea resetului pentru canalul S2MM.

        if (((mm2s_cr & DMACR_RESET) == 0) &&
            ((s2mm_cr & DMACR_RESET) == 0)) {
            return 0;
        }

        timeout--;
        usleep(10);
    }

    fprintf(stderr, "ERROR: DMA reset timeout\n");
    return -1;
}

// Pornirea canalelor DMA MM2S și S2MM

// Pornește canalele DMA și verifică dacă au apărut erori.
static int dma_start_channels(void)
{
    /*
     * Pornim canalele DMA.
     */
    reg_write(dma_regs, MM2S_DMACR, DMACR_RS);
    // Pune canalul MM2S în modul running.
    reg_write(dma_regs, S2MM_DMACR, DMACR_RS);
    // Pune canalul S2MM în modul running.

    /*
     * Stergem eventuale interrupt-uri vechi.
     */
    reg_write(dma_regs, MM2S_DMASR, DMASR_IRQ_ALL);
    reg_write(dma_regs, S2MM_DMASR, DMASR_IRQ_ALL);

    usleep(100);

    uint32_t mm2s_sr = reg_read(dma_regs, MM2S_DMASR);
    uint32_t s2mm_sr = reg_read(dma_regs, S2MM_DMASR);

    if (dma_has_error(mm2s_sr)) {
        fprintf(stderr, "ERROR: MM2S error after start\n");
        print_dma_status("MM2S", mm2s_sr);
        return -1;
    }

    if (dma_has_error(s2mm_sr)) {
        fprintf(stderr, "ERROR: S2MM error after start\n");
        print_dma_status("S2MM", s2mm_sr);
        return -1;
    }

    return 0;
}

// Funcții care așteaptă terminarea transferurilor DMA

// Așteaptă finalizarea transferului MM2S, adică DDR către accelerator.
static int dma_wait_mm2s_done(void)
{
    int timeout = 10000000;

    while (timeout > 0) {
        uint32_t sr = reg_read(dma_regs, MM2S_DMASR);
        // Citește starea canalului MM2S.

        if (dma_has_error(sr)) {
            fprintf(stderr, "ERROR: MM2S DMA error\n");
            print_dma_status("MM2S", sr);
            return -1;
        }

        /*
         * Pentru transfer simplu, IOC_IRQ sau IDLE indica finalizare.
         */
        if ((sr & DMASR_IOC_IRQ) || (sr & DMASR_IDLE)) {
            reg_write(dma_regs, MM2S_DMASR, DMASR_IRQ_ALL);
            return 0;
        }

        timeout--;
    }

    fprintf(stderr, "ERROR: MM2S DMA timeout\n");
    print_dma_status("MM2S", reg_read(dma_regs, MM2S_DMASR));
    return -1;
}

// Așteaptă finalizarea transferului S2MM, adică accelerator către DDR.
static int dma_wait_s2mm_done(void)
{
    int timeout = 10000000;

    while (timeout > 0) {
        uint32_t sr = reg_read(dma_regs, S2MM_DMASR);
        // Citește starea canalului S2MM.

        if (dma_has_error(sr)) {
            fprintf(stderr, "ERROR: S2MM DMA error\n");
            print_dma_status("S2MM", sr);
            return -1;
        }

        /*
         * Pentru S2MM, IOC_IRQ apare cand s-a primit TLAST corect.
         * IDLE poate fi folosit ca semn de finalizare, dar IOC_IRQ este mai sigur.
         */
        if ((sr & DMASR_IOC_IRQ) || (sr & DMASR_IDLE)) {
            reg_write(dma_regs, S2MM_DMASR, DMASR_IRQ_ALL);
            return 0;
        }

        timeout--;
    }

    fprintf(stderr, "ERROR: S2MM DMA timeout\n");
    print_dma_status("S2MM", reg_read(dma_regs, S2MM_DMASR));
    return -1;
}

// Așteptarea finalizării acceleratorului HLS

// Așteaptă ca acceleratorul HLS să termine inferența.
static int hls_wait_done(void)
{
    int timeout = 10000000;

    while (timeout > 0) {
        uint32_t ctrl = reg_read(hls_regs, HLS_CTRL);
        // Citește registrul de control al acceleratorului HLS.

        if ((ctrl & HLS_AP_DONE) || (ctrl & HLS_AP_IDLE)) {
            return 0;
        }

        timeout--;
    }

    fprintf(stderr, "ERROR: HLS timeout, CTRL = 0x%08X\n",
            reg_read(hls_regs, HLS_CTRL));

    return -1;
}

// Conversie pentru afișarea scorurilor out_t = ap_fixed<11,6>

// Transformă un scor raw ap_fixed<11,6> în float pentru afișare.
static float fixed11_6_raw_to_float(uint32_t raw_word)
{
    int32_t raw = (int32_t)raw_word;
    return ((float)raw) / 32.0f;
    // Formatul ap_fixed<11,6> are 5 biți fracționari, deci factorul este 32.
}

// Pentru argmax se compară direct valorile raw, deoarece toate au același factor de scalare.
// Găsește indexul scorului maxim din vectorul de ieșire.
static int argmax_fixed_output(const volatile uint32_t *arr, int size)
{
    int idx = 0;
    int32_t max_val = (int32_t)arr[0];
    // Primul scor este considerat inițial maxim.

    for (int i = 1; i < size; i++) {
        int32_t val = (int32_t)arr[i];
        // Scorul este interpretat ca valoare semnată.

        if (val > max_val) {
            max_val = val;
            idx = i;
            // Salvează clasa care are scorul cel mai mare.
        }
    }

    return idx;
}

// Afișează toate cele 10 scoruri returnate de accelerator.
static void print_outputs(void)
{
    printf("Output raw / scaled x1000:\n");

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        int32_t raw = (int32_t)out_buf[i];
        float val = fixed11_6_raw_to_float(out_buf[i]);
        // Conversia este folosită doar pentru afișare, nu pentru decizie.
        int scaled = (int)(val * 1000.0f);

        printf("  %d: raw=%d, scaled=%d\n", i, raw, scaled);
    }
}

// Pregătirea bufferului de intrare pentru DMA

// Copiază cei 784 bytes primiți prin TCP în bufferul DMA pe 32 biți.
static void expand_net_input_to_dma_buffer(void)
{
    for (int i = 0; i < INPUT_SIZE; i++) {
        in_buf[i] = (uint32_t)net_input[i];
        // Valoarea utilă rămâne în cei 8 biți inferiori ai cuvântului de 32 biți.
    }
}

// Secvența completă pentru rularea unei inferențe pe acceleratorul FPGA

// Configurează DMA, pornește acceleratorul și așteaptă rezultatul.
static int run_fpga_inference(void)
{
    /*
     * Curatam output-ul.
     */
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        out_buf[i] = 0;
        // Curăță scorul vechi din bufferul de ieșire.
    }

    /*
     * Reset DMA pentru fiecare inferenta.
     * E mai robust pentru testare.
     */
    if (dma_reset() != 0) {
        return -1;
    }

    if (dma_start_channels() != 0) {
        return -1;
    }

    /*
     * Programare adrese fizice DMA.
     */
    reg_write(dma_regs, S2MM_DA, OUT_BUF_PHYS);
    // Spune canalului S2MM unde să scrie scorurile.
    reg_write(dma_regs, MM2S_SA, IN_BUF_PHYS);
    // Spune canalului MM2S de unde să citească imaginea.

    /*
     * Ordine recomandata:
     *
     * 1. Pornim S2MM, ca sa fie pregatita receptia output-ului.
     * 2. Pornim IP-ul HLS.
     * 3. Pornim MM2S, adica trimitem input-ul.
     */

    reg_write(dma_regs, S2MM_LENGTH, OUTPUT_DMA_BYTES);
    // Pornește recepția celor 10 scoruri.

    reg_write(hls_regs, HLS_CTRL, HLS_AP_START);
    // Pornește acceleratorul CNN.

    reg_write(dma_regs, MM2S_LENGTH, INPUT_DMA_BYTES);
    // Pornește trimiterea imaginii către accelerator.

    if (dma_wait_mm2s_done() != 0) {
        return -1;
    }

    if (dma_wait_s2mm_done() != 0) {
        return -1;
    }

    if (hls_wait_done() != 0) {
        return -1;
    }

    return 0;
}

// Maparea registrelor și bufferelor fizice în spațiul aplicației Linux

// Mapează registrele DMA, registrele HLS și bufferele DDR cu /dev/mem.
static int setup_mmaps(void)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    // /dev/mem permite accesul la adrese fizice din spațiul user.

    if (fd < 0) {
        perror("open /dev/mem");
        return -1;
    }

    dma_regs = (volatile uint32_t *)mmap(
    // Mapează registrele AXI DMA.
        NULL,
        REG_MAP_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        DMA_BASE_PHYS
    );

    if (dma_regs == MAP_FAILED) {
        perror("mmap dma_regs");
        close(fd);
        return -1;
    }

    hls_regs = (volatile uint32_t *)mmap(
    // Mapează registrele de control ale acceleratorului.
        NULL,
        REG_MAP_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        HLS_BASE_PHYS
    );

    if (hls_regs == MAP_FAILED) {
        perror("mmap hls_regs");
        close(fd);
        return -1;
    }

    in_buf = (volatile uint32_t *)mmap(
    // Mapează bufferul de intrare din DDR.
        NULL,
        BUF_MAP_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        IN_BUF_PHYS
    );

    if (in_buf == MAP_FAILED) {
        perror("mmap in_buf");
        close(fd);
        return -1;
    }

    out_buf = (volatile uint32_t *)mmap(
    // Mapează bufferul de ieșire din DDR.
        NULL,
        BUF_MAP_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        OUT_BUF_PHYS
    );

    if (out_buf == MAP_FAILED) {
        perror("mmap out_buf");
        close(fd);
        return -1;
    }

    close(fd);

    return 0;
}

// Configurarea socketului TCP pe care se conectează aplicația Python

// Creează serverul TCP care ascultă pe portul 12345.
static int setup_server_socket(void)
{
    int server_fd;
    int opt = 1;
    struct sockaddr_in addr;

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    // Creează un socket TCP IPv4.

    if (server_fd < 0) {
        perror("socket");
        return -1;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        close(server_fd);
        // Închide socketul serverului la final.
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    // Inițializează structura adresei cu zero.

    addr.sin_family = AF_INET;
    // Folosește IPv4.
    addr.sin_port = htons(SERVER_PORT);
    // Setează portul serverului în format network byte order.
    addr.sin_addr.s_addr = INADDR_ANY;
    // Acceptă conexiuni pe orice interfață de rețea a plăcii.

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    // Leagă socketul de portul ales.
        perror("bind");
        close(server_fd);
        // Închide socketul serverului la final.
        return -1;
    }

    if (listen(server_fd, 1) < 0) {
    // Pune socketul în modul de așteptare a conexiunilor.
        perror("listen");
        close(server_fd);
        // Închide socketul serverului la final.
        return -1;
    }

    return server_fd;
}

// Funcția principală a serverului embedded

// Pornește serverul, așteaptă clienți și procesează fiecare cifră primită.
int main(void)
{
    int server_fd;

    printf("=====================================\n");
    printf(" FPGA CNN TCP Server - FIXED 8_4\n");
    printf("=====================================\n");

    printf("HLS_BASE_PHYS     = 0x%08X\n", HLS_BASE_PHYS);
    printf("DMA_BASE_PHYS     = 0x%08X\n", DMA_BASE_PHYS);
    printf("IN_BUF_PHYS       = 0x%08X\n", IN_BUF_PHYS);
    printf("OUT_BUF_PHYS      = 0x%08X\n", OUT_BUF_PHYS);
    printf("INPUT_NET_BYTES   = %d\n", INPUT_NET_BYTES);
    printf("INPUT_DMA_BYTES   = %d\n", INPUT_DMA_BYTES);
    printf("OUTPUT_DMA_BYTES  = %d\n", OUTPUT_DMA_BYTES);
    printf("SERVER_PORT       = %d\n", SERVER_PORT);
    printf("=====================================\n");

    if (setup_mmaps() != 0) {
        fprintf(stderr, "ERROR: setup_mmaps failed\n");
        return 1;
    }

    server_fd = setup_server_socket();

    if (server_fd < 0) {
        fprintf(stderr, "ERROR: setup_server_socket failed\n");
        return 1;
    }

    printf("Server pornit. Astept client pe portul %d...\n", SERVER_PORT);

    while (1) {
        int client_fd;

        client_fd = accept(server_fd, NULL, NULL);
        // Așteaptă conectarea aplicației Python.

        if (client_fd < 0) {
            perror("accept");
            continue;
        }

        printf("\nClient conectat.\n");

        /*
         * Primim exact 784 bytes.
         * Daca Python trimite inca float32, aici se va bloca sau datele vor fi gresite.
         */
        if (recv_exact(client_fd, net_input, INPUT_NET_BYTES) != 0) {
        // Citește imaginea primită de la PC.
            fprintf(stderr, "ERROR: recv_exact failed\n");
            close(client_fd);
            // Închide conexiunea cu clientul curent.
            continue;
        }

        printf("Am primit %d bytes de la client.\n", INPUT_NET_BYTES);

        /*
         * uint8[784] -> uint32[784]
         */
        expand_net_input_to_dma_buffer();

        /*
         * Rulam acceleratorul.
         */
        if (run_fpga_inference() != 0) {
        // Rulează inferența hardware pentru imaginea primită.
            fprintf(stderr, "ERROR: run_fpga_inference failed\n");

            print_dma_status("MM2S", reg_read(dma_regs, MM2S_DMASR));
            print_dma_status("S2MM", reg_read(dma_regs, S2MM_DMASR));
            fprintf(stderr, "HLS CTRL = 0x%08X\n", reg_read(hls_regs, HLS_CTRL));

            close(client_fd);
            // Închide conexiunea cu clientul curent.
            continue;
        }

        /*
         * Afisare debug.
         */
        print_outputs();

        /*
         * Calculam predicția.
         */
        int32_t pred = (int32_t)argmax_fixed_output(out_buf, OUTPUT_SIZE);
        // Predicția este indexul scorului maxim.

        /*
         * Trimitem predicția inapoi la Python:
         * 4 bytes, int32 little-endian.
         */
        if (send_exact(client_fd, &pred, sizeof(pred)) != 0) {
        // Trimite predicția înapoi către aplicația Python.
            fprintf(stderr, "ERROR: send_exact failed\n");
            close(client_fd);
            // Închide conexiunea cu clientul curent.
            continue;
        }

        printf("Prediction sent: %d\n", pred);

        close(client_fd);
        // Închide conexiunea cu clientul curent.
    }

    close(server_fd);
    // Închide socketul serverului la final.

    return 0;
}
