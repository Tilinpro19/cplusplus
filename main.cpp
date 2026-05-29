// ============================================================================
//  HPC parallel sort of 20M floats from datos.txt
//  - OpenMP task-based parallel merge sort (zero std::thread creation)
//  - Per-thread chunked parsing with a locale-free SWAR-style float parser
//  - Single ping-pong scratch buffer (no per-call allocations)
//
//  Build:
//      GCC  : g++ -O3 -march=native -std=c++20 -fopenmp main.cpp -o main
//      Clang: clang++ -O3 -march=native -std=c++20 -fopenmp main.cpp -o main
//      MSVC : cl /O2 /std:c++20 /openmp:llvm /EHsc main.cpp
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <omp.h>

using namespace std;

// ---------------------------------------------------------------------------
// Sort tuning
// ---------------------------------------------------------------------------
static constexpr size_t INS_CUTOFF = 64;        // insertion sort threshold
static constexpr size_t PAR_MIN    = 32'768;    // min range to spawn an OMP task

// ---------------------------------------------------------------------------
// Insertion sort over dst[lo..hi]  (inclusive)
// ---------------------------------------------------------------------------
static inline void insertionSort(float* __restrict a, size_t lo, size_t hi) noexcept {
    for (size_t i = lo + 1; i <= hi; ++i) {
        float key = a[i];
        size_t j = i;
        while (j > lo && a[j - 1] > key) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = key;
    }
}

// ---------------------------------------------------------------------------
// Merge two sorted halves of src into dst, range [lo,hi] inclusive.
//   left  = src[lo  ..mid]
//   right = src[mid+1..hi]
// Restrict qualifiers let the compiler vectorize the streaming loops.
// ---------------------------------------------------------------------------
static inline void mergeRanges(const float* __restrict src,
                               float* __restrict dst,
                               size_t lo, size_t mid, size_t hi) noexcept {
    size_t i = lo, j = mid + 1, k = lo;
    while (i <= mid && j <= hi) {
        if (src[i] <= src[j]) dst[k++] = src[i++];
        else                  dst[k++] = src[j++];
    }
    while (i <= mid) dst[k++] = src[i++];
    while (j <= hi)  dst[k++] = src[j++];
}

// ---------------------------------------------------------------------------
// Ping-pong merge sort (Sedgewick scheme).
// Result of mergeSort(src,dst,lo,hi) is written to dst[lo..hi].
// Children are called with swapped roles, so no copy-back is ever needed.
// Concurrency: OpenMP tasks running on a persistent worker pool.
// ---------------------------------------------------------------------------
static void mergeSort(float* src, float* dst, size_t lo, size_t hi) noexcept {
    if (hi <= lo) return;

    if (hi - lo < INS_CUTOFF) {
        insertionSort(dst, lo, hi);
        return;
    }

    const size_t mid = lo + (hi - lo) / 2;

    if (hi - lo > PAR_MIN) {
        #pragma omp task default(none) firstprivate(src, dst, lo, mid)
        mergeSort(dst, src, lo, mid);

        mergeSort(dst, src, mid + 1, hi);

        #pragma omp taskwait
    } else {
        mergeSort(dst, src, lo, mid);
        mergeSort(dst, src, mid + 1, hi);
    }

    // Sorted halves now live in src; merge them into dst.
    if (src[mid] <= src[mid + 1]) {                // already in order
        memcpy(dst + lo, src + lo, (hi - lo + 1) * sizeof(float));
    } else {
        mergeRanges(src, dst, lo, mid, hi);
    }
}

// ---------------------------------------------------------------------------
// Verification: kept functional, parallelized with a reduction.
// ---------------------------------------------------------------------------
bool validar_orden(const vector<float>& vec) noexcept {
    if (vec.size() < 2) return true;
    const float* a = vec.data();
    const ptrdiff_t n = static_cast<ptrdiff_t>(vec.size());
    int bad = 0;
    #pragma omp parallel for reduction(+:bad) schedule(static)
    for (ptrdiff_t i = 1; i < n; ++i) {
        if (a[i] < a[i - 1]) bad = 1;
    }
    return bad == 0;
}

// ---------------------------------------------------------------------------
// Ultra-fast, locale-free float parser. No exceptions, no allocations.
// Handles optional sign, decimal point, and exponent (e/E +/- digits).
// ---------------------------------------------------------------------------
static const double kPow10[] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
    1e10,1e11,1e12,1e13,1e14,1e15,1e16,1e17,1e18,1e19,
    1e20,1e21,1e22
};

static inline double pow10fast(int e) noexcept {
    if (e >= 0) {
        if (e <= 22) return kPow10[e];
        return std::pow(10.0, e);
    } else {
        int ne = -e;
        if (ne <= 22) return 1.0 / kPow10[ne];
        return std::pow(10.0, e);
    }
}

static inline bool isWs(char c) noexcept {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

static inline float parseFloat(const char*& p, const char* end) noexcept {
    bool neg = false;
    if (p < end && (*p == '-' || *p == '+')) { neg = (*p == '-'); ++p; }

    uint64_t mantissa = 0;
    int      digits   = 0;
    while (p < end) {
        unsigned c = static_cast<unsigned>(*p) - '0';
        if (c > 9) break;
        mantissa = mantissa * 10 + c;
        ++p; ++digits;
    }

    int fracDigits = 0;
    if (p < end && *p == '.') {
        ++p;
        while (p < end) {
            unsigned c = static_cast<unsigned>(*p) - '0';
            if (c > 9) break;
            mantissa = mantissa * 10 + c;
            ++p; ++fracDigits;
        }
    }

    int exp = 0;
    if (p < end && (*p == 'e' || *p == 'E')) {
        ++p;
        bool eneg = false;
        if (p < end && (*p == '-' || *p == '+')) { eneg = (*p == '-'); ++p; }
        while (p < end) {
            unsigned c = static_cast<unsigned>(*p) - '0';
            if (c > 9) break;
            exp = exp * 10 + static_cast<int>(c);
            ++p;
        }
        if (eneg) exp = -exp;
    }

    double value = static_cast<double>(mantissa);
    int realExp  = exp - fracDigits;
    if (realExp != 0) value *= pow10fast(realExp);
    return neg ? -static_cast<float>(value) : static_cast<float>(value);
}

// ---------------------------------------------------------------------------
// Parallel file loader.
//   1. Single blocking read of the whole file into a contiguous blob.
//   2. Cut the blob into N chunks aligned to newline boundaries.
//   3. Each OMP thread parses its chunk into a thread-local vector.
//   4. Prefix-sum offsets and parallel memcpy into the final array.
// ---------------------------------------------------------------------------
static bool leerArchivo(const char* ruta, vector<float>& out) {
    ifstream in(ruta, ios::binary | ios::ate);
    if (!in) return false;

    const streamsize size = in.tellg();
    if (size <= 0) { out.clear(); return true; }
    in.seekg(0, ios::beg);

    vector<char> blob(static_cast<size_t>(size) + 1);
    if (!in.read(blob.data(), size)) return false;
    blob[static_cast<size_t>(size)] = '\0';

    const size_t total = static_cast<size_t>(size);
    const int    T     = std::max(1, omp_get_max_threads());

    // Chunk boundaries aligned to '\n'
    vector<size_t> starts(static_cast<size_t>(T) + 1);
    starts[0] = 0;
    starts[T] = total;
    for (int t = 1; t < T; ++t) {
        size_t pos = total * static_cast<size_t>(t) / static_cast<size_t>(T);
        while (pos < total && blob[pos] != '\n') ++pos;
        if (pos < total) ++pos;
        starts[t] = pos;
    }

    vector<vector<float>> partes(static_cast<size_t>(T));
    const size_t reserveHint = (20'000'000u / static_cast<size_t>(T)) + 4096;

    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        const char* p   = blob.data() + starts[tid];
        const char* end = blob.data() + starts[tid + 1];

        vector<float>& local = partes[tid];
        local.reserve(reserveHint);

        while (p < end) {
            while (p < end && isWs(*p)) ++p;
            if (p >= end) break;
            const char c = *p;
            if (c != '-' && c != '+' && c != '.' && (c < '0' || c > '9')) {
                ++p;
                continue;
            }
            local.push_back(parseFloat(p, end));
        }
    }

    // Prefix-sum the per-thread counts.
    vector<size_t> offsets(static_cast<size_t>(T) + 1, 0);
    for (int t = 0; t < T; ++t) offsets[t + 1] = offsets[t] + partes[t].size();
    const size_t totalCount = offsets[T];

    out.resize(totalCount);

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        if (!partes[t].empty()) {
            memcpy(out.data() + offsets[t],
                   partes[t].data(),
                   partes[t].size() * sizeof(float));
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 8;
    omp_set_num_threads(static_cast<int>(hc));

    vector<float> numbers;

    // --- Load --------------------------------------------------------------
    auto loadStart = chrono::high_resolution_clock::now();
    if (!leerArchivo("datos.txt", numbers)) {
        cout << "ERROR: no se pudo abrir 'datos.txt'\n";
        return 1;
    }
    auto loadEnd = chrono::high_resolution_clock::now();
    auto loadMs  = chrono::duration_cast<chrono::milliseconds>(loadEnd - loadStart);

    cout << "Vector (size): "  << numbers.size() << '\n';
    cout << "Tiempo de carga: " << loadMs.count() << " mili.\n";
    cout << "Hilos: "          << hc << '\n';

    // --- Sort --------------------------------------------------------------
    auto start = chrono::high_resolution_clock::now();

    if (numbers.size() > 1) {
        vector<float> buffer(numbers);                  // single auxiliary buffer

        #pragma omp parallel
        {
            #pragma omp single nowait
            mergeSort(buffer.data(), numbers.data(), 0, numbers.size() - 1);
        }
    }

    auto end      = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Tiempo: " << duration.count() << " mili." << '\n';

    // --- Verify ------------------------------------------------------------
    cout << "Ordenado: " << (validar_orden(numbers) ? "Sí" : "No") << '\n';
    return 0;
}
