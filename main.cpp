// ============================================================================
//  Ultra-fast sort of 20,000,000 floats from datos.txt
//
//  Sort algorithm: Parallel LSD Radix Sort over the IEEE-754 bit pattern.
//    - encode(float)  : bit-pattern transform that preserves total float order
//                       in unsigned 32-bit space (sign-aware bit flip).
//    - 4 LSD passes of 8 bits each, parallelized with OpenMP.
//    - decode back to float in place.
//
//  Build:
//      g++ -O3 -march=native -std=c++20 -fopenmp main.cpp -o main
//      clang++ -O3 -march=native -std=c++20 -fopenmp main.cpp -o main
//      cl /O2 /std:c++20 /openmp:llvm /EHsc main.cpp
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <array>
#include <thread>
#include <algorithm>
#include <omp.h>

using namespace std;

// ---------------------------------------------------------------------------
// IEEE-754 order-preserving bit transform
//   - Positives (sign bit 0)  -> flip only the sign bit (becomes 1xxx..).
//   - Negatives (sign bit 1)  -> flip every bit         (becomes 0xxx..).
// After encode, ascending uint32_t order == ascending float order.
// ---------------------------------------------------------------------------
static inline uint32_t encodeFloat(float f) noexcept {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t mask = (uint32_t)(-(int32_t)(u >> 31)) | 0x80000000u;
    return u ^ mask;
}

static inline float decodeFloat(uint32_t u) noexcept {
    uint32_t mask = ((u >> 31) - 1u) | 0x80000000u;
    uint32_t r = u ^ mask;
    float f;
    std::memcpy(&f, &r, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// Parallel LSD Radix Sort (4 passes of 8 bits) over uint32_t keys.
//   keys : input/output array (sorted in place on return)
//   aux  : scratch buffer of identical size
//   n    : number of elements
// Per-thread histograms eliminate atomics; per-thread offset cursors avoid
// false sharing during the scatter phase.
// ---------------------------------------------------------------------------
static void parallelRadixSortU32(uint32_t* keys, uint32_t* aux, size_t n) noexcept {
    const int T = std::max(1, omp_get_max_threads());

    // Per-thread histograms (T x 256). We reuse them across passes.
    vector<array<size_t, 256>> hist(static_cast<size_t>(T));
    vector<array<size_t, 256>> off (static_cast<size_t>(T));

    uint32_t* src = keys;
    uint32_t* dst = aux;

    for (int pass = 0; pass < 4; ++pass) {
        const int shift = pass * 8;

        // ---- Reset histograms -----------------------------------------
        for (int t = 0; t < T; ++t) hist[t].fill(0);

        // ---- Pass 1: per-thread histogram of bucket counts ------------
        #pragma omp parallel num_threads(T)
        {
            const int tid = omp_get_thread_num();
            auto& myHist  = hist[static_cast<size_t>(tid)];
            #pragma omp for schedule(static)
            for (long long i = 0; i < static_cast<long long>(n); ++i) {
                ++myHist[(src[i] >> shift) & 0xFFu];
            }
        }

        // ---- Prefix sum across all (bucket, thread) pairs -------------
        size_t cum = 0;
        for (int b = 0; b < 256; ++b) {
            for (int t = 0; t < T; ++t) {
                off[static_cast<size_t>(t)][static_cast<size_t>(b)] = cum;
                cum += hist[static_cast<size_t>(t)][static_cast<size_t>(b)];
            }
        }

        // ---- Pass 2: scatter to dst using thread-local cursors --------
        #pragma omp parallel num_threads(T)
        {
            const int tid = omp_get_thread_num();
            size_t local[256];
            std::memcpy(local,
                        off[static_cast<size_t>(tid)].data(),
                        sizeof(local));
            #pragma omp for schedule(static)
            for (long long i = 0; i < static_cast<long long>(n); ++i) {
                const uint32_t v = src[i];
                const size_t   p = local[(v >> shift) & 0xFFu]++;
                dst[p] = v;
            }
        }

        std::swap(src, dst);
    }

    // After 4 swaps, src == keys: result already lives in `keys`. No copy.
}

// ---------------------------------------------------------------------------
// Sort a vector<float> using the radix sort above.
// Allocates one uint32_t keys buffer and one uint32_t scratch buffer.
// ---------------------------------------------------------------------------
static void sortFloats(vector<float>& v) noexcept {
    const size_t n = v.size();
    if (n < 2) return;

    vector<uint32_t> keys(n);
    vector<uint32_t> aux(n);

    // Encode floats -> order-preserving uint32 keys.
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        keys[i] = encodeFloat(v[i]);
    }

    parallelRadixSortU32(keys.data(), aux.data(), n);

    // Decode keys back to floats, writing in place over the original vector.
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        v[i] = decodeFloat(keys[i]);
    }
}

// ---------------------------------------------------------------------------
// Verification: parallel, kept functionally identical (returns bool).
// ---------------------------------------------------------------------------
bool validar_orden(const vector<float>& vec) {
    if (vec.size() < 2) return true;
    const float* a = vec.data();
    const long long n = static_cast<long long>(vec.size());
    int bad = 0;
    #pragma omp parallel for reduction(+:bad) schedule(static)
    for (long long i = 1; i < n; ++i) {
        if (a[i] < a[i - 1]) bad = 1;
    }
    return bad == 0;
}

// ---------------------------------------------------------------------------
// Fast, locale-free float parser used by the parallel file loader.
// Handles optional sign, fractional part, and decimal exponent.
// ---------------------------------------------------------------------------
static const double kPow10[] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
    1e10,1e11,1e12,1e13,1e14,1e15,1e16,1e17,1e18,1e19,
    1e20,1e21,1e22
};

static inline double pow10fast(int e) noexcept {
    if (e >= 0)  return (e <= 22) ? kPow10[e]      : std::pow(10.0,  e);
    int ne = -e; return (ne <= 22) ? 1.0 / kPow10[ne] : std::pow(10.0, e);
}

static inline bool isWs(char c) noexcept {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

static inline float parseFloat(const char*& p, const char* end) noexcept {
    bool neg = false;
    if (p < end && (*p == '-' || *p == '+')) { neg = (*p == '-'); ++p; }

    uint64_t mantissa = 0;
    while (p < end) {
        unsigned c = static_cast<unsigned>(*p) - '0';
        if (c > 9) break;
        mantissa = mantissa * 10 + c;
        ++p;
    }

    int fracDigits = 0;
    if (p < end && *p == '.') {
        ++p;
        while (p < end) {
            unsigned c = static_cast<unsigned>(*p) - '0';
            if (c > 9) break;
            mantissa = mantissa * 10 + c;
            ++p;
            ++fracDigits;
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
// Parallel file loader. One single read() into a contiguous blob, chunked at
// newline boundaries, then parsed in parallel into per-thread vectors and
// concatenated via parallel memcpy into the final vector<float>.
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

    vector<size_t> starts(static_cast<size_t>(T) + 1);
    starts[0] = 0;
    starts[static_cast<size_t>(T)] = total;
    for (int t = 1; t < T; ++t) {
        size_t pos = total * static_cast<size_t>(t) / static_cast<size_t>(T);
        while (pos < total && blob[pos] != '\n') ++pos;
        if (pos < total) ++pos;
        starts[static_cast<size_t>(t)] = pos;
    }

    vector<vector<float>> partes(static_cast<size_t>(T));
    const size_t reserveHint = (20'000'000u / static_cast<size_t>(T)) + 4096;

    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        const char* p   = blob.data() + starts[static_cast<size_t>(tid)];
        const char* end = blob.data() + starts[static_cast<size_t>(tid) + 1];

        vector<float>& local = partes[static_cast<size_t>(tid)];
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

    vector<size_t> offsets(static_cast<size_t>(T) + 1, 0);
    for (int t = 0; t < T; ++t)
        offsets[static_cast<size_t>(t) + 1] =
            offsets[static_cast<size_t>(t)] + partes[static_cast<size_t>(t)].size();

    out.resize(offsets[static_cast<size_t>(T)]);

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        const auto& part = partes[static_cast<size_t>(t)];
        if (!part.empty()) {
            std::memcpy(out.data() + offsets[static_cast<size_t>(t)],
                        part.data(),
                        part.size() * sizeof(float));
        }
    }
    return true;
}

// ===========================================================================
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 8;
    omp_set_num_threads(static_cast<int>(hc));

    vector<float> numbers;
    if (!leerArchivo("datos.txt", numbers)) {
        cout << "ERROR: no se pudo abrir 'datos.txt'\n";
        return 1;
    }

    cout << "Vector (size): " << numbers.size() << endl;

    // Touch every page once so the first OMP loop doesn't pay first-touch
    // page-fault costs inside the timed region.
    {
        volatile float sink = 0.0f;
        const size_t n = numbers.size();
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            sink = numbers[i];
        }
        (void)sink;
    }

    auto start = chrono::high_resolution_clock::now();

    // ===== INJECTED PARALLEL RADIX SORT ====================================
    sortFloats(numbers);
    // =======================================================================

    auto end      = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "Tiempo: " << duration.count() << " mili." << endl;
    cout << "Ordenado: " << (validar_orden(numbers) ? "Sí" : "No") << endl;
    return 0;
}
