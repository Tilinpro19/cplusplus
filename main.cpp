#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include <cmath>
#include <cstring>
#include <cstdlib>

using namespace std;

// ---------------------------------------------------------------------------
// Parametros de ajuste del merge sort paralelo
// ---------------------------------------------------------------------------
static const int    INS_CUTOFF = 32;     // por debajo de esto: insertion sort
static const size_t PAR_MIN    = 50000;  // rango minimo para lanzar un hilo
static int          g_maxDepth = 3;      // se calcula segun los nucleos de la CPU

// Insertion sort para rangos pequenos [lo, hi] (inclusivo).
static inline void insertionSort(float* a, size_t lo, size_t hi){
    for(size_t i = lo + 1; i <= hi; ++i){
        float key = a[i];
        size_t j = i;
        while(j > lo && a[j-1] > key){
            a[j] = a[j-1];
            --j;
        }
        a[j] = key;
    }
}

// Mezcla las dos mitades ordenadas de src en dst sobre [lo, hi] (inclusivo).
// Usa el unico buffer auxiliar (sin asignar memoria aqui). Segmentos disjuntos:
// seguro entre hilos sin mutex.
static void merge(const float* src, float* dst, size_t lo, size_t mid, size_t hi){
    size_t i = lo, j = mid + 1, k = lo;
    while(i <= mid && j <= hi){
        if(src[i] <= src[j]) dst[k++] = src[i++];
        else                 dst[k++] = src[j++];
    }
    while(i <= mid) dst[k++] = src[i++];
    while(j <= hi)  dst[k++] = src[j++];
}

// Merge sort paralelo (ping-pong: src/dst alternan roles por nivel, sin copia
// de retorno; el resultado de sort(src,dst,lo,hi) queda en dst[lo..hi]).
// Concurrencia con std::thread + std::promise + std::future mientras
// depth < g_maxDepth; mas abajo, secuencial.
static void mergeSort(float* src, float* dst, size_t lo, size_t hi, int depth){
    if(hi <= lo) return;

    if(hi - lo < (size_t)INS_CUTOFF){
        insertionSort(dst, lo, hi);
        return;
    }

    size_t mid = lo + (hi - lo) / 2;

    if(depth < g_maxDepth && (hi - lo) > PAR_MIN){
        promise<void> promiseIzquierdo;
        future<void>  futureIzquierdo = promiseIzquierdo.get_future();

        thread hiloIzquierdo([&](){
            mergeSort(dst, src, lo, mid, depth + 1);   // roles intercambiados
            promiseIzquierdo.set_value();
        });

        mergeSort(dst, src, mid + 1, hi, depth + 1);   // mitad derecha aqui

        futureIzquierdo.get();
        hiloIzquierdo.join();
    } else {
        mergeSort(dst, src, lo, mid, depth + 1);
        mergeSort(dst, src, mid + 1, hi, depth + 1);
    }

    if(src[mid] <= src[mid + 1]){                      // camino rapido: ya ordenado
        memcpy(dst + lo, src + lo, (hi - lo + 1) * sizeof(float));
    } else {
        merge(src, dst, lo, mid, hi);
    }
}

bool validar_orden(const vector<float>& vec){
    for(size_t i = 1; i < vec.size(); ++i){
        if(vec[i] < vec[i-1]){
            return false;
        }
    }
    return true;
}

// Lectura ultra rapida: lee TODO el archivo de una vez a un bloque y luego
// parsea los numeros con strtof. Evita el cuello de botella de "in >> num".
static bool leerArchivo(const char* ruta, vector<float>& out){
    ifstream in(ruta, ios::binary | ios::ate);
    if(!in) return false;

    streamsize size = in.tellg();
    in.seekg(0, ios::beg);

    vector<char> blob((size_t)size + 1);
    if(!in.read(blob.data(), size)) return false;
    blob[(size_t)size] = '\0';

    out.clear();
    out.reserve(20000000);

    const char* p = blob.data();
    char* end = nullptr;
    while(true){
        float v = strtof(p, &end);
        if(p == end) break;          // ya no hay mas numeros
        out.push_back(v);
        p = end;
    }
    return true;
}

int main(){
    vector<float> numbers;

    auto loadStart = chrono::high_resolution_clock::now();
    if(!leerArchivo("datos.txt", numbers)){
        cout << "ERROR: no se pudo abrir 'datos.txt'" << endl;
        return 1;
    }
    auto loadEnd = chrono::high_resolution_clock::now();
    auto loadMs = chrono::duration_cast<chrono::milliseconds>(loadEnd - loadStart);

    cout << "Vector (size): " << numbers.size() << endl;
    cout << "Tiempo de carga: " << loadMs.count() << " mili." << endl;

    auto start = chrono::high_resolution_clock::now();

    // Aqui debe ir el algoritmo de ordenamiento...
    if(numbers.size() > 1){
        unsigned hc = thread::hardware_concurrency();
        if(hc == 0) hc = 8;
        g_maxDepth = (int)ceil(log2((double)hc));     // ~1 tarea por nucleo

        vector<float> buffer(numbers);                // UN buffer auxiliar (copia)
        mergeSort(buffer.data(), numbers.data(), 0, numbers.size() - 1, 0);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Tiempo: " << duration.count() << " mili." << endl;
    cout << "Ordenado: " << (validar_orden(numbers) ? "Sí" : "No") << endl;
    return 0;
}
