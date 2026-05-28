#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include <cmath>
#include <cstring>

using namespace std;

// ---------------------------------------------------------------------------
// Parametros de ajuste
// ---------------------------------------------------------------------------
static const int    INS_CUTOFF = 32;     // por debajo de esto: insertion sort
static const size_t PAR_MIN    = 50000;  // rango minimo para lanzar un hilo
static int          g_maxDepth = 3;      // se calcula segun los nucleos de la CPU

// ---------------------------------------------------------------------------
// Insertion sort para rangos pequenos [lo, hi] (inclusivo).
// Opera sobre el buffer destino, que ya contiene los valores originales.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Mezcla las dos mitades ordenadas de src en dst sobre [lo, hi] (inclusivo).
//   mitad izquierda = src[lo   .. mid]
//   mitad derecha   = src[mid+1.. hi]
// Usa el unico buffer auxiliar pasado por puntero (sin asignar memoria aqui).
// Los segmentos son disjuntos en memoria: seguro entre hilos sin mutex.
// ---------------------------------------------------------------------------
static void merge(const float* src, float* dst, size_t lo, size_t mid, size_t hi){
    size_t i = lo, j = mid + 1, k = lo;
    while(i <= mid && j <= hi){
        if(src[i] <= src[j]) dst[k++] = src[i++];
        else                 dst[k++] = src[j++];
    }
    while(i <= mid) dst[k++] = src[i++];
    while(j <= hi)  dst[k++] = src[j++];
}

// ---------------------------------------------------------------------------
// Merge sort paralelo con esquema ping-pong (sin copia de retorno).
// Los roles de src/dst se intercambian en cada nivel; el resultado de
// sort(src, dst, lo, hi) queda en dst[lo..hi]. Precondicion: src y dst
// contienen datos identicos al entrar (se garantiza con la copia inicial).
//
// Concurrencia: mientras estamos "poco profundos" en el arbol
// (depth < g_maxDepth) la mitad izquierda se manda a un std::thread y se
// sincroniza con std::promise / std::future. Mas abajo todo es secuencial
// para no saturar la CPU con hilos.
// ---------------------------------------------------------------------------
static void sort(float* src, float* dst, size_t lo, size_t hi, int depth){
    if(hi <= lo) return;                         // 0 o 1 elemento: ya esta en dst

    if(hi - lo < (size_t)INS_CUTOFF){            // rango pequeno
        insertionSort(dst, lo, hi);
        return;
    }

    size_t mid = lo + (hi - lo) / 2;

    if(depth < g_maxDepth && (hi - lo) > PAR_MIN){
        // ---- Rama paralela: mitad izquierda en un hilo trabajador --------
        promise<void> promiseIzquierdo;
        future<void>  futureIzquierdo = promiseIzquierdo.get_future();

        thread hiloIzquierdo([&](){
            sort(dst, src, lo, mid, depth + 1);  // roles intercambiados
            promiseIzquierdo.set_value();
        });

        sort(dst, src, mid + 1, hi, depth + 1);  // mitad derecha en el hilo actual

        futureIzquierdo.get();                   // espera el resultado izquierdo
        hiloIzquierdo.join();
    } else {
        // ---- Rama secuencial --------------------------------------------
        sort(dst, src, lo, mid, depth + 1);
        sort(dst, src, mid + 1, hi, depth + 1);
    }

    // Las mitades ordenadas quedaron en src. Se mezclan hacia dst.
    if(src[mid] <= src[mid + 1]){                // camino rapido: ya ordenado
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

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    vector<float> numbers;
    ifstream in("output.txt");

    float num;
    while(in >> num){
        numbers.push_back(num);
    }
    in.close();
    cout << "Vector (size): " << numbers.size() << endl;

    const size_t n = numbers.size();
    if(n == 0){
        cout << "Archivo vacio o no encontrado." << endl;
        return 1;
    }

    // Configura el paralelismo segun la CPU real
    unsigned hc = thread::hardware_concurrency();
    if(hc == 0) hc = 8;
    g_maxDepth = (int)ceil(log2((double)hc));
    cout << "Hilos de hardware: " << hc
         << " (profundidad paralela = " << g_maxDepth << ")" << endl;

    // UN solo buffer auxiliar, asignado una vez, usado como espacio de trabajo
    vector<float> buffer(numbers);   // copia identica => invariante del ping-pong

    auto start = chrono::high_resolution_clock::now();

    // src = buffer, dst = numbers  ->  el resultado ordenado queda en numbers
    sort(buffer.data(), numbers.data(), 0, n - 1, 0);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Tiempo: " << duration.count() << " mili." << endl;
    cout << "Ordenado: " << (validar_orden(numbers) ? "Sí" : "No") << endl;
    return 0;
}
