#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <future>

// Umbral para evitar el sobrecosto de crear demasiados hilos
const int UMBRAL = 100000;

void merge(std::vector<double>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<double> L(n1), R(n2);
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    while (i < n1) { arr[k] = L[i]; i++; k++; }
    while (j < n2) { arr[k] = R[j]; j++; k++; }
}

// Función de ordenamiento concurrente usando el enfoque de tu clase (Pág. 44-45)
void parallelMergeSort(std::vector<double>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        // Si el subarreglo es grande, aplicamos concurrencia asíncrona
        if ((right - left) > UMBRAL) {
            // 1. Crear el promise y obtener su future asociado (Pág. 44)
            std::promise<void> promiseIzquierdo;
            std::future<void> futureIzquierdo = promiseIzquierdo.get_future();

            // 2. Crear un hilo exclusivo para la mitad izquierda usando una expresión lambda (Pág. 45)
            std::thread hiloIzquierdo([&]() {
                parallelMergeSort(arr, left, mid);
                promiseIzquierdo.set_value(); // Notifica que terminó (Pág. 45)
            });

            // 3. El hilo actual resuelve de forma síncrona la mitad derecha
            parallelMergeSort(arr, mid + 1, right);

            // 4. Sincronización: Esperar el resultado del futuro y unir el hilo (Pág. 45)
            futureIzquierdo.get(); 
            hiloIzquierdo.join();  
        } else {
            // Si el tamaño es menor al umbral, procesamos secuencialmente para ser eficientes
            parallelMergeSort(arr, left, mid);
            parallelMergeSort(arr, mid + 1, right);
        }

        // Fusión final de ambas mitades
        merge(arr, left, mid, right);
    }
}

int main() {
    const size_t N = 100000000; // 1 Millón de datos
    std::cout << "Generando " << N << " numeros reales aleatorios...\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-10000.0, 10000.0);

    std::vector<double> numeros(N);
    for (size_t i = 0; i < N; ++i) numeros[i] = dis(gen);

    std::cout << "Iniciando ordenamiento concurrente (Merge Sort + Future/Promise)...\n";

    auto inicio = std::chrono::high_resolution_clock::now();

    parallelMergeSort(numeros, 0, numeros.size() - 1);

    auto fin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duracion = fin - inicio;

    std::cout << "=========================================\n";
    std::cout << " TAMANO DEL ARREGLO: " << N << " elementos\n";
    std::cout << " TIEMPO DE EJECUCION: " << duracion.count() << " ms\n";
    std::cout << "=========================================\n";

    bool ordenado = true;
    for (size_t i = 1; i < numeros.size(); ++i) {
        if (numeros[i] < numeros[i - 1]) { ordenado = false; break; }
    }
    std::cout << "Verificacion de orden: " << (ordenado ? "OK" : "ERROR") << "\n";

    return 0;
}