#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;

bool validar_orden(const vector<float>& vec){
    for(size_t i = 1; i < vec.size(); ++i){
        if(vec[i] < vec[i-1]){
            return false;
        }
    }
    return true;
}

int main(){  
    vector<float> numbers;
    ifstream in("output.txt");

    float num;
    while(in >> num){
        numbers.push_back(num);
    }
    in.close();
    cout << "Vector (size): " << numbers.size() << endl;

    auto start = chrono::high_resolution_clock::now();
    
    // Aqui debe ir el algoritmo de ordenamiento...
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Tiempo: " << duration.count() << " mili." << endl;
    cout << "Ordenado: " << (validar_orden(numbers) ? "Sí" : "No") << endl;
    return 0;
}