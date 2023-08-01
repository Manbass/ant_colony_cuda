#include <chrono>
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <queue>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <time.h>
#include <CL/cl.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include <yaml-cpp/yaml.h>
using namespace std;


#define vap_coef 0.9
#define grid_size 1024
#define max_size 10
#define max_dist 10
#define min_dist 0
#define num_ants 1024
#define device 0

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

const float e = 2.71828182846;

clock_t begin_clock;
clock_t begin_clock2;

char* arg1;
char* arg2;
char* arg3;
int* used;
int* adj_matrix;
float* cost;
int* pos;
int* roads;
int* d;
float* pheromons;
int* ants;
int length;
float* rands;
float range;
int st;
int en;
int def;
int* road;
int iterations;

ifstream fin;

void dejkstra() {
    clock_t  beg = clock();
    int n, s, f, l, min1 = 100000000, nmin = 0;
    int* a = new int[length*length];
    d = new int[length];
    int* flag = new int[length];
    for (int i = 0; i < length; i++) flag[i] = 1;
    n = length;
    s = st;
    f = en;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            a[i*length + j] = adj_matrix[i*length + j];
            if (a[i*length + j] == 0 && i != j) a[i*length + j] = 640000;
        }
    //*********************************************************************
    l = s;
    for (int i = 0; i < n; i++) d[i] = a[l*length + i];
    flag[l] = 0;
    for (int i = 0; i < n - 1; i++)
    {
        min1 = 100000000;
        nmin = l;
        for (int j = 0; j < n; j++)
            if (flag[j] != 0 && min1 > d[j])
            {
                min1 = d[j];
                nmin = j;
            }

        l = nmin;

        flag[l] = 0;

        for (int j = 0; j < n; j++)
            if (flag[j] != 0)
                d[j] = std::min(d[j], a[l*length + j] + d[l]);

    }
    for (int i = 0; i < length; ++i) {
        if (d[i] == 640000) d[i] = 0;
    }
    cout << clock() - beg << endl;
}


template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

vector<int> getway(int start, int end) {
    vector<int> way;
    range = 0;
    int pos;
    float val = 0;
    int prev = start;
    for (int i = 0; (i < length) && (prev != end); ++i) {
        pos = road[prev];
        way.push_back(pos);
        range += adj_matrix[prev*length + pos];
        prev = pos;
        val = 0;
    }
    return way;
}

void input(bool type) {
    if (!type) {
        length = max_size;
        // some inner algorithm required to have size being power of 2
        int powered_length = pow(2, ceil(log(length)/log(2)));
        def = 0;
        adj_matrix = new int[powered_length*powered_length];
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                adj_matrix[i*powered_length + j] = (min_dist + rand() % max_dist) * (rand() % 2);
                def = max(def, adj_matrix[i*powered_length + j]);
            }
        }
        length = powered_length;
    }
    else {
        fin >> length;
        def = 0;
        adj_matrix = new int[length*length];
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                fin >> adj_matrix[i*length + j];
                def = max(def, adj_matrix[i*length + j]);
            }
        }
    }
    road = new int[length];

}

void init(int start_pos) {
    // Filling pheromons by default
    pheromons = new float[length*length];
    fill_n(pheromons, length*length, def);
    // Filling roads by start ants position
    roads = new int[num_ants*length];
    fill_n(roads, num_ants*length, start_pos);
    rands = new float[num_ants*length];
}

// setuping random state
__global__ void setup_kernel(curandState *state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, idx, 0, &state[idx]);
}

__global__ void getRoadsCore(curandState* dstate, float* dpheromons, int* droads, int* dadj_matrix, const int dlength, const int start, const int end, float* drands) {
    unsigned long long thid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long curid = threadIdx.x;
    unsigned long long blid = blockIdx.x;
    unsigned long long block_and_dlength = blockDim.x * dlength * gridDim.x;
    extern __shared__ float clear[];
    __shared__ int prev;
    if (curid < dlength) {
        clear[curid] = 1;
    }
    __syncthreads();
    if (curid == 0) {
        clear[start] = 0;
        prev = start;
    }
    __syncthreads();
    // while end is not achived
    for (int turn = 1; clear[end] != 0 && turn < dlength; ++turn) {
        if (dadj_matrix[prev * dlength + curid] != 0) {
            float a = dpheromons[prev * dlength + curid];
            float b = 1.0 * dadj_matrix[prev * dlength + curid];
            clear[dlength + curid] = clear[curid] * a / (b)*a*a*a*a;
        }
        else {
            clear[dlength + curid] = 0;
        }

        int offset = 1;
        for (int d = dlength >> 1; d > 0; d >>= 1) // build sum in place up the tree
        {
            __syncthreads();
            if (curid < d)
            {
                int ai = offset * (2 * curid + 1) - 1;
                int bi = offset * (2 * curid + 2) - 1;
                clear[dlength + bi] += clear[dlength + ai];
            }
            offset *= 2;
        }
        if (curid == 0) {
            clear[2 * dlength] = clear[2 * dlength - 1];
            clear[2 * dlength - 1] = 0;
        } // clear the last element
        __syncthreads();
        for (int d = 1; d < dlength; d *= 2) // traverse down tree & build scan
        {
            offset >>= 1;
            __syncthreads();
            if (curid < d)
            {
                int ai = offset * (2 * curid + 1) - 1;
                int bi = offset * (2 * curid + 2) - 1;
                float t = clear[dlength + ai];
                clear[dlength + ai] = clear[dlength + bi];
                clear[dlength + bi] += t;
            }
        }
        __syncthreads();
        //thrust::inclusive_scan(thrust::system::cuda::par, &(clear[block_and_dlength + thid * dlength]), &(clear[block_and_dlength + (thid + 1) * dlength]), &(clear[block_and_dlength + thid * dlength]));
        float last = clear[dlength * 2];
        drands[blid*dlength + curid] = clear[dlength + curid];
        if (curid == 0) {
            clear[2 * dlength + 1] = curand_uniform(&(dstate[blid]));
        }
        __syncthreads();
        float myrandf = clear[2 * dlength + 1];
        myrandf *= (last - 0.0000001);
        if (myrandf < 0) {
            myrandf = 0;
        }
        if ((curid == 0 || (clear[dlength + curid] < myrandf)) && (clear[dlength + curid + 1] >= myrandf)) {
            clear[curid] = 0;
            droads[blid*dlength + prev] = curid;
            prev = curid;
        }
        __syncthreads();
    }
}


__global__ void updateCore(curandState* dstate, float* dpheromons, int* droads, int* dadj_matrix, const int dlength, const int start, const int end, const int ddef, float* dsums) {
    unsigned long long thid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long blid = blockIdx.x;
    unsigned long long curid = threadIdx.x;
    unsigned long long fullthreads = blockDim.x * gridDim.x;
    extern __shared__ float sizes[];
    int pos = 0;
    while ((pos * fullthreads + thid) < dlength*dlength) {
        dpheromons[pos * fullthreads + thid] *= vap_coef;
        //dpheromons[pos * fullthreads + thid] += 1*(1 - vap_coef);
        pos++;
    }
    sizes[curid] = 0;
    if (curid == end) {
        sizes[curid] = -2147483648.0;
    }
    __syncthreads();
    if (curid < dlength) {
        if (curid != end) {
            if (droads[blid*dlength + curid] != start) {
                sizes[curid] = dadj_matrix[curid * dlength + droads[blid*dlength + curid]];
                if (droads[blid*dlength + curid] == end) {
                    sizes[end] = 0;
                }
            }
        }
    }
    __syncthreads();
    for (unsigned int s = 1; s < dlength; s *= 2) {
        if (curid < dlength && curid % (2 * s) == 0) {
            sizes[curid] += sizes[curid + s];
        }
        __syncthreads();
    }
    if (curid == 0) {
        if (sizes[0] > 0) {
            dsums[blid] = ddef / sizes[0] * ddef / sizes[0] * ddef / sizes[0] / sizes[0];
        }
        else {
            dsums[blid] = 0;
        }
    }
}

__global__ void updateCore2(curandState* dstate, float* dpheromons, int* droads, int* dadj_matrix, const int dlength, const int start, const int end, const int ddef, float* dsums) {
    unsigned long long thid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long blid = blockIdx.x;
    unsigned long long curid = threadIdx.x;
    unsigned long long fullthreads = blockDim.x * gridDim.x;
    extern __shared__ float sizes[];
    if (curid < grid_size) {
        sizes[curid] = dsums[curid];
    }
    __syncthreads();
    if (thid < dlength) {
        for (int i = 0; i < grid_size; ++i) {
            if (sizes[i] > 0) {
                if (droads[i*dlength + thid] != start) {
                    dpheromons[thid * dlength + droads[i*dlength + thid]] += sizes[i];
                }
            }
        }
    }
}


__global__ void findRoadCore(float* dpheromons, int* droad, const int dlength, bool* dfound) {
    unsigned long long thid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long blid = blockIdx.x;
    unsigned long long curid = threadIdx.x;
    unsigned long long fullthreads = blockDim.x * gridDim.x;
    extern __shared__ float sizes[];
    __shared__ bool local_found;
    sizes[curid] = 0;
    for (int i = 0; i < dlength; ++i) {
        if (dpheromons[curid*dlength + i] > sizes[curid]) {
            sizes[curid] = dpheromons[curid*dlength + i];
            sizes[curid + dlength] = i;
        }
    }
    if (sizes[curid + dlength] != 0) {
        local_found = true;
    }
    droad[curid] = sizes[curid + dlength];

    __syncthreads();
    if (curid == 0) {
        dfound[0] = local_found;
    }
}

bool getTurn() {
    // global: float* pheromons, int* roads, int* adj_matrix
    float* dpheromons; int* droads; int* dadj_matrix; float* drands; float* dclear; float* dsums; int* droad; bool* dfound;
    // choose GPU
    cudaSetDevice(device);
    // state for random generation in core
    curandState *dstate;
    checkCudaErrors(cudaMalloc(&dstate, num_ants * sizeof(curandState)));
    setup_kernel << <grid_size, 1 >> > (dstate);
//    cout << "all fine after setup\n";
    // allocation memory on device
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMalloc(&dpheromons, sizeof(float) * length * length));
    checkCudaErrors(cudaMalloc(&dadj_matrix, sizeof(int) * length * length));
    checkCudaErrors(cudaMalloc(&droads, sizeof(int) * num_ants * length));
    checkCudaErrors(cudaMalloc(&drands, num_ants * sizeof(float) * length));
    checkCudaErrors(cudaMalloc(&dclear, 2 * num_ants * length * sizeof(float)));
    checkCudaErrors(cudaMalloc(&dsums, grid_size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&droad, length * sizeof(int)));
    checkCudaErrors(cudaMalloc(&dfound, 1 * sizeof(bool)));
    // copy data to device
    checkCudaErrors(cudaMemcpy(dpheromons, pheromons, sizeof(float) * length * length, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dadj_matrix, adj_matrix, sizeof(int) * length * length, cudaMemcpyHostToDevice));
//    cout << "all fine after allocation\n";

    for (int i = 0; i < int(log(1.0 / def) / log(vap_coef)); ++i) {
        checkCudaErrors(cudaMemcpy(droads, roads, sizeof(int) * num_ants * length, cudaMemcpyHostToDevice));
        getRoadsCore << <grid_size, length, (length * 2 + 3) * sizeof(float) >> > (dstate, dpheromons, droads, dadj_matrix, length, st, en, drands);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
//        cout << "all fine after core\n";
        updateCore << <grid_size, 1024, (1024) * sizeof(float) >> > (dstate, dpheromons, droads, dadj_matrix, length, st, en, def, dsums);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        updateCore2 << <1, 1024, (grid_size * 3) * sizeof(float) >> > (dstate, dpheromons, droads, dadj_matrix, length, st, en, def, dsums);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
    }
    findRoadCore << <1, length, length * 2 * sizeof(float) >> > (dpheromons, droad, length, dfound);
//    cout << "all fine after 2nd core\n";

    bool* found;
    found = new bool[1];

    checkCudaErrors(cudaMemcpy(road, droad, sizeof(int)* length, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(roads, droads, sizeof(int)* num_ants * length, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(rands, drands, sizeof(float)* num_ants * length, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pheromons, dpheromons, sizeof(float)* length * length, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(found, dfound, sizeof(bool)* 1, cudaMemcpyDeviceToHost));

    cudaFree(dpheromons);
    cudaFree(droads);
    cudaFree(dadj_matrix);
    cudaFree(drands);
    cudaFree(droad);
    cudaFree(dfound);
    return found[0];
}

int get_edges() {
    int counter = 0;
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < length; ++j) {
            if (adj_matrix[i*length + j] != 0) {
                counter++;
            }
        }
    }
    return counter;
}


template< class T>
void disp(T arr, int x, int y) {
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            cout << arr[i*y + j] << ' ';
        }
        cout << "\n";
    }
    cout << "\n";
}

void manual_input () {
    int number_of_nodes;
    cout << "Please, enter number of nodes: ";
    cin >> number_of_nodes;
    cout << "Enter adjacency matrix, in following format:\n x x x\n x x x \n x x x\n";
    int powered_length = pow(2, ceil(log(number_of_nodes)/log(2)));
    adj_matrix = new int[powered_length*powered_length];
    for (int i = 0; i < number_of_nodes; ++i) {
        for (int j = 0; j < number_of_nodes; ++j) {
            cin >> adj_matrix[powered_length * i + j];
            def = max(def, adj_matrix[i*powered_length + j]);
        }
    }
    length = powered_length;
};

void file_input () {
    string filename;
    cout << "File should contain graph in format \n n \n x x x \n x x x \n x x x \n Please, enter path to file: ";
    cin >> filename;
    fin = ifstream(filename);
    int number_of_nodes;
    fin >> number_of_nodes;
    int powered_length = pow(2, ceil(log(number_of_nodes) / log(2)));
    adj_matrix = new int[powered_length * powered_length];
    for (int i = 0; i < number_of_nodes; ++i) {
        for (int j = 0; j < number_of_nodes; ++j) {
            fin >> adj_matrix[powered_length * i + j];
            def = max(def, adj_matrix[i * powered_length + j]);
        }
    }
    length = powered_length;
    fin.close();
}

int main() {
    bool tryagain = true;
    char mode;
    def = 0;
    while (tryagain) {
        tryagain = false;
        cout << "To input graph manually enter 'm', to upload from file - 'f': \n";
        cin >> mode;

        switch (mode) {
            case 'm':
                manual_input();
                break;
            case 'f':
                file_input();
                break;
            default:
                cout << "Unknown command received, try again? (y/n)";
                char yes;
                cin >> yes;
                if (yes == 'y') {
                    tryagain = true;
                    continue;
                }
        };

        srand(time(NULL));
        cout << "Please enter start node: ";
        cin >> st;
        cout << "Please enter finish node: ";
        cin >> en;


        bool found;
        vector<int> way;
        road = new int[length];
        if (st < length && en < length && st != en) {
            init(st);
            found = getTurn();
            way = getway(st, en);
        }

        if (!found) {
            cout << "path not found";
            return 0;
        } else {
            cout << "\nway: \n";
            disp(way, 1, way.size());
        }
    }

    return 0;
}