#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void initialize_bucket(int *bucket, int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= range) return;
  bucket[i] = 0;
}

__global__ void scatter(int *bucket, int *key, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void scan(int *bucket, int *temp,  int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= range) return;
  for(int j=1; j<range; j<<=1){
    temp[i] = bucket[i];
    __syncthreads();

    bucket[i] += temp[i-j];
    __syncthreads();
  }
}

__global__ void sort(int *bucket, int *key, int n, int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  key[i] = 0;
  for(int j=1; j<range; j++){
    if(i >= bucket[j-1]){
      key[i] = j;
    }
  }
  return;
}

int main() {
  int n = 50;
  int range = 5;
  
  // std::vector<int> key(n);
  
  int *key, *bucket, *temp;
  cudaMallocManaged(&key, n*sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  // std::vector<int> bucket(range); 
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&temp, range*sizeof(int));

  // for (int i=0; i<range; i++) {
  //   bucket[i] = 0;
  // }
  initialize_bucket<<<(range+1024-1)/1024,1024>>>(bucket, range);
  cudaDeviceSynchronize();

  // for (int i=0; i<n; i++) {
  //   bucket[key[i]]++;
  // }
  scatter<<<(n+1024-1)/1024,1024>>>(bucket, key, n);
  cudaDeviceSynchronize();

  // for (int i=0, j=0; i<range; i++) {
  //   for (; bucket[i]>0; bucket[i]--) {
  //     key[j++] = i;
  //   }
  // }
  scan<<<(range+1024-1)/1024,1024>>>(bucket, temp, range);
  cudaDeviceSynchronize();
  
  sort<<<(n+1024-1)/1024,1024>>>(bucket, key, n, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
