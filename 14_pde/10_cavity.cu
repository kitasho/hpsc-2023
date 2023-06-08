#include <cstdlib>
#include <cstdio>
#include <chrono>

using namespace std;

const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;
const double dx = 2. / (nx - 1);
const double dy = 2. / (ny - 1);
const double dt = 0.01;
const double rho = 1.00;
const double nu = 0.02;

dim3 block( 1024, 1);
dim3 grid( (nx + 1024 - 1) / 1024, ny);

__global__ void calc_b(double *u, double *v, double *b){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int i = idy * nx + idx;
    if (idx >= 1 && idx < nx-1 && idy >= 1 && idy < ny-1){
        b[i] = rho * (1 / dt *
                    ((u[i+1] - u[i-1]) / (2 * dx) + (v[i+nx] - v[i-nx]) / (2 * dy)) - 
                    ((u[i+1] - u[i-1]) / (2 * dx)) * ((u[i+1] - u[i-1]) / (2 * dx))
                        - 2 * ((u[i+nx] - u[i-nx]) / (2 * dy) * (v[i+1] - v[i-1]) / (2 * dx)) -
                    ((v[i+nx] - v[i-nx]) / (2 * dy)) * ((v[i+nx] - v[i-nx]) / (2 * dy)));
    }
}

__global__ void update_p(double *p, double *pn){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int i = idy * nx + idx;
    if (idx < nx && idy < ny){
        pn[i] = p[i];
    }
}

__global__ void calc_p(double *p, double *pn, double *b){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int i = idy * nx + idx;
    if (idx >= 1 && idx < nx-1 && idy >= 1 && idy < ny-1){
        p[i] = ( dy * dy * (pn[i+1] + pn[i-1]) + dx * dx * (pn[i+nx] + pn[i-nx]) - b[i] *dx*dx*dy*dy ) / (2 * ( dx*dx + dy*dy ));
    }
}

__global__ void boundary_p(double *p){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int i = idy * nx + idx;
    if (idx == 0 && idy < ny){
        p[i] = p[i+1];
    }
    if (idx == nx-1 && idy < ny){
        p[i] = p[i-1];
    }
    if (idy ==0 && idx < nx){
        p[i] = p[i+nx];
    }
    if (idy == ny-1 && idx < nx){
        p[i] = 0.0;
    }
}

__global__ void update_uv(double *u, double *v, double *un, double *vn){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int i = idy * nx + idx;
    if (idx < nx && idy < ny){
        un[i] = u[i];
        vn[i] = v[i];
    }
}

__global__ void advection_diffusion2D(double *u, double *v, double *un, double *vn, double *p){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int i = idy * nx + idx;
    if (idx >= 1 && idx < nx-1 && idy >= 1 && idy < ny){
        u[i] = un[i]    - un[i] * dt / dx * (un[i] - un[i-1])
                        - un[i] * dt / dy * (un[i] - un[i-nx])
                        - dt / (2. * rho * dx) * (p[i+1] - p[i-1])
                        + nu * dt / (dx*dx) * (un[i+1] -2*un[i] + un[i-1])
                        + nu * dt / (dy*dy) * (un[i+nx] -2*un[i] + un[i-nx]);
        v[i] = vn[i]    - vn[i] * dt / dx * (vn[i] - vn[i-1])
                        - vn[i] * dt / dy * (vn[i] - vn[i-nx])
                        - dt / (2. * rho * dy) * (p[i+nx] - p[i-nx])            //collect from (2 * rho * dx) in 10_cavity.py
                        + nu * dt / (dx*dx) * (vn[i+1] -2*vn[i] + vn[i-1])
                        + nu * dt / (dy*dy) * (vn[i+nx] -2*vn[i] + vn[i-nx]);
    }
}

__global__ void boundary_uv(double *u, double *v){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int i = idy * nx + idx;
    if (idx == 0 && idy < ny){
        u[i] = 0.0;
        v[i] = 0.0;
    }
    if (idx == nx-1 && idy < ny){
        u[i] = 0.0;
        v[i] = 0.0;
    }
    if (idy ==0 && idx < nx){
        u[i] = 0.0;
        v[i] = 0.0;
    }
    if (idy == ny-1 && idx < nx){
        u[i] = 1.0;
        v[i] = 0.0;
    }
}

void output(int step, double *u, double *v, double *p) {
    char filename[100];
    sprintf(filename, "result_%d.csv", step);

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int index = j * nx + i;
            fprintf(file, "%f,%f,%f,%f,%f\n", i * dx, j * dy, u[index], v[index], p[index]);
        }
    }

    fclose(file);

    printf("Output file %s\n", filename);
}

int main (){
    double *u, *v, *b, *p,*un, *vn, *pn;
    cudaMallocManaged(&u,  nx * ny * sizeof(double));
    cudaMallocManaged(&v,  nx * ny * sizeof(double));
    cudaMallocManaged(&b,  nx * ny * sizeof(double));
    cudaMallocManaged(&p,  nx * ny * sizeof(double));
    cudaMallocManaged(&un, nx * ny * sizeof(double));
    cudaMallocManaged(&vn, nx * ny * sizeof(double));
    cudaMallocManaged(&pn, nx * ny * sizeof(double));

    cudaMemset(u, 0, nx * ny * sizeof(double));
    cudaMemset(v, 0, nx * ny * sizeof(double));
    cudaMemset(b, 0, nx * ny * sizeof(double));
    cudaMemset(p, 0, nx * ny * sizeof(double));
    cudaMemset(un, 0, nx * ny * sizeof(double));
    cudaMemset(vn, 0, nx * ny * sizeof(double));
    cudaMemset(pn, 0, nx * ny * sizeof(double));
    
    auto tic = chrono::steady_clock::now();
    int n;
    for(n=0; n<nt; n++){
        
        calc_b<<<grid, block>>>(u, v, b);
        cudaDeviceSynchronize();

        for(int it=0; it<nit; it++){
            update_p<<<grid, block>>>(p, pn);
            cudaDeviceSynchronize();

            calc_p<<<grid, block>>>(p, pn, b);
            cudaDeviceSynchronize();

            boundary_p<<<grid, block>>>(p);
            cudaDeviceSynchronize();
        }

        update_uv<<<grid, block>>>(u, v, un, vn);
        cudaDeviceSynchronize();

        advection_diffusion2D<<<grid, block>>>(u, v, un, vn, p);
        cudaDeviceSynchronize();
        
        boundary_uv<<<grid, block>>>(u, v);
        cudaDeviceSynchronize();

        if (n % 50 == 0){
            output(n, u, v, p);
        }
    }
    auto toc = chrono::steady_clock::now();
    double time = chrono::duration<double>(toc-tic).count();
    output(n, u, v, p);
    printf("calc time : %lf s\n",time);
    cudaFree(u);
    cudaFree(v);
    cudaFree(b);
    cudaFree(p);
    cudaFree(un);
    cudaFree(vn);
    cudaFree(pn);
    return 0;
}
