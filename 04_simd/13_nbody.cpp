#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  float j[N];  //add
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j[i] = i; //add
  }
  __m256 jvec = _mm256_load_ps(j);
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 mask, rxvec, ryvec, invrvec, invr3vec, fxvec, fyvec;
  for(int i=0; i<N; i++) {
    __m256 ivec =_mm256_set1_ps(i);
    // if(i != j) {
    mask = _mm256_cmp_ps(ivec,jvec,_CMP_NEQ_OQ);

    // float rx = x[i] - x[j];
    // float ry = y[i] - y[j];
    rxvec = _mm256_sub_ps(_mm256_set1_ps(x[i]), xvec);
    ryvec = _mm256_sub_ps(_mm256_set1_ps(y[i]), yvec);

    // float r = std::sqrt(rx * rx + ry * ry);
    invrvec = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec,rxvec),_mm256_mul_ps(ryvec,ryvec)));
    invr3vec = _mm256_mul_ps(invrvec,_mm256_mul_ps(invrvec,invrvec));

    // fx[i] -= rx * m[j] / (r * r * r);
    // fy[i] -= ry * m[j] / (r * r * r);
    //blend with mask
    fxvec = _mm256_blendv_ps(_mm256_setzero_ps(),_mm256_mul_ps(_mm256_mul_ps(rxvec,mvec),invr3vec),mask);
    fyvec = _mm256_blendv_ps(_mm256_setzero_ps(),_mm256_mul_ps(_mm256_mul_ps(ryvec,mvec),invr3vec),mask);

    __m256 tmp_x = _mm256_permute2f128_ps(fxvec,fxvec,1);
    tmp_x = _mm256_add_ps(tmp_x,fxvec);
    tmp_x = _mm256_hadd_ps(tmp_x,tmp_x);
    tmp_x = _mm256_hadd_ps(tmp_x,tmp_x);
    float fxi[N];
    _mm256_store_ps(fxi, tmp_x);
    fx[i] -= fxi[0];

    __m256 tmp_y = _mm256_permute2f128_ps(fyvec,fyvec,1);
    tmp_y = _mm256_add_ps(tmp_y,fyvec);
    tmp_y = _mm256_hadd_ps(tmp_y,tmp_y);
    tmp_y = _mm256_hadd_ps(tmp_y,tmp_y);
    float fyi[N];
    _mm256_store_ps(fyi, tmp_y);
    fy[i] -= fyi[0];
    
    // }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
