#include "../src/semicircle_packing/area_kernel.cpp"
extern "C" void pair_area_matrix(int n,int m,const double* a,const double* b,double* out) {
 for(int i=0;i<n;i++) for(int j=0;j<m;j++) out[i*m+j]=area(a+3*i,b+3*j);
}
