#include<stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define N 960
static double A[N][N], B[N][N], C[N][N];

void mm(double A[N][N], double B[N][N], double C[N][N])
{
  int i,j,k;

  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      for (k=0; k<N; k++)        
	C[i][j] += A[i][k] * B[k][j];
}

double seconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

int main(void) {
    double t, t_ref;
    
    memset(C, 0, sizeof(C));
    int i,j;
    for (i=0; i<N; ++i)
	for (j=0; j<N; ++j)
	  A[i][j] = B[i][j] = (double)(i+j)/(j+0.5);
        

    for(i=0; i<3; ++i){
      t = seconds();
      mm(A, B, C);
      t = seconds() - t;
      printf("Time: %f sec\n", t);
    }
    return 0;
}
