#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define WIDTH 640
#define HEIGHT 480
#define BLOCKSIZE 80

uint8_t img_in[HEIGHT][WIDTH] = {0};
uint8_t img_out[HEIGHT][WIDTH] = {0};

double get_seconds(void) {
  struct timeval now;
  gettimeofday(&now, NULL);
  return now.tv_sec + now.tv_usec / 1000000.0;
}

void read_pgm(void) {
  int tmp;
  FILE *fp = fopen("in.pgm", "r");
  fseek(fp, 15, SEEK_CUR);

  for (size_t i = 0; i < HEIGHT; i++) {
    for (size_t j = 0; j < WIDTH; j++) {
      fscanf(fp, "%d", &tmp);
      img_in[i][j] = tmp;
    }
  }
}

void write_pgm(void) {
  FILE *fp = fopen("out.pgm", "w");
  fprintf(fp, "P2\n%d %d\n255\n", WIDTH, HEIGHT);

  for (size_t i = 0; i < HEIGHT; i++) {
    for (size_t j = 0; j < WIDTH; j++) {
      fprintf(fp, "%d ", img_out[i][j]);
    }
    fputc('\n', fp);
  }
}

void line_extractor_unoptimized(void) {
  double start = get_seconds();
  for (size_t i = 0; i < HEIGHT; i++) {
    for (size_t j = 0; j < WIDTH; j++) {
      // 縦方向の線抽出
      if (j < WIDTH - 1) {
        img_out[i][j] = 255 - abs(img_in[i][j + 1] - img_in[i][j]);
      }

      // 横方向の線抽出
      if (i < HEIGHT - 1) {
        img_out[i][j] &= 255 - abs(img_in[i + 1][j] - img_in[i][j]);
      }
    }
  }
  printf("Unoptimized: %f [s]\n", get_seconds() - start);
}

void line_extractor_blocking(void) {
  double start = get_seconds();
  for (size_t si = 0; si < HEIGHT; si += BLOCKSIZE) {
    for (size_t sj = 0; sj < WIDTH; sj += BLOCKSIZE) {
      for (size_t i = si; i < si + BLOCKSIZE; i++) {
        for (size_t j = sj; j < sj + BLOCKSIZE; j++) {
          // 縦方向の線抽出
          if (j < WIDTH - 1) {
            img_out[i][j] = 255 - abs(img_in[i][j + 1] - img_in[i][j]);
          }

          // 横方向の線抽出
          if (i < HEIGHT - 1) {
            img_out[i][j] &= 255 - abs(img_in[i + 1][j] - img_in[i][j]);
          }
        }
      }
    }
  }
  printf("Blocking: %f [s]\n", get_seconds() - start);
}

void line_extractor_blocking_omp(void) {
  double start = get_seconds();
#pragma omp parallel for
  for (size_t si = 0; si < HEIGHT; si += BLOCKSIZE) {
    for (size_t sj = 0; sj < WIDTH; sj += BLOCKSIZE) {
      for (size_t i = si; i < si + BLOCKSIZE; i++) {
        for (size_t j = sj; j < sj + BLOCKSIZE; j++) {
          // 縦方向の線抽出
          if (j < WIDTH - 1) {
            img_out[i][j] = 255 - abs(img_in[i][j + 1] - img_in[i][j]);
          }

          // 横方向の線抽出
          if (i < HEIGHT - 1) {
            img_out[i][j] &= 255 - abs(img_in[i + 1][j] - img_in[i][j]);
          }
        }
      }
    }
  }
  printf("Blocking with OpenMP: %f [s]\n", get_seconds() - start);
}

int main(void) {
  read_pgm();

  line_extractor_unoptimized();
  line_extractor_blocking();
  line_extractor_blocking_omp();

  write_pgm();
  return 0;
}
