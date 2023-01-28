#define checkErrors(err)  privCheckErrors (err, __FILE__, __LINE__)

inline void privCheckErrors(CUresult result, const char *file, const int line) {
  if(CUDA_SUCCESS!=result) {
    fprintf(stderr,"error = %d at %s:%d\n",result,file,line);
    exit(-1);
  }
}

#define VEC_LENGTH 128
#define SEED       59
