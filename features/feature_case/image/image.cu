
struct TexList {
  cudaTextureObject_t tex1;  
  cudaTextureObject_t tex2;
  cudaTextureObject_t tex3;
};

__device__ void texlist_device(TexList list) {
  int a;
  tex1Dfetch(&a, list.tex1, 1);
}

__global__ void texlist_kernel(TexList list) {
  int b;
  texlist_device(list);
  tex1Dfetch(&b, list.tex2, 1);
  tex1Dfetch(&b, list.tex3, 1);
}

cudaTextureObject_t createTexture(int *data, size_t size) {
  cudaTextureObject_t tex;
  cudaResourceDesc res;
  cudaTextureDesc desc;
  res.resType = cudaResourceTypeLinear;
  res.res.linear.devPtr = data;
  res.res.linear.desc.f = cudaChannelFormatKindSigned;
  res.res.linear.desc.x = sizeof(int) * 8; // bits per channel
  res.res.linear.sizeInBytes = 32 * sizeof(int);
  desc.addressMode[0] = cudaAddressModeClamp;
  desc.addressMode[1] = cudaAddressModeClamp;
  desc.addressMode[2] = cudaAddressModeClamp;
  desc.filterMode = cudaFilterModeLinear;
  cudaCreateTextureObject(&tex, &res, &desc, NULL);
  return tex;
}

void test(TexList list) {
  texlist_kernel<<<1, 1>>>(list);
}

int main() {
  TexList tex;
  int data[32];
  for (unsigned i = 0; i < 32; ++i)
    data[i] = i;
  
  tex.tex1 = createTexture(data, 32);
  tex.tex2 = createTexture(data, 32);
  tex.tex3 = createTexture(data, 32);
  test(tex);
}


