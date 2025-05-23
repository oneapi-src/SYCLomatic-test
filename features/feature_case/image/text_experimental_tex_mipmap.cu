void set_3D_descriptor(CUDA_ARRAY3D_DESCRIPTOR &desc) {
  desc.Width = 1;
  desc.Depth = 2;
  desc.Height = 1;
  desc.Format = CU_AD_FORMAT_SIGNED_INT16;
  desc.NumChannels = 2;
}

int main() {
  CUDA_ARRAY3D_DESCRIPTOR desc;
  unsigned int numMipmapLevels = 2;
  set_3D_descriptor(desc);

  CUmipmappedArray mmArray;

  cuMipmappedArrayCreate(&mmArray, &desc, numMipmapLevels);

  CUmipmappedArray *pArray;
  cuMipmappedArrayCreate(pArray, &desc, numMipmapLevels);

  CUarray level_arr;
  cuMipmappedArrayGetLevel(&level_arr, mmArray, 1);

  CUtexref texRef;
  cuTexRefSetMipmappedArray(texRef, mmArray, 0);

  CUfilter_mode fm = CU_TR_FILTER_MODE_POINT;

  cuTexRefSetMipmapFilterMode(texRef, fm);

  cuTexRefGetMipmapFilterMode(&fm, texRef);

  float min_clamp, max_clamp;
  cuTexRefGetMipmapLevelClamp(&min_clamp, &max_clamp, texRef);

  CUmipmappedArray anotherArray;
  cuTexRefGetMipmappedArray(&anotherArray, texRef);

  cuMipmappedArrayDestroy(mmArray);

  cuMipmappedArrayDestroy(*pArray);

  return 0;
}
