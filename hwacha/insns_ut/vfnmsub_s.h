require_fp;
softfloat_roundingMode = VRM;
WRITE_SFRD(f32_mulAdd(FRS1 ^ (uint32_t)INT32_MIN, FRS2, FRS3));
set_fp_exceptions;
