#include <miner.h>
#include <cuda_helper.h>

__device__ const uint32_t sbox[] = {
	0x7b777c63, 0xc56f6bf2, 0x2b670130, 0x76abd7fe, 0x7dc982ca, 0xf04759fa, 0xafa2d4ad, 0xc072a49c, 0x2693fdb7, 0xccf73f36, 0xf1e5a534, 0x1531d871, 0xc323c704, 0x9a059618, 0xe2801207, 0x75b227eb, 0x1a2c8309, 0xa05a6e1b, 0xb3d63b52, 0x842fe329, 0xed00d153, 0x5bb1fc20, 0x39becb6a, 0xcf584c4a, 0xfbaaefd0, 0x85334d43, 0x7f02f945, 0xa89f3c50, 0x8f40a351, 0xf5389d92, 0x21dab6bc, 0xd2f3ff10, 0xec130ccd, 0x1744975f, 0x3d7ea7c4, 0x73195d64, 0xdc4f8160, 0x88902a22, 0x14b8ee46, 0xdb0b5ede, 0x0a3a32e0, 0x5c240649, 0x62acd3c2, 0x79e49591, 0x6d37c8e7, 0xa94ed58d, 0xeaf4566c, 0x08ae7a65, 0x2e2578ba, 0xc6b4a61c, 0x1f74dde8, 0x8a8bbd4b, 0x66b53e70, 0x0ef60348, 0xb9573561, 0x9e1dc186, 0x1198f8e1, 0x948ed969, 0xe9871e9b, 0xdf2855ce, 0x0d89a18c, 0x6842e6bf, 0x0f2d9941, 0x16bb54b0,
	0x7b777c63, 0xc56f6bf2, 0x2b670130, 0x76abd7fe, 0x7dc982ca, 0xf04759fa, 0xafa2d4ad, 0xc072a49c, 0x2693fdb7, 0xccf73f36, 0xf1e5a534, 0x1531d871, 0xc323c704, 0x9a059618, 0xe2801207, 0x75b227eb, 0x1a2c8309, 0xa05a6e1b, 0xb3d63b52, 0x842fe329, 0xed00d153, 0x5bb1fc20, 0x39becb6a, 0xcf584c4a, 0xfbaaefd0, 0x85334d43, 0x7f02f945, 0xa89f3c50, 0x8f40a351, 0xf5389d92, 0x21dab6bc, 0xd2f3ff10, 0xec130ccd, 0x1744975f, 0x3d7ea7c4, 0x73195d64, 0xdc4f8160, 0x88902a22, 0x14b8ee46, 0xdb0b5ede, 0x0a3a32e0, 0x5c240649, 0x62acd3c2, 0x79e49591, 0x6d37c8e7, 0xa94ed58d, 0xeaf4566c, 0x08ae7a65, 0x2e2578ba, 0xc6b4a61c, 0x1f74dde8, 0x8a8bbd4b, 0x66b53e70, 0x0ef60348, 0xb9573561, 0x9e1dc186, 0x1198f8e1, 0x948ed969, 0xe9871e9b, 0xdf2855ce, 0x0d89a18c, 0x6842e6bf, 0x0f2d9941, 0x16bb54b0,
	0x7b777c63, 0xc56f6bf2, 0x2b670130, 0x76abd7fe, 0x7dc982ca, 0xf04759fa, 0xafa2d4ad, 0xc072a49c, 0x2693fdb7, 0xccf73f36, 0xf1e5a534, 0x1531d871, 0xc323c704, 0x9a059618, 0xe2801207, 0x75b227eb, 0x1a2c8309, 0xa05a6e1b, 0xb3d63b52, 0x842fe329, 0xed00d153, 0x5bb1fc20, 0x39becb6a, 0xcf584c4a, 0xfbaaefd0, 0x85334d43, 0x7f02f945, 0xa89f3c50, 0x8f40a351, 0xf5389d92, 0x21dab6bc, 0xd2f3ff10, 0xec130ccd, 0x1744975f, 0x3d7ea7c4, 0x73195d64, 0xdc4f8160, 0x88902a22, 0x14b8ee46, 0xdb0b5ede, 0x0a3a32e0, 0x5c240649, 0x62acd3c2, 0x79e49591, 0x6d37c8e7, 0xa94ed58d, 0xeaf4566c, 0x08ae7a65, 0x2e2578ba, 0xc6b4a61c, 0x1f74dde8, 0x8a8bbd4b, 0x66b53e70, 0x0ef60348, 0xb9573561, 0x9e1dc186, 0x1198f8e1, 0x948ed969, 0xe9871e9b, 0xdf2855ce, 0x0d89a18c, 0x6842e6bf, 0x0f2d9941, 0x16bb54b0,
	0x7b777c63, 0xc56f6bf2, 0x2b670130, 0x76abd7fe, 0x7dc982ca, 0xf04759fa, 0xafa2d4ad, 0xc072a49c, 0x2693fdb7, 0xccf73f36, 0xf1e5a534, 0x1531d871, 0xc323c704, 0x9a059618, 0xe2801207, 0x75b227eb, 0x1a2c8309, 0xa05a6e1b, 0xb3d63b52, 0x842fe329, 0xed00d153, 0x5bb1fc20, 0x39becb6a, 0xcf584c4a, 0xfbaaefd0, 0x85334d43, 0x7f02f945, 0xa89f3c50, 0x8f40a351, 0xf5389d92, 0x21dab6bc, 0xd2f3ff10, 0xec130ccd, 0x1744975f, 0x3d7ea7c4, 0x73195d64, 0xdc4f8160, 0x88902a22, 0x14b8ee46, 0xdb0b5ede, 0x0a3a32e0, 0x5c240649, 0x62acd3c2, 0x79e49591, 0x6d37c8e7, 0xa94ed58d, 0xeaf4566c, 0x08ae7a65, 0x2e2578ba, 0xc6b4a61c, 0x1f74dde8, 0x8a8bbd4b, 0x66b53e70, 0x0ef60348, 0xb9573561, 0x9e1dc186, 0x1198f8e1, 0x948ed969, 0xe9871e9b, 0xdf2855ce, 0x0d89a18c, 0x6842e6bf, 0x0f2d9941, 0x16bb54b0
};
//#define XT(x) (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1b))
#define XT(x) (((x) << 1) ^ (((x) >> 7) ? 0x1b : 0))
__global__ void verus_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce);
static uint32_t *d_nonces[MAX_GPUS];
__device__ __constant__ uint8_t blockhash_half[32];
__device__ __constant__ uint32_t ptarget;

__host__
void verus_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_nonces[thr_id], 2 * sizeof(uint32_t)));
}

void verus_setBlock(uint8_t *blockf, uint32_t *pTargetIn)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(ptarget, (void**)&pTargetIn[7], sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(blockhash_half, (void**)blockf, 32, 0, cudaMemcpyHostToDevice));
}

__host__
void verus_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces)
{
	cudaMemset(d_nonces[thr_id], 0xff, 2 * sizeof(uint32_t));
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	verus_gpu_hash << <grid, block >> >(threads, startNonce, d_nonces[thr_id]);
	cudaThreadSynchronize();
	cudaMemcpy(resNonces, d_nonces[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	//memcpy(resNonces, h_nonces[thr_id], NBN * sizeof(uint32_t));

}


// Simulate _mm_aesenc_si128 instructions from AESNI
__device__ __forceinline__ void aesenc(unsigned char *s, uint32_t *sharedMemory1)
{
	uint32_t  t, u;
	uint32_t v[4][4];

	v[0][0] = ((uint8_t*)&sharedMemory1[0])[s[ 0]];
	v[3][1] = ((uint8_t*)&sharedMemory1[0])[s[ 1]];
	v[2][2] = ((uint8_t*)&sharedMemory1[0])[s[ 2]];
	v[1][3] = ((uint8_t*)&sharedMemory1[0])[s[ 3]];
	v[1][0] = ((uint8_t*)&sharedMemory1[0])[s[ 4]];
	v[0][1] = ((uint8_t*)&sharedMemory1[0])[s[ 5]];
	v[3][2] = ((uint8_t*)&sharedMemory1[0])[s[ 6]];
	v[2][3] = ((uint8_t*)&sharedMemory1[0])[s[ 7]];
	v[2][0] = ((uint8_t*)&sharedMemory1[0])[s[ 8]];
	v[1][1] = ((uint8_t*)&sharedMemory1[0])[s[ 9]];
	v[0][2] = ((uint8_t*)&sharedMemory1[0])[s[10]];
	v[3][3] = ((uint8_t*)&sharedMemory1[0])[s[11]];
	v[3][0] = ((uint8_t*)&sharedMemory1[0])[s[12]];
	v[2][1] = ((uint8_t*)&sharedMemory1[0])[s[13]];
	v[1][2] = ((uint8_t*)&sharedMemory1[0])[s[14]];
	v[0][3] = ((uint8_t*)&sharedMemory1[0])[s[15]];

	t = v[0][0];
	u = v[0][0] ^ v[0][1] ^ v[0][2] ^ v[0][3];
	v[0][0] = v[0][0] ^ u ^ XT(v[0][0] ^ v[0][1]);
	v[0][1] = v[0][1] ^ u ^ XT(v[0][1] ^ v[0][2]);
	v[0][2] = v[0][2] ^ u ^ XT(v[0][2] ^ v[0][3]);
	v[0][3] = v[0][3] ^ u ^ XT(v[0][3] ^ t);
	t = v[1][0];
	u = v[1][0] ^ v[1][1] ^ v[1][2] ^ v[1][3];
	v[1][0] = v[1][0] ^ u ^ XT(v[1][0] ^ v[1][1]);
	v[1][1] = v[1][1] ^ u ^ XT(v[1][1] ^ v[1][2]);
	v[1][2] = v[1][2] ^ u ^ XT(v[1][2] ^ v[1][3]);
	v[1][3] = v[1][3] ^ u ^ XT(v[1][3] ^ t);
	t = v[2][0];
	u = v[2][0] ^ v[2][1] ^ v[2][2] ^ v[2][3];
	v[2][0] = v[2][0] ^ u ^ XT(v[2][0] ^ v[2][1]);
	v[2][1] = v[2][1] ^ u ^ XT(v[2][1] ^ v[2][2]);
	v[2][2] = v[2][2] ^ u ^ XT(v[2][2] ^ v[2][3]);
	v[2][3] = v[2][3] ^ u ^ XT(v[2][3] ^ t);
	t = v[3][0];
	u = v[3][0] ^ v[3][1] ^ v[3][2] ^ v[3][3];
	v[3][0] = v[3][0] ^ u ^ XT(v[3][0] ^ v[3][1]);
	v[3][1] = v[3][1] ^ u ^ XT(v[3][1] ^ v[3][2]);
	v[3][2] = v[3][2] ^ u ^ XT(v[3][2] ^ v[3][3]);
	v[3][3] = v[3][3] ^ u ^ XT(v[3][3] ^ t);

	s[0] = v[0][0];
	s[1] = v[0][1];
	s[2] = v[0][2];
	s[3] = v[0][3];
	s[4] = v[1][0];
	s[5] = v[1][1];
	s[6] = v[1][2];
	s[7] = v[1][3];
	s[8] = v[2][0];
	s[9] = v[2][1];
	s[10] = v[2][2];
	s[11] = v[2][3];
	s[12] = v[3][0];
	s[13] = v[3][1];
	s[14] = v[3][2];
	s[15] = v[3][3];
}

__device__ __forceinline__ void aesenc_n1(unsigned char *s, uint32_t *sharedMemory1)
{
	uint32_t  t, u;
	uint32_t v[4][4];

	v[0][0] = ((uint8_t*)&sharedMemory1[0])[s[ 0]];
	v[3][1] = ((uint8_t*)&sharedMemory1[0])[s[ 1]];
	v[2][2] = ((uint8_t*)&sharedMemory1[0])[s[ 2]];
	v[1][3] = ((uint8_t*)&sharedMemory1[0])[s[ 3]];
	v[1][0] = 0x63;
	v[0][1] = 0x63;
	v[3][2] = 0x63;
	v[2][3] = 0x63;
	v[2][0] = 0x63;
	v[1][1] = 0x63;
	v[0][2] = 0x63;
	v[3][3] = 0x63;
	v[3][0] = 0x63;
	v[2][1] = 0x63;
	v[1][2] = 0x63;
	v[0][3] = 0x63;

	t = v[0][0];
	u = v[0][0] ^ v[0][1] ^ v[0][2] ^ v[0][3];
	v[0][0] = v[0][0] ^ u ^ XT(v[0][0] ^ v[0][1]);
	v[0][1] = v[0][1] ^ u ^ XT(v[0][1] ^ v[0][2]);
	v[0][2] = v[0][2] ^ u ^ XT(v[0][2] ^ v[0][3]);
	v[0][3] = v[0][3] ^ u ^ XT(v[0][3] ^ t);
	t = v[1][0];
	u = v[1][0] ^ v[1][1] ^ v[1][2] ^ v[1][3];
	v[1][0] = v[1][0] ^ u ^ XT(v[1][0] ^ v[1][1]);
	v[1][1] = v[1][1] ^ u ^ XT(v[1][1] ^ v[1][2]);
	v[1][2] = v[1][2] ^ u ^ XT(v[1][2] ^ v[1][3]);
	v[1][3] = v[1][3] ^ u ^ XT(v[1][3] ^ t);
	t = v[2][0];
	u = v[2][0] ^ v[2][1] ^ v[2][2] ^ v[2][3];
	v[2][0] = v[2][0] ^ u ^ XT(v[2][0] ^ v[2][1]);
	v[2][1] = v[2][1] ^ u ^ XT(v[2][1] ^ v[2][2]);
	v[2][2] = v[2][2] ^ u ^ XT(v[2][2] ^ v[2][3]);
	v[2][3] = v[2][3] ^ u ^ XT(v[2][3] ^ t);
	t = v[3][0];
	u = v[3][0] ^ v[3][1] ^ v[3][2] ^ v[3][3];
	v[3][0] = v[3][0] ^ u ^ XT(v[3][0] ^ v[3][1]);
	v[3][1] = v[3][1] ^ u ^ XT(v[3][1] ^ v[3][2]);
	v[3][2] = v[3][2] ^ u ^ XT(v[3][2] ^ v[3][3]);
	v[3][3] = v[3][3] ^ u ^ XT(v[3][3] ^ t);

	s[0] = v[0][0];
	s[1] = v[0][1];
	s[2] = v[0][2];
	s[3] = v[0][3];
	s[4] = v[1][0];
	s[5] = v[1][1];
	s[6] = v[1][2];
	s[7] = v[1][3];
	s[8] = v[2][0];
	s[9] = v[2][1];
	s[10] = v[2][2];
	s[11] = v[2][3];
	s[12] = v[3][0];
	s[13] = v[3][1];
	s[14] = v[3][2];
	s[15] = v[3][3];
}

__device__ __forceinline__ void aesenc_s1(unsigned char *s, uint32_t *sharedMemory1)
{
	uint32_t  t, u;
	uint32_t v[4][4];

	v[0][0] = ((uint8_t*)&sharedMemory1[0])[s[ 0]];
	v[3][1] = ((uint8_t*)&sharedMemory1[0])[s[ 1]];
	v[2][2] = ((uint8_t*)&sharedMemory1[0])[s[ 2]];
	v[1][3] = ((uint8_t*)&sharedMemory1[0])[s[ 3]];
	v[1][0] = ((uint8_t*)&sharedMemory1[0])[s[ 4]];
	v[0][1] = ((uint8_t*)&sharedMemory1[0])[s[ 5]];
	v[3][2] = ((uint8_t*)&sharedMemory1[0])[s[ 6]];
	v[2][3] = ((uint8_t*)&sharedMemory1[0])[s[ 7]];
	v[2][0] = ((uint8_t*)&sharedMemory1[0])[s[ 8]];
	v[1][1] = ((uint8_t*)&sharedMemory1[0])[s[ 9]];
	v[0][2] = ((uint8_t*)&sharedMemory1[0])[s[10]];
	v[3][3] = ((uint8_t*)&sharedMemory1[0])[s[11]];
	v[3][0] = 0x0f;
	v[2][1] = 0x0f;
	v[1][2] = 0x0f;
	v[0][3] = 0x0f;

	t = v[0][0];
	u = v[0][0] ^ v[0][1] ^ v[0][2] ^ v[0][3];
	v[0][0] = v[0][0] ^ u ^ XT(v[0][0] ^ v[0][1]);
	v[0][1] = v[0][1] ^ u ^ XT(v[0][1] ^ v[0][2]);
	v[0][2] = v[0][2] ^ u ^ XT(v[0][2] ^ v[0][3]);
	v[0][3] = v[0][3] ^ u ^ XT(v[0][3] ^ t);
	t = v[1][0];
	u = v[1][0] ^ v[1][1] ^ v[1][2] ^ v[1][3];
	v[1][0] = v[1][0] ^ u ^ XT(v[1][0] ^ v[1][1]);
	v[1][1] = v[1][1] ^ u ^ XT(v[1][1] ^ v[1][2]);
	v[1][2] = v[1][2] ^ u ^ XT(v[1][2] ^ v[1][3]);
	v[1][3] = v[1][3] ^ u ^ XT(v[1][3] ^ t);
	t = v[2][0];
	u = v[2][0] ^ v[2][1] ^ v[2][2] ^ v[2][3];
	v[2][0] = v[2][0] ^ u ^ XT(v[2][0] ^ v[2][1]);
	v[2][1] = v[2][1] ^ u ^ XT(v[2][1] ^ v[2][2]);
	v[2][2] = v[2][2] ^ u ^ XT(v[2][2] ^ v[2][3]);
	v[2][3] = v[2][3] ^ u ^ XT(v[2][3] ^ t);
	t = v[3][0];
	u = v[3][0] ^ v[3][1] ^ v[3][2] ^ v[3][3];
	v[3][0] = v[3][0] ^ u ^ XT(v[3][0] ^ v[3][1]);
	v[3][1] = v[3][1] ^ u ^ XT(v[3][1] ^ v[3][2]);
	v[3][2] = v[3][2] ^ u ^ XT(v[3][2] ^ v[3][3]);
	v[3][3] = v[3][3] ^ u ^ XT(v[3][3] ^ t);

	s[0] = v[0][0];
	s[1] = v[0][1];
	s[2] = v[0][2];
	s[3] = v[0][3];
	s[4] = v[1][0];
	s[5] = v[1][1];
	s[6] = v[1][2];
	s[7] = v[1][3];
	s[8] = v[2][0];
	s[9] = v[2][1];
	s[10] = v[2][2];
	s[11] = v[2][3];
	s[12] = v[3][0];
	s[13] = v[3][1];
	s[14] = v[3][2];
	s[15] = v[3][3];
}


__device__ __forceinline__ void aesenc_s2(unsigned char *s, uint32_t *sharedMemory1)
{
	uint32_t  t, u;
	uint32_t v[4][4];

	v[0][0] = ((uint8_t*)&sharedMemory1[0])[s[ 0]];
	v[3][1] = ((uint8_t*)&sharedMemory1[0])[s[ 1]];
	v[2][2] = ((uint8_t*)&sharedMemory1[0])[s[ 2]];
	v[1][3] = ((uint8_t*)&sharedMemory1[0])[s[ 3]];
	v[1][0] = ((uint8_t*)&sharedMemory1[0])[s[ 4]];
	v[0][1] = ((uint8_t*)&sharedMemory1[0])[s[ 5]];
	v[3][2] = ((uint8_t*)&sharedMemory1[0])[s[ 6]];
	v[2][3] = ((uint8_t*)&sharedMemory1[0])[s[ 7]];
	v[2][0] = 0x0f;
	v[1][1] = 0x0f;
	v[0][2] = 0x0f;
	v[3][3] = 0x0f;
	v[3][0] = ((uint8_t*)&sharedMemory1[0])[s[12]];
	v[2][1] = ((uint8_t*)&sharedMemory1[0])[s[13]];
	v[1][2] = ((uint8_t*)&sharedMemory1[0])[s[14]];
	v[0][3] = ((uint8_t*)&sharedMemory1[0])[s[15]];

	t = v[0][0];
	u = v[0][0] ^ v[0][1] ^ v[0][2] ^ v[0][3];
	v[0][0] = v[0][0] ^ u ^ XT(v[0][0] ^ v[0][1]);
	v[0][1] = v[0][1] ^ u ^ XT(v[0][1] ^ v[0][2]);
	v[0][2] = v[0][2] ^ u ^ XT(v[0][2] ^ v[0][3]);
	v[0][3] = v[0][3] ^ u ^ XT(v[0][3] ^ t);
	t = v[1][0];
	u = v[1][0] ^ v[1][1] ^ v[1][2] ^ v[1][3];
	v[1][0] = v[1][0] ^ u ^ XT(v[1][0] ^ v[1][1]);
	v[1][1] = v[1][1] ^ u ^ XT(v[1][1] ^ v[1][2]);
	v[1][2] = v[1][2] ^ u ^ XT(v[1][2] ^ v[1][3]);
	v[1][3] = v[1][3] ^ u ^ XT(v[1][3] ^ t);
	t = v[2][0];
	u = v[2][0] ^ v[2][1] ^ v[2][2] ^ v[2][3];
	v[2][0] = v[2][0] ^ u ^ XT(v[2][0] ^ v[2][1]);
	v[2][1] = v[2][1] ^ u ^ XT(v[2][1] ^ v[2][2]);
	v[2][2] = v[2][2] ^ u ^ XT(v[2][2] ^ v[2][3]);
	v[2][3] = v[2][3] ^ u ^ XT(v[2][3] ^ t);
	t = v[3][0];
	u = v[3][0] ^ v[3][1] ^ v[3][2] ^ v[3][3];
	v[3][0] = v[3][0] ^ u ^ XT(v[3][0] ^ v[3][1]);
	v[3][1] = v[3][1] ^ u ^ XT(v[3][1] ^ v[3][2]);
	v[3][2] = v[3][2] ^ u ^ XT(v[3][2] ^ v[3][3]);
	v[3][3] = v[3][3] ^ u ^ XT(v[3][3] ^ t);

	s[0] = v[0][0];
	s[1] = v[0][1];
	s[2] = v[0][2];
	s[3] = v[0][3];
	s[4] = v[1][0];
	s[5] = v[1][1];
	s[6] = v[1][2];
	s[7] = v[1][3];
	s[8] = v[2][0];
	s[9] = v[2][1];
	s[10] = v[2][2];
	s[11] = v[2][3];
	s[12] = v[3][0];
	s[13] = v[3][1];
	s[14] = v[3][2];
	s[15] = v[3][3];
}


// Simulate _mm_unpacklo_epi32
__device__ __forceinline__ void unpacklo32(unsigned char *t, unsigned char *a, unsigned char *b)
{
	uint32_t* t32 = (uint32_t*)t;
	uint32_t* a32 = (uint32_t*)a;
	uint32_t* b32 = (uint32_t*)b;
	t32[0] = a32[0];
	t32[2] = a32[1];
	t32[1] = b32[0];
	t32[3] = b32[1];
}

// Simulate _mm_unpackhi_epi32
__device__ __forceinline__ void unpackhi32(unsigned char *t, unsigned char *a, unsigned char *b)
{
	uint32_t* t32 = (uint32_t*)t;
	uint32_t* a32 = (uint32_t*)a;
	uint32_t* b32 = (uint32_t*)b;
	t32[0] = a32[2];
	t32[1] = b32[2];
	t32[2] = a32[3];
	t32[3] = b32[3];
}

// Simulate _mm_unpacklo_epi32
__device__ __forceinline__ void unpacklo32s(unsigned char *a, unsigned char *b)
{
	uint32_t* a32 = (uint32_t*)a;
	uint32_t* b32 = (uint32_t*)b;
	a32[2] = a32[1];
	a32[1] = b32[0];
	a32[3] = b32[1];
}


__global__ __launch_bounds__(256, 2)
void verus_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce)
{
	uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	int i, j;
	unsigned char s[64], tmp[16];
	uint32_t nounce = startNonce + thread;

	__shared__ uint32_t sharedMemory1[256];
	//if (threadIdx.x < 64)
	sharedMemory1[threadIdx.x] = __ldg(&sbox[threadIdx.x]);

	#pragma unroll
	for (int i = 0; i < 16; i++) s[i] = blockhash_half[i];
	#pragma unroll
	for (int i = 0; i < 16; i++) tmp[i] = blockhash_half[i + 16];
	((uint32_t *)&s)[8] = nounce;
	__syncthreads();

	aesenc_n1(s + 32, sharedMemory1);
	aesenc(s + 32, sharedMemory1);

	//unpacklo32(tmp, s, s + 16);
	//unpackhi32(s, s, s + 16);

	//unpacklo32(s + 16, s + 32, s + 48);
	((uint32_t*)s)[ 4] = ((uint32_t*)s)[ 8];
	((uint32_t*)s)[ 6] = ((uint32_t*)s)[ 9];

	//unpackhi32(s + 32, s + 32, s + 48);
	((uint32_t*)s)[ 8] = ((uint32_t*)s)[10];
	((uint32_t*)s)[10] = ((uint32_t*)s)[11];

	unpacklo32(s + 48, s, s + 32);
	unpackhi32(s, s, s + 32);
	unpackhi32(s + 32, s + 16, tmp);
	unpacklo32s(s + 16, tmp);

	aesenc_s1(s, sharedMemory1);
	aesenc_s2(s + 16, sharedMemory1);
	aesenc_s2(s + 32, sharedMemory1);
	aesenc_s1(s + 48, sharedMemory1);
	aesenc(s, sharedMemory1);
	aesenc(s + 16, sharedMemory1);
	aesenc(s + 32, sharedMemory1);
	aesenc(s + 48, sharedMemory1);
	unpacklo32(tmp, s, s + 16);
	unpackhi32(s, s, s + 16);
	unpacklo32(s + 16, s + 32, s + 48);
	unpackhi32(s + 32, s + 32, s + 48);
	unpacklo32(s + 48, s, s + 32);
	unpackhi32(s, s, s + 32);
	unpackhi32(s + 32, s + 16, tmp);
	unpacklo32s(s + 16, tmp);

	#pragma nounroll
	for (i = 2; i < 3; ++i) {
		#pragma unroll
		for (j = 0; j < 2; ++j) {
			aesenc(s, sharedMemory1);
			aesenc(s + 16, sharedMemory1);
			aesenc(s + 32, sharedMemory1);
			aesenc(s + 48, sharedMemory1);
		}
		unpacklo32(tmp, s, s + 16);
		unpackhi32(s, s, s + 16);
		unpacklo32(s + 16, s + 32, s + 48);
		unpackhi32(s + 32, s + 32, s + 48);
		unpacklo32(s + 48, s, s + 32);
		unpackhi32(s, s, s + 32);
		unpackhi32(s + 32, s + 16, tmp);
		unpacklo32s(s + 16, tmp);
	}

	aesenc(s, sharedMemory1);
	aesenc(s + 16, sharedMemory1);
	aesenc(s + 32, sharedMemory1);
	aesenc(s + 48, sharedMemory1);
	aesenc(s, sharedMemory1);
	aesenc(s + 16, sharedMemory1);
	aesenc(s + 32, sharedMemory1);
	aesenc(s + 48, sharedMemory1);
	unpacklo32(tmp, s, s + 16);
	unpackhi32(s, s, s + 16);
	unpacklo32(s + 16, s + 32, s + 48);
	unpackhi32(s + 32, s + 32, s + 48);
	unpacklo32(s + 48, s, s + 32);
	unpackhi32(s, s, s + 32);
	unpackhi32(s + 32, s + 16, tmp);
	unpacklo32s(s + 16, tmp);

	aesenc(s, sharedMemory1);
	aesenc(s + 16, sharedMemory1);
	aesenc(s + 32, sharedMemory1);
	aesenc(s + 48, sharedMemory1);
	aesenc(s, sharedMemory1);
	aesenc(s + 16, sharedMemory1);
	aesenc(s + 32, sharedMemory1);
	aesenc(s + 48, sharedMemory1);
	unpackhi32(s, s, s + 16);
	unpackhi32(s + 32, s + 32, s + 48);
	unpacklo32(s + 48, s, s + 32);

	//if (((uint64_t*)&s[48])[0] < ((uint64_t*)&ptarget)[3]) resNonce[0] = nounce;
	if (((uint32_t*)&s[52])[0] <= ptarget) {
		uint32_t tmp = atomicExch(&resNonce[0], nounce);
		if (tmp != UINT32_MAX) resNonce[1] = tmp;
	}
}
