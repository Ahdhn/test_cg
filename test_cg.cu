#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdint>
#include <numeric>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;


//********************** CUDA_ERROR
inline void HandleError(cudaError_t err, const char *file, int line) {
	//Error handling micro, wrap it around function whenever possible
	if (err != cudaSuccess) {
		printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);

#ifdef _WIN32
		system("pause");
#else
		exit(EXIT_FAILURE);
#endif
	}
}
#define CUDA_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
//******************************************************************************


//********************** testing cg kernel 
__global__ void testing_cg_grid_sync(const uint32_t num_elements,
	uint32_t *d_arr){
	
	uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid < num_elements){

		uint32_t my_element = d_arr[tid];
				
		//to sync across the whole grid 
		cg::grid_group barrier = cg::this_grid();

		//to sync within a single block 
		//cg::thread_block barrier = cg::this_thread_block();

		//wait for all reads 
		barrier.sync();		

		uint32_t tar_id = num_elements - tid - 1;

		d_arr[tar_id] = my_element;
	}
	return;
}
//******************************************************************************


//********************** execute  
void execute_test(const int sm_count){
	
	//host array 
	//const uint32_t arr_size = 1 << 20; //1M 
	const uint32_t arr_size = 1680*80;
	uint32_t* h_arr = (uint32_t*)malloc(arr_size * sizeof(uint32_t));
	//with with sequential numbers
	std::iota(h_arr, h_arr + arr_size, 0);

	//device array 
	uint32_t* d_arr;
	CUDA_ERROR(cudaMalloc((void**)&d_arr, arr_size*sizeof(uint32_t)));
	CUDA_ERROR(cudaMemcpy(d_arr, h_arr, arr_size*sizeof(uint32_t), 
		cudaMemcpyHostToDevice));

	//launch config
	const int threads = 80;

	//following the same steps done in conjugateGradientMultiBlockCG.cu 
	//cuda sample to launch kernel that sync across grid 
	//https://github.com/NVIDIA/cuda-samples/blob/master/Samples/conjugateGradientMultiBlockCG/conjugateGradientMultiBlockCG.cu#L436

	int num_blocks_per_sm = 0;
	CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
		(void*)testing_cg_grid_sync, threads, 0));

	dim3 grid_dim(sm_count * num_blocks_per_sm, 1, 1), block_dim(threads, 1, 1);

	printf("\n Launching %d blcoks, each containing %d threads \n", grid_dim.x,
		block_dim.x);

	if(arr_size > grid_dim.x*block_dim.x){
         printf("\n The grid size (numBlocks*numThreads) is less than array size.\n");
         printf("This will result into mismatch error (incorrect output erro)\n");
         exit(EXIT_FAILURE);
    }

    if((int(grid_dim.x*block_dim.x) - int(arr_size)) / threads > 0 ){
    	printf("\n At least one block might not see the sync barrier. This will (probabily) result into the code never exits.\n");
    	exit(EXIT_FAILURE);
    }
		
	//argument passed to the kernel 	
	void *kernel_args[] = {		
		(void *)&arr_size,
	    (void *)&d_arr,};


	//finally launch the kernel 
	cudaLaunchCooperativeKernel((void*)testing_cg_grid_sync,
		grid_dim, block_dim, kernel_args);


	//make sure everything went okay
	CUDA_ERROR(cudaGetLastError());
	CUDA_ERROR(cudaDeviceSynchronize());
	

	//get results on the host 
	CUDA_ERROR(cudaMemcpy(h_arr, d_arr, arr_size*sizeof(uint32_t),
		cudaMemcpyDeviceToHost));

	//validate 
	for (uint32_t i = 0; i < arr_size; i++){
		if (h_arr[i] != arr_size - i - 1){
			printf("\n Result mismatch in h_arr[%u] = %u\n", i, h_arr[i]);
			exit(EXIT_FAILURE);
		}
	}	
}
//******************************************************************************

int main(int argc, char**argv) {

	//set to Titan V
	uint32_t device_id = 0;	
	cudaSetDevice(device_id);

	//get sm count 
	cudaDeviceProp devProp;	
	CUDA_ERROR(cudaGetDeviceProperties(&devProp, device_id));
	int sm_count = devProp.multiProcessorCount;
	
	//execute 
	execute_test(sm_count);

	printf("\n Mission accomplished \n");
	return 0;
}
