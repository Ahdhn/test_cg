#ifndef _SMOOTHING_
#define _SMOOTHING_

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

//********************** Testing CG Grid Sync 

__global__ void testing_cg_grid_sync(const uint32_t num_elements, 
	uint32_t *d_arr){
	
	uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid < num_elements){
		uint32_t my_element = d_arr[tid];
				
		//cg::grid_group barrier = cg::this_grid();
		cg::thread_block barrier = cg::this_thread_block();

		//wait for all reads 
		barrier.sync();

		uint32_t tar_id = num_elements - tid - 1;

		d_arr[tar_id] = my_element;
	}
	
}

//**************************************************************************	


#endif 