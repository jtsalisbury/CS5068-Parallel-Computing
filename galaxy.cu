/* 
N-body simulation of a galaxy 
Authors: JT Salisbury, Sydney O'Connor, Kyle Bush, Caroline Northrop 
Parallel Computing 6068
*/

#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include "../../../cuda_by_example/common/book.h"

#define CONST_GRAVITY 0.00000000006673
#define CONST_TIME 1
#define CONST_MAX_NUM_POINTS 50

// define structure containing all attributes of a simulation point
struct point {
	int id;
	float x_pos;
	float x_vel;
    //float y_pos;
    //float y_vel;
    float mass;
};

//TODO
// parse csv file 
/* assumptions:
variable names: id, x_pos, x_vel, y_pos, y_vel, mass
output: vector (in order above) of the elements 
*/
void parse_input(std::string filename){

}

// physics helper functions
__device__ float compute_force(float m1, float m2, float dist){
	return CONST_GRAVITY * (m1 * m2/(dist * dist));
}

__device__ float compute_acceleration(float mass, float force){
	return force/mass;
}

__device__ float compute_distance(float pos1, float pos2){
	return abs(pos1 - pos2);
}

__device__ float compute_updated_pos(float pos, float vel, float acceleration){
	return pos + (vel*CONST_TIME) +(.5 * acceleration * CONST_TIME * CONST_TIME);
}

__device__ float compute_updated_velocity(float vel, float acceleration){
    return vel + (acceleration*CONST_TIME); 
}

// TODO: only performs x-component work so far, need to add y-component
__global__ void perform_iteration(point * sim_points_in, point * sim_points_out){
	// get the ids for each block and thread
	int k = blockIdx.x;
	int i = threadIdx.x;

    // TODO: I don't think this works as expected. We'll need to keep looking into it
	// TODO: create this variable in shared memory instead to improve performance
	// create a variable to store the total force on the object 
	float total_force = 0;

    // read the position and mass of the object
    float x_pos1 = sim_points_in[k].x_pos;
    float x_vel1 = sim_points_in[k].x_vel;
    float m1 = sim_points_in[k].mass;

    // obtain the positions of the 2nd object
    float x_pos2  = sim_points_in[i].x_pos;

    // calculate the distance between the 2 objects
    float dist = compute_distance(x_pos1, x_pos2);

    // obtain the masses of the 2 objects 
    float m2 = sim_points_in[i].mass;
            
    // calculate the force between the 2 objects
    float force_to_add = compute_force(m1, m2, dist);
            
    // update the running sum, using atomicadd to lock the bin 
    atomicAdd(&total_force, force_to_add);
    __syncthreads();

    // so now, we just need to use a single thread to complete the remaining calculations (multiple threads would just perform repetitive calculations
    if (k == 0){
        // update the acceleration
        float acceleration = compute_acceleration(m1, total_force);

        // update the velocity
        float updated_vel = compute_updated_velocity(x_vel1, acceleration);

        // update the position
        float updated_pos = compute_updated_pos(x_pos1, x_vel1, acceleration);

        // store updated position and velocity
        sim_points_out[k].mass = sim_points_in[k].mass;
        sim_points_out[k].x_vel = updated_vel;
        sim_points_out[k].x_pos = updated_pos;
    }
}



// TODO 
// animation stuff





// TODO
// main function to perform physic operations 
int main(){
    // define array to insert information from the csv file into 
    // CPU side
    point sim_points_in[CONST_MAX_NUM_POINTS];
    point sim_points_out[CONST_MAX_NUM_POINTS];

    // GPU side pointers
    point * dev_sim_points_in;
    point * dev_sim_points_out;

    // allocate memory on GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_sim_points_in, CONST_MAX_NUM_POINTS * sizeof(point) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_sim_points_out, CONST_MAX_NUM_POINTS * sizeof(point) ) );

    // fill CPU side with data
    // TODO

    while(1)
    {
        // copy simulation point array to GPU
        HANDLE_ERROR( cudaMemcpy( dev_sim_points_in, sim_points_in, CONST_MAX_NUM_POINTS * sizeof(point),
        cudaMemcpyHostToDevice ) );

        perform_iteration<<<CONST_MAX_NUM_POINTS, CONST_MAX_NUM_POINTS>>>(dev_sim_points_in, dev_sim_points_out);

        // copy the updated simulation points back to CPU
        HANDLE_ERROR( cudaMemcpy( sim_points_out, dev_sim_points_out, CONST_MAX_NUM_POINTS * sizeof(point),
        cudaMemcpyDeviceToHost ) );

        // copy the output data to the input data
        memcpy(&sim_points_in, &sim_points_out, CONST_MAX_NUM_POINTS * sizeof(point));
    }

    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_sim_points_in ) );
    HANDLE_ERROR( cudaFree( dev_sim_points_out ) );

    return 0;
} 
