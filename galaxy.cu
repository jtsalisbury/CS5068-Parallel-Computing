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
void parse_input(std::string filename) {

}

// physics helper functions
__device__ float compute_force(float m1, float m2, float dist) {
	return CONST_GRAVITY * (m1 * m2/(dist * dist));
}

__device__ float compute_acceleration(float mass, float force) {
	return force/mass;
}

__device__ float compute_distance(float pos1, float pos2) {
	return abs(pos1 - pos2);
}

__device__ float compute_updated_pos(float pos, float vel, float acceleration) {
	return pos + (vel*CONST_TIME) +(.5 * acceleration * CONST_TIME * CONST_TIME);
}

__device__ float compute_updated_velocity(float vel, float acceleration) {
    return vel + (acceleration*CONST_TIME); 
}

// TODO: only performs x-component work so far, need to add y-component
__global__ void calculate_all_forces(point * sim_points_in, float * total_force) {
	// get the ids for each block and thread
	int k = blockIdx.x;
	int i = threadIdx.x;

    if (k == i) {
        // there is no force exerted on an object by the object itself
        total_force[k * CONST_MAX_NUM_POINTS + i] = 0;
    }
    else {
        // read the position and mass of the object
        float x_pos1 = sim_points_in[k].x_pos;
        float m1 = sim_points_in[k].mass;

        // obtain the positions of the 2nd object
        float x_pos2  = sim_points_in[i].x_pos;

        // calculate the distance between the 2 objects
        float dist = compute_distance(x_pos1, x_pos2);

        // obtain the masses of the 2 objects 
        float m2 = sim_points_in[i].mass;
                
        // calculate the force between the 2 objects
        float force_to_add = compute_force(m1, m2, dist);
                
        // add the force to the total force matrix
        total_force[k * CONST_MAX_NUM_POINTS + i] = force_to_add;
    }
	
}

__global__ void update_sim_points(float * total_force_reduced, point * sim_points_in, point * sim_points_out) {
    // get the ids for each block and thread
    int k = blockIdx.x;

    // get initial position, velocity, and mass
    float x_pos1 = sim_points_in[k].x_pos;
    float x_vel1 = sim_points_in[k].x_vel;
    float m1 = sim_points_in[k].mass;
    
    // update the acceleration
    float acceleration = compute_acceleration(m1, total_force_reduced[k]);

    // update the velocity
    float updated_vel = compute_updated_velocity(x_vel1, acceleration);

    // update the position
    float updated_pos = compute_updated_pos(x_pos1, x_vel1, acceleration);

    // store updated position and velocity
    sim_points_out[k].mass = m1;
    sim_points_out[k].x_vel = updated_vel;
    sim_points_out[k].x_pos = updated_pos;
}

// TODO 
// animation stuff

// TODO
// main function to perform physic operations 
int main() {
    // define array to insert information from the csv file into 
    // CPU side
    point sim_points_in[CONST_MAX_NUM_POINTS];
    point sim_points_out[CONST_MAX_NUM_POINTS];
    float total_force[CONST_MAX_NUM_POINTS * CONST_MAX_NUM_POINTS];
    float total_force_reduced[CONST_MAX_NUM_POINTS];

    // GPU side pointers
    point * dev_sim_points_in;
    point * dev_sim_points_out;
    float * dev_total_force;
    float * dev_total_force_reduced;

    // allocate memory on GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_sim_points_in, CONST_MAX_NUM_POINTS * sizeof(point) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_sim_points_out, CONST_MAX_NUM_POINTS * sizeof(point) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_total_force, CONST_MAX_NUM_POINTS * CONST_MAX_NUM_POINTS * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_total_force_reduced, CONST_MAX_NUM_POINTS * sizeof(float) ) );
    

    // fill CPU side with data
    // TODO

    while(1) {
        // copy simulation point array to GPU
        HANDLE_ERROR( cudaMemcpy( dev_sim_points_in, sim_points_in, CONST_MAX_NUM_POINTS * sizeof(point),
        cudaMemcpyHostToDevice ) );

        // run kernel - calculate all forces on every body in the simulation
        calculate_all_forces<<<CONST_MAX_NUM_POINTS, CONST_MAX_NUM_POINTS>>>(dev_sim_points_in, dev_total_force);

        // copy the total force matrix to CPU
        HANDLE_ERROR( cudaMemcpy( total_force, dev_total_force, CONST_MAX_NUM_POINTS * CONST_MAX_NUM_POINTS * sizeof(float),
        cudaMemcpyDeviceToHost ) );

        // perform a reduction
        for (int k = 0; k < CONST_MAX_NUM_POINTS; k++) {
            // reset the running sum to 0
            float running_sum = 0;
            for (int i = 0; i < CONST_MAX_NUM_POINTS; i++) {
                // add together all forces from every object
                running_sum += total_force[k * CONST_MAX_NUM_POINTS + i];
            }
            // store the resulting total force in a new array
            total_force_reduced[k] = running_sum;
        }

        // copy the total force array to the GPU
        HANDLE_ERROR( cudaMemcpy( dev_total_force_reduced, total_force_reduced, CONST_MAX_NUM_POINTS * sizeof(float),
        cudaMemcpyHostToDevice ) );

        // run kernel - calculate updated position and velocity for the object
        update_sim_points<<<CONST_MAX_NUM_POINTS, 1>>>(dev_total_force_reduced, dev_sim_points_in, dev_sim_points_out);

        // copy simulation point array to CPU
        HANDLE_ERROR( cudaMemcpy( sim_points_out, dev_sim_points_out, CONST_MAX_NUM_POINTS * sizeof(point),
        cudaMemcpyDeviceToHost ) );

        // copy the output data to the input data
        memcpy(&sim_points_in, &sim_points_out, CONST_MAX_NUM_POINTS * sizeof(point));
    }

    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_sim_points_in ) );
    HANDLE_ERROR( cudaFree( dev_sim_points_out ) );
    HANDLE_ERROR( cudaFree( dev_total_force ) );
    HANDLE_ERROR( cudaFree( dev_total_force_reduced ) );

    return 0;
} 
