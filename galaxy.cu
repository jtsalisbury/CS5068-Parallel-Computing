/*
N-body simulation of a galaxy 
Authors: JT Salisbury, Sydney O'Connor, Kyle Bush, Caroline Northrop 
Parallel Computing 6068
*/

#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
#include "../../../cuda_by_example/common/book.h"
#include "../../../cuda_by_example/common/cpu_anim.h"

#define CONST_GRAVITY 0.00000000006673
#define CONST_TIME 1/8 // or 1/16
#define CONST_NUM_POINTS 4092
#define DIM 1024
#define MASS_SCALE 120000.0f
#define VELOCITY_SCALE 8.0f

// Note: potential bug if this runs for too long, some of the bodies' positions may overflow and cause the body to move to (0,0). This will adversely impact the other bodies movements

// define structure containing all attributes of a simulation point
struct Point {
    int id;
    float x_pos;
    float x_vel;
    float y_pos;
    float y_vel;
    float mass;
};

struct DataBlock {
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;

    Point *dev_sim_points_in;
    Point *dev_sim_points_out;
    float *dev_total_force_x;
    float *dev_total_force_reduced_x;
    float *dev_total_force_y;
    float *dev_total_force_reduced_y;

    Point sim_points_in[CONST_NUM_POINTS];
    Point sim_points_out[CONST_NUM_POINTS];
    float total_force_x[CONST_NUM_POINTS * CONST_NUM_POINTS];
    float total_force_y[CONST_NUM_POINTS * CONST_NUM_POINTS];
    float total_force_reduced_x[CONST_NUM_POINTS];
    float total_force_reduced_y[CONST_NUM_POINTS];
};

void parse_input(std::string path, Point * sim_points) {
    //read file
    std::ifstream data(path); 

    if (!data.is_open())
    {
        exit(EXIT_FAILURE);
    }

    int id;
    float x_pos;
    float x_vel;
    float y_pos;
    float y_vel;
    float mass;
    char delimiter;

    int counter = 0;

    data.ignore(1000, '\n'); //ignore first line

    // While we still have lines and haven't loaded enough bodies
    while(data >> id >> delimiter >> x_pos >> delimiter >> x_vel >> delimiter >> y_pos >> delimiter >> y_vel >> delimiter >> mass && counter < CONST_NUM_POINTS){

	// Scale the position (to try and center our galaxy, since 0, 0 is the bottom left not the middle), velocites and mass. Positions are already normalized to our 0 - 1023 scale
        Point p;
        p.id = id;
        p.x_pos = x_pos / 5 + 400;
        p.x_vel = x_vel * VELOCITY_SCALE;
        p.y_pos = y_pos / 5 + 300;
        p.y_vel = y_vel * VELOCITY_SCALE;
        p.mass = mass * MASS_SCALE;

        sim_points[counter] = p;

        counter++;
    }    

    data.close();
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
	return ((pos + (vel*CONST_TIME) +(.5 * acceleration * CONST_TIME * CONST_TIME)));
}

__device__ float compute_updated_velocity(float vel, float acceleration) {
    return vel + (acceleration*CONST_TIME); 
}

// TODO: only performs x-component work so far, need to add y-component
__global__ void calculate_all_forces(Point * sim_points_in, float * total_force_x, float * total_force_y) {
	// get the ids for each block and thread
    int k = blockIdx.x;
    int x_or_y = blockIdx.y;
    int i = threadIdx.x;

    if (x_or_y == 0) {
        // x-component logic
        if (k == i) {
            // there is no force exerted on an object by the object itself
            total_force_x[k * CONST_NUM_POINTS + i] = 0;
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
            total_force_x[k * CONST_NUM_POINTS + i] = force_to_add;
        }
    }
    else {
        // y-component logic
        if (k == i) {
            // there is no force exerted on an object by the object itself
            total_force_y[k * CONST_NUM_POINTS + i] = 0;
        }
        else {
            // read the position and mass of the object
            float y_pos1 = sim_points_in[k].y_pos;
            float m1 = sim_points_in[k].mass;
    
            // obtain the positions of the 2nd object
            float y_pos2  = sim_points_in[i].y_pos;
    
            // calculate the distance between the 2 objects
            float dist = compute_distance(y_pos1, y_pos2);
    
            // obtain the masses of the 2 objects 
            float m2 = sim_points_in[i].mass;
                    
            // calculate the force between the 2 objects
            float force_to_add = compute_force(m1, m2, dist);
                    
            // add the force to the total force matrix
            total_force_y[k * CONST_NUM_POINTS + i] = force_to_add;
        }
    }
	
}

__global__ void update_sim_points(float * total_force_reduced_x, float * total_force_reduced_y, 
                                  Point * sim_points_in, Point * sim_points_out) {
    // get the ids for each block
    int k = blockIdx.x;
    int x_or_y = blockIdx.y;

    if (x_or_y == 0) {
        // x-component logic
        // get initial position, velocity, and mass
        float x_pos1 = sim_points_in[k].x_pos;
        float x_vel1 = sim_points_in[k].x_vel;
        float m1 = sim_points_in[k].mass;
        
        // update the acceleration
        float acceleration_x = compute_acceleration(m1, total_force_reduced_x[k]);

        // update the velocity
        float updated_vel_x = compute_updated_velocity(x_vel1, acceleration_x);

        // update the position
        float updated_pos_x = compute_updated_pos(x_pos1, x_vel1, acceleration_x);

        // store updated position, velocity, and mass
        sim_points_out[k].mass = m1;
        sim_points_out[k].x_vel = updated_vel_x;
        sim_points_out[k].x_pos = updated_pos_x;
    } else {
        // y-component logic
        // get initial position, velocity, and mass
        float y_pos1 = sim_points_in[k].y_pos;
        float y_vel1 = sim_points_in[k].y_vel;
        float m1 = sim_points_in[k].mass;
        
        // update the acceleration
        float acceleration_y = compute_acceleration(m1, total_force_reduced_y[k]);

        // update the velocity
        float updated_vel_y = compute_updated_velocity(y_vel1, acceleration_y);

        // update the position
        float updated_pos_y = compute_updated_pos(y_pos1, y_vel1, acceleration_y);

        // store updated position and velocity
        sim_points_out[k].y_vel = updated_vel_y;
        sim_points_out[k].y_pos = updated_pos_y;
    }
}

__device__ void updatePointColor(int x, int y, unsigned char* bitmap, int col) {
    // Ensure we are in bounds
    if (x < 0 || y < 0 || x > DIM - 1 || y > DIM - 1) {
        return;
    }

    // Update the bitmap
    int offset = x + y * DIM;
    bitmap[offset*4 + 0] = col;
    bitmap[offset*4 + 1] = col;
    bitmap[offset*4 + 2] = col;
    bitmap[offset*4 + 3] = col;
}

__global__ void update_bitmap(Point * sim_points_in, Point * sim_points_out, unsigned char * bitmap)
{    
    // get the ids for each block
    int k = blockIdx.x;

    // get the initial and final positions of each object
    int x_pos1 = round(sim_points_in[k].x_pos);
    int y_pos1 = round(sim_points_in[k].y_pos);
    int updated_pos_x = round(sim_points_out[k].x_pos);
    int updated_pos_y = round(sim_points_out[k].y_pos);

    __syncthreads();

    // Update the bitmap to our new body position
    updatePointColor(x_pos1, y_pos1, bitmap, 0);
    updatePointColor(updated_pos_x, updated_pos_y, bitmap, 255);
}

// animation stuff
void generate_frame(DataBlock *d, int ticks) {
    // copy simulation point array to GPU
    HANDLE_ERROR( cudaMemcpy( d->dev_sim_points_in, d->sim_points_in, CONST_NUM_POINTS * sizeof(Point),
    cudaMemcpyHostToDevice ) );

    // Allocating enough blocks for each object's x- and y-components
    dim3 grid(CONST_NUM_POINTS, 2);

    // run kernel - calculate all forces on every body in the simulation
    calculate_all_forces<<<grid, CONST_NUM_POINTS>>>(d->dev_sim_points_in, d->dev_total_force_x, d->dev_total_force_y);

    // copy the total force matrices to CPU
    HANDLE_ERROR( cudaMemcpy( d->total_force_x, d->dev_total_force_x, CONST_NUM_POINTS * CONST_NUM_POINTS * sizeof(float),
    cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( d->total_force_y, d->dev_total_force_y, CONST_NUM_POINTS * CONST_NUM_POINTS * sizeof(float),
    cudaMemcpyDeviceToHost ) );

    // perform reductions
    for (int k = 0; k < CONST_NUM_POINTS; k++) {
        // reset the running sum to 0
        float running_sum = 0;
        for (int i = 0; i < CONST_NUM_POINTS; i++) {
            // add together all forces from every object
            running_sum += (d->total_force_x)[k * CONST_NUM_POINTS + i];
        } 
        // store the resulting total force in a new array
        (d->total_force_reduced_x)[k] = running_sum;
    }
    for (int k = 0; k < CONST_NUM_POINTS; k++) {
        // reset the running sum to 0
        float running_sum = 0;
        for (int i = 0; i < CONST_NUM_POINTS; i++) {
            // add together all forces from every object
            running_sum += (d->total_force_y)[k * CONST_NUM_POINTS + i];
        } 
        // store the resulting total force in a new array
        (d->total_force_reduced_y)[k] = running_sum;
    }

    // copy the total force arrays to the GPU
    HANDLE_ERROR( cudaMemcpy( d->dev_total_force_reduced_x, d->total_force_reduced_x, CONST_NUM_POINTS * sizeof(float),
    cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( d->dev_total_force_reduced_y, d->total_force_reduced_y, CONST_NUM_POINTS * sizeof(float),
    cudaMemcpyHostToDevice ) );

    // run kernel - calculate updated position and velocity for the object
    update_sim_points<<<grid, 1>>>(d->dev_total_force_reduced_x, d->dev_total_force_reduced_y, 
        d->dev_sim_points_in, d->dev_sim_points_out);

    // run kernel - update bitmap
    update_bitmap<<<CONST_NUM_POINTS, 1>>>(d->dev_sim_points_in, d->dev_sim_points_out, d->dev_bitmap);

    // copy simulation point array to CPU
    HANDLE_ERROR( cudaMemcpy( d->sim_points_out, d->dev_sim_points_out, CONST_NUM_POINTS * sizeof(Point),
    cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaMemcpy( d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost ) );

    // copy the output data to the input data
    memcpy(&(d->sim_points_in), &(d->sim_points_out), CONST_NUM_POINTS * sizeof(Point));
}

void cleanup(DataBlock *d) {
    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( d->dev_sim_points_in ) );
    HANDLE_ERROR( cudaFree( d->dev_sim_points_out ) );
    HANDLE_ERROR( cudaFree( d->dev_total_force_x ) );
    HANDLE_ERROR( cudaFree( d->dev_total_force_reduced_x ) );
    HANDLE_ERROR( cudaFree( d->dev_total_force_y ) );
    HANDLE_ERROR( cudaFree( d->dev_total_force_reduced_y ) );
    HANDLE_ERROR( cudaFree( d->dev_bitmap ) ); 
}

// TODO
// main function to perform physic operations 
int main() {

    // Initialize datablock
    DataBlock data;

    // initialize bitmap
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    // Load input data
    parse_input("bodies.csv", data.sim_points_in);

    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_sim_points_in), CONST_NUM_POINTS * sizeof(Point) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_sim_points_out), CONST_NUM_POINTS * sizeof(Point) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_total_force_x), CONST_NUM_POINTS * CONST_NUM_POINTS * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_total_force_reduced_x), CONST_NUM_POINTS * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_total_force_y), CONST_NUM_POINTS * CONST_NUM_POINTS * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_total_force_reduced_y), CONST_NUM_POINTS * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_bitmap), bitmap.image_size() ) );

    bitmap.anim_and_exit( (void (*)(void*,int))generate_frame, (void (*)(void*))cleanup );

    return 0;
} 
