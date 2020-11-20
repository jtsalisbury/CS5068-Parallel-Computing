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
#include "../../../cuda_by_example/common/cpu_anim.h"

#define CONST_GRAVITY 0.00000000006673
#define CONST_TIME 1
#define CONST_MAX_NUM_POINTS 50
#define DIM 1024
#define TIME_OFFSET 10

// define structure containing all attributes of a simulation point
struct point {
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

    point *dev_sim_points_in;
    point *dev_sim_points_out;
    float *dev_total_force;
    float *dev_total_force_reduced;

    point sim_points_in[CONST_MAX_NUM_POINTS];
    point sim_points_out[CONST_MAX_NUM_POINTS];
    float total_force[CONST_MAX_NUM_POINTS * CONST_MAX_NUM_POINTS];
    float total_force_reduced[CONST_MAX_NUM_POINTS];
};

//TODO
// parse csv file 
/* assumptions:
variable names: id, x_pos, x_vel, y_pos, y_vel, mass
output: vector (in order above) of the elements 
*/

// reference: http://www.cplusplus.com/forum/beginner/193916/
void print_points(point p){
    std::cout << p.id << " "<< p.x_pos << " " << p.x_vel << " " << p.y_pos << " " << p.y_vel << " " << p.mass << "\n";
}

void parse_input() {
    //read file
    std::ifstream data("particles.csv");
    if (!data.is_open())
    {
        exit(EXIT_FAILURE);
    }
    std::string str;
    // getline(data, str); // skip the first line

    std::vector<point> my_points;
    int id;
    float x_pos;
    float x_vel;
    float y_pos;
    float y_vel;
    float mass;
    char delimiter;

    data.ignore(1000, '\n'); //ignore first line
    while(data >> id >> delimiter >> x_pos >> delimiter >> x_vel >> delimiter >> y_pos >> delimiter >> y_vel >> delimiter >> mass){
        point p;
        p.id;
        p.x_pos;
        p.x_vel;
        p.y_pos;
        p.y_vel;
        p.mass;

        my_points.push_back(p); //this line is causing issues
    }

    std::cout << "ID" << " X_POS" << " \n";
    for(int x(0); x<my_points.size(); ++x){
        print_points(my_points.at(x)); 
    }
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

__global__ void update_sim_points(float * total_force_reduced, point * sim_points_in, point * sim_points_out, unsigned char * bitmap) {
    // get the ids for each block and thread
    int k = blockIdx.x;

    // get initial position, velocity, and mass
    float x_pos1 = sim_points_in[k].x_pos;
    float x_vel1 = sim_points_in[k].x_vel;
    float m1 = sim_points_in[k].mass;

    // placeholders
    float y_pos1 = 2.0f;//sim_points_in[k].y_pos;
    
    // update the acceleration
    float acceleration = compute_acceleration(m1, total_force_reduced[k]);

    // update the velocity
    float updated_vel = compute_updated_velocity(x_vel1, acceleration);

    // update the position
    float updated_pos_x = compute_updated_pos(x_pos1, x_vel1, acceleration);
    
    // placeholder
    float updated_pos_y = 2.0f;//compute_updated_pos(y_pos1, y_vel1, acceleration);

    // store updated position and velocity
    sim_points_out[k].mass = m1;
    sim_points_out[k].x_vel = updated_vel;
    sim_points_out[k].x_pos = updated_pos_x;
        
    // update the bitmap only if in range
    if (x_pos1 < DIM && y_pos1 < DIM) {
        int oldOffset = x_pos1 + y_pos1 * gridDim.x;
        bitmap[oldOffset*4 + 0] = 0;
        bitmap[oldOffset*4 + 1] = 0;
        bitmap[oldOffset*4 + 2] = 0;
        bitmap[oldOffset*4 + 3] = 0;
    }

    if (x_pos2 < DIM && y_pos2 < DIM) {
        int newOffset = updated_pos_x + updated_pos_y * gridDim.x;
        bitmap[newOffset*4 + 0] = 255;
        bitmap[newOffset*4 + 1] = 255;
        bitmap[newOffset*4 + 2] = 255;
        bitmap[newOffset*4 + 3] = 255;
    }
}

// animation stuff
void generate_frame(DataBlock *d, int ticks) {
    // Only perform updates every N ticks
    if (ticks % TIME_OFFSET != 0) {
        return;
    }

    // allocate memory on GPU

    // copy simulation point array to GPU
    HANDLE_ERROR( cudaMemcpy( d->dev_sim_points_in, d->sim_points_in, CONST_MAX_NUM_POINTS * sizeof(point),
    cudaMemcpyHostToDevice ) );

    // run kernel - calculate all forces on every body in the simulation
    calculate_all_forces<<<CONST_MAX_NUM_POINTS, CONST_MAX_NUM_POINTS>>>(d->dev_sim_points_in, d->dev_total_force);

    // copy the total force matrix to CPU
    HANDLE_ERROR( cudaMemcpy( d->total_force, d->dev_total_force, CONST_MAX_NUM_POINTS * CONST_MAX_NUM_POINTS * sizeof(float),
    cudaMemcpyDeviceToHost ) );

    // perform a reduction
    for (int k = 0; k < CONST_MAX_NUM_POINTS; k++) {
        // reset the running sum to 0
        float running_sum = 0;
        for (int i = 0; i < CONST_MAX_NUM_POINTS; i++) {
            // add together all forces from every object
            running_sum += (d->total_force)[k * CONST_MAX_NUM_POINTS + i];
        } 
        // store the resulting total force in a new array
        (d->total_force_reduced)[k] = running_sum;
    }

    // copy the total force array to the GPU
    HANDLE_ERROR( cudaMemcpy( d->dev_total_force_reduced, d->total_force_reduced, CONST_MAX_NUM_POINTS * sizeof(float),
    cudaMemcpyHostToDevice ) );

    // run kernel - calculate updated position and velocity for the object
    update_sim_points<<<CONST_MAX_NUM_POINTS, 1>>>(d->dev_total_force_reduced, d->dev_sim_points_in, d->dev_sim_points_out, d->dev_bitmap);

    // copy simulation point array to CPU
    HANDLE_ERROR( cudaMemcpy( d->sim_points_out, d->dev_sim_points_out, CONST_MAX_NUM_POINTS * sizeof(point),
    cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaMemcpy( d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost ) );

    // copy the output data to the input data
    memcpy(&(d->sim_points_in), &(d->sim_points_out), CONST_MAX_NUM_POINTS * sizeof(point));
}

void cleanup(DataBlock *d) {
    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( d->dev_sim_points_in ) );
    HANDLE_ERROR( cudaFree( d->dev_sim_points_out ) );
    HANDLE_ERROR( cudaFree( d->dev_total_force ) );
    HANDLE_ERROR( cudaFree( d->dev_total_force_reduced ) );

    HANDLE_ERROR( cudaFree( d->dev_bitmap ) ); 
}

// TODO
// main function to perform physic operations 
int main() {
    //Note: when parsing input, put it in the data.sim_points_in array

    DataBlock data;

    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_bitmap), bitmap.image_size() ) );

    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_sim_points_in), CONST_MAX_NUM_POINTS * sizeof(point) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_sim_points_out), CONST_MAX_NUM_POINTS * sizeof(point) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_total_force), CONST_MAX_NUM_POINTS * CONST_MAX_NUM_POINTS * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&(data.dev_total_force_reduced), CONST_MAX_NUM_POINTS * sizeof(float) ) );

    bitmap.anim_and_exit( (void (*)(void*,int))generate_frame, (void (*)(void*))cleanup );

    return 0;
} 