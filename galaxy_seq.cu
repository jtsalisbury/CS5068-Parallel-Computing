/*
N-body simulation of a galaxy (sequential)
Authors: JT Salisbury, Sydney O'Connor, Kyle Bush, Caroline Northrop 
Parallel Computing 6068
*/

#include <cmath>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
#include "gputimer.h"
#include "../../../cuda_by_example/common/book.h"
#include "../../../cuda_by_example/common/cpu_anim.h"

#define CONST_GRAVITY 0.00000000006673
#define CONST_TIME 1/8 // or 1/16
#define CONST_NUM_POINTS 4092
#define DIM 1024
#define MASS_SCALE 120000.0f
#define VELOCITY_SCALE 8.0f
#define FILENAME_INPUT_POINTS "bodies.csv"
#define FILENAME_OUTPUT_TIMING_DATA "timing_data_sequential.csv"

// Note: potential bug if this runs for too long, some of the bodies' positions may overflow and cause the body to move to (0,0). This will adversely impact the other bodies movements

// set this to 1 to record timing data and 0 to turn off recording
#define RECORD_TIMING_DATA 1

#if RECORD_TIMING_DATA
// create output file stream (globally, to avoid constant opens/closes)
std::ofstream file_timing_data;

void create_output_timing_file(std::string path) {
    // create an output .csv file with the timing data
    file_timing_data.open(path, std::fstream::out | std::fstream::app);
    if (!file_timing_data.is_open())
    {
        exit(EXIT_FAILURE);
    }

    // add fields as first line of .csv
    file_timing_data << "calc_forces_ms,update_points_ms,update_bitmap_ms" << std::endl;
}
#endif

// define structure containing all attributes of a simulation point
struct Point {
    int id;
    float x_pos;
    float x_vel;
    float y_pos;
    float y_vel;
    float mass;
};

// define a structure for holding timing data
struct TimingData {
    float calc_forces_ms;
    float update_points_ms;
    float update_bitmap_ms;
};

struct DataBlock {
    CPUAnimBitmap *bitmap;

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
    
    // close the file
    data.close();

    data.close();
}

// physics helper functions
float compute_force(float m1, float m2, float dist) {
	if (dist * dist == 0) {
	    return 0;
	}

	return CONST_GRAVITY * (m1 * m2/(dist * dist));
}

float compute_acceleration(float mass, float force) {
	if (mass == 0) {
	    return 0;
	}

	return force/mass;
}

float compute_distance(float pos1, float pos2) {
	return abs(pos1 - pos2);
}

float compute_updated_pos(float pos, float vel, float acceleration) {
	return ((pos + (vel*CONST_TIME) +(.5 * acceleration * CONST_TIME * CONST_TIME)));
}

float compute_updated_velocity(float vel, float acceleration) {
    return vel + (acceleration*CONST_TIME); 
}

// TODO: only performs x-component work so far, need to add y-component
void calculate_all_forces(Point * sim_points_in, float * total_force_x, float * total_force_y) {

    for (int k = 0; k < CONST_NUM_POINTS; k++) {
        for (int i = 0; i < CONST_NUM_POINTS; i++) {
            // x-component logic
            if (k == i) {
                // there is no force exerted on an object by the object itself
                total_force_x[k * CONST_NUM_POINTS + i] = 0;
		continue;
	    }
            
            // read the position and mass of the object
            float x_pos1 = sim_points_in[k].x_pos;
            float xm1 = sim_points_in[k].mass;
        
            // obtain the positions of the 2nd object
            float x_pos2  = sim_points_in[i].x_pos;
        
            // calculate the distance between the 2 objects
            float xdist = compute_distance(x_pos1, x_pos2);
        
            // obtain the masses of the 2 objects 
            float xm2 = sim_points_in[i].mass;
                        
            // calculate the force between the 2 objects
            float xforce_to_add = compute_force(xm1, xm2, xdist);
                        
            // add the force to the total force matrix
            total_force_x[k * CONST_NUM_POINTS + i] = xforce_to_add;

	    // read the position and mass of the object
            float y_pos1 = sim_points_in[k].y_pos;
            float ym1 = sim_points_in[k].mass;
        
            // obtain the positions of the 2nd object
            float y_pos2  = sim_points_in[i].y_pos;
        
            // calculate the distance between the 2 objects
            float ydist = compute_distance(y_pos1, y_pos2);
        
            // obtain the masses of the 2 objects 
            float ym2 = sim_points_in[i].mass;
                        
            // calculate the force between the 2 objects
            float yforce_to_add = compute_force(ym1, ym2, ydist);
                        
            // add the force to the total force matrix
            total_force_y[k * CONST_NUM_POINTS + i] = yforce_to_add;
        }
    }
}

void update_sim_points(float * total_force_reduced_x, float * total_force_reduced_y, 
                                  Point * sim_points_in, Point * sim_points_out) {

    for (int k = 0; k < CONST_NUM_POINTS; k++) {
        // x-component logic
        // get initial position, velocity, and mass
        float x_pos1 = sim_points_in[k].x_pos;
        float x_vel1 = sim_points_in[k].x_vel;
        float xm1 = sim_points_in[k].mass;
        
        // update the acceleration
        float acceleration_x = compute_acceleration(xm1, total_force_reduced_x[k]);

        // update the velocity
        float updated_vel_x = compute_updated_velocity(x_vel1, acceleration_x);

        // update the position
        float updated_pos_x = compute_updated_pos(x_pos1, x_vel1, acceleration_x);

        // store updated position, velocity, and mass
        sim_points_out[k].mass = xm1;
        sim_points_out[k].x_vel = updated_vel_x;
        sim_points_out[k].x_pos = updated_pos_x;

	// y-component logic
        // get initial position, velocity, and mass
        float y_pos1 = sim_points_in[k].y_pos;
        float y_vel1 = sim_points_in[k].y_vel;
        float ym1 = sim_points_in[k].mass;
        
        // update the acceleration
        float acceleration_y = compute_acceleration(ym1, total_force_reduced_y[k]);

        // update the velocity
        float updated_vel_y = compute_updated_velocity(y_vel1, acceleration_y);

        // update the position
        float updated_pos_y = compute_updated_pos(y_pos1, y_vel1, acceleration_y);

        // store updated position and velocity
        sim_points_out[k].y_vel = updated_vel_y;
        sim_points_out[k].y_pos = updated_pos_y;
    }
}

void updatePointColor(int x, int y, unsigned char* bitmap, int col) {
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

void update_bitmap(Point* sim_points_in, Point* sim_points_out, unsigned char* bitmap)
{    
    for (int k = 0; k < CONST_NUM_POINTS; k++) {
        // get the initial and final positions of each object
        float x_pos1 = round(sim_points_in[k].x_pos);
        float y_pos1 = round(sim_points_in[k].y_pos);
        float updated_pos_x = round(sim_points_out[k].x_pos);
        float updated_pos_y = round(sim_points_out[k].y_pos);

        // Update the bitmap to our new body position
        updatePointColor(x_pos1, y_pos1, bitmap, 0);
        updatePointColor(updated_pos_x, updated_pos_y, bitmap, 255);

	// update the input for our next frame
	sim_points_in[k] = sim_points_out[k];
    }
}

// animation stuff
void generate_frame(DataBlock *d, int ticks) {

    GpuTimer timer;
    TimingData time_data;

    timer.Start();
    // run kernel - calculate all forces on every body in the simulation
    calculate_all_forces(d->sim_points_in, d->total_force_x, d->total_force_y);
    timer.Stop();

    time_data.calc_forces_ms = timer.Elapsed();

    // perform reductions
    for (int k = 0; k < CONST_NUM_POINTS; k++) {
        // reset the running sum to 0
        float running_sum_x = 0;
        for (int i = 0; i < CONST_NUM_POINTS; i++) {
            // add together all forces from every object
            running_sum_x += (d->total_force_x)[k * CONST_NUM_POINTS + i];
        } 
        // store the resulting total force in a new array
        (d->total_force_reduced_x)[k] = running_sum_x;

	// reset the running sum to 0
        float running_sum_y = 0;
        for (int i = 0; i < CONST_NUM_POINTS; i++) {
            // add together all forces from every object
            running_sum_y += (d->total_force_y)[k * CONST_NUM_POINTS + i];
        } 
        // store the resulting total force in a new array
        (d->total_force_reduced_y)[k] = running_sum_y;
    }

    // calculate updated position and velocity for the object
    timer.Start();

    update_sim_points(d->total_force_reduced_x, d->total_force_reduced_y, d->sim_points_in, d->sim_points_out);
    timer.Stop();

    time_data.update_points_ms = timer.Elapsed();

    timer.Start();

    unsigned char* ptr = (d->bitmap)->get_ptr();
    update_bitmap(d->sim_points_in, d->sim_points_out, ptr);

    timer.Stop();
    
    time_data.update_bitmap_ms = timer.Elapsed();

#if RECORD_TIMING_DATA
    file_timing_data << time_data.calc_forces_ms << "," << time_data.update_points_ms << "," << time_data.update_bitmap_ms << std::endl;
#endif
  
}

void cleanup(DataBlock *d) {
#if RECORD_TIMING_DATA
    // close the file
    file_timing_data.close();
#endif
}

// main function to perform physic operations 
int main() {

    // initialize datablock
    DataBlock data;

    // initialize bitmap
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    // load input data
    parse_input(FILENAME_INPUT_POINTS, data.sim_points_in);

#if RECORD_TIMING_DATA
    // create file to store timing data
    create_output_timing_file(FILENAME_OUTPUT_TIMING_DATA);
#endif

    // animate the bitmap
    bitmap.anim_and_exit( (void (*)(void*,int))generate_frame, (void (*)(void*))cleanup );

    return 0;
} 
