# CS6068-Parallel-Computing

This program is intended to demonstrate a 2D N-body simulation of particles in a galaxy. 

## Usage 

- It is necessary to have the cuda_by_example folder downloaded to your local system, as the program has dependencies from files in the "common" folder
- It is necessary to have an input CSV file including the following information: id, x_pos, x_vel, y_pos, y_vel, mass
- The program uses"bodies.csv" file by default 
 
To run the parallel implementation of our program,
 
bash$ module load cuda <br />
bash$ module load virtualgl <br />
bash$ nvcc galaxy.cu -lm -lGL -lGLU -lglut <br />
bash$ ./a.out <br />

To run the sequential implementation of our program, 

bash$ module load cuda <br />
bash$ module load virtualgl <br />
bash$ nvcc galaxy_seq.cu -lm -lGL -lGLU -lglut <br />
bash$ ./a.out <br />


 
