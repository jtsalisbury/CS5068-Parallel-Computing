# Python script to perform physics calculations to check the CUDA code
# Parallel Computing - CS 5168
# Final project

CONST_GRAVITY = 6.673 * (10 ** -11)
CONST_TIME = 1
CONST_ITERATIONS = 50
CONST_SOFTENING_FACTOR = 1

# helper functions
def compute_force(mass_1, mass_2, dist):
    return CONST_GRAVITY * mass_1 * mass_2/(dist ** 2 + CONST_SOFTENING_FACTOR)

def compute_accel(mass, force):
    return force/mass

def compute_distance(pos_1, pos_2):
    return abs(pos_1 - pos_2)

def compute_pos(pos, vel, accel):
    return pos + vel * CONST_TIME + 0.5 * accel * (CONST_TIME ** 2)

def compute_vel(vel, accel):
    return vel + accel * CONST_TIME

# parse the csv file
with open("particles_simple.csv", 'r') as in_file: 
    # read in rows one at a time
    all_lines = in_file.readlines()
    first_line = True
    fields = []
    parsed_data = []

    for line in all_lines:
        if first_line:
            # split into the fields
            line_stripped = line.rstrip()
            fields = line_stripped.split(',')
            first_line = False
        else:
            # populated parsed data
            new_dict = {}
            line_stripped = line.rstrip()
            data = line_stripped.split(',')
            for k in range(len(fields)):
                if (fields[k] == "id"):
                    new_dict[fields[k]] = int(data[k])
                else:
                    new_dict[fields[k]] = float(data[k])
            parsed_data.append(new_dict)

    # print out initial point locations
    # print out updated point locations
    print("Initial positions")
    for i in range(len(parsed_data)):
        print("Point #{} - X: {}, Y: {}".format(parsed_data[i]["id"], parsed_data[i]["x_pos"], parsed_data[i]["y_pos"]))

    for iteration in range(CONST_ITERATIONS):
        print_iteration_num = False
        parsed_data_next = []

        for i in range(len(parsed_data)):
            total_force_x = 0
            total_force_y = 0

            # read the attributes of the 1st object
            x_pos_i = parsed_data[i]["x_pos"]
            x_vel_i = parsed_data[i]["x_vel"]
            y_pos_i = parsed_data[i]["y_pos"]
            y_vel_i = parsed_data[i]["y_vel"]
            mass_i = parsed_data[i]["mass"]

            for j in range(len(parsed_data)):
                # read the attributes of the 2nd object (minus velocity)
                x_pos_j = parsed_data[j]["x_pos"]
                y_pos_j = parsed_data[j]["y_pos"]
                mass_j = parsed_data[j]["mass"]

                # perform physics calculations
                dist_x = compute_distance(x_pos_i, x_pos_j)
                force_to_add_x = compute_force(mass_i, mass_j, dist_x)

                dist_y = compute_distance(y_pos_i, y_pos_j)
                force_to_add_y = compute_force(mass_i, mass_j, dist_y)

                if x_pos_i > x_pos_j:
                    # pull is to the left
                    total_force_x -= force_to_add_x
                elif x_pos_i < x_pos_j:
                    # pull is to the right
                    total_force_x += force_to_add_x
                # if it's the same point, don't add any force
                
                if y_pos_i > y_pos_j:
                    # pull is down
                    total_force_y -= force_to_add_y
                elif y_pos_i < y_pos_j:
                    # pull is up
                    total_force_y += force_to_add_y
                # if it's the same point, don't add any force

            # update acceleration
            accel_x = compute_accel(mass_i, total_force_x)
            accel_y = compute_accel(mass_i, total_force_y)

            # update velocity
            vel_x = compute_vel(x_vel_i, accel_x)
            vel_y = compute_vel(y_vel_i, accel_y)

            # update position
            pos_x = compute_pos(x_pos_i, x_vel_i, accel_x)
            pos_y = compute_pos(y_pos_i, y_vel_i, accel_y)

            # update positions and velocities in parsed_data
            new_dict = {}
            new_dict["id"] = parsed_data[i]["id"]
            new_dict["x_pos"] = pos_x
            new_dict["x_vel"] = vel_x
            new_dict["y_pos"] = pos_y
            new_dict["y_vel"] = vel_y
            new_dict["mass"] = parsed_data[i]["mass"]
            parsed_data_next.append(new_dict)

            # print out updated point locations
            if not print_iteration_num:
                print("\nIteration: {}".format(iteration + 1))
                print_iteration_num = True
            
            print("Point #{} - X: {}, Y: {}".format(parsed_data[i]["id"], pos_x, pos_y))

        parsed_data = parsed_data_next


        