num_of_files = 4
num_steps = 9 
num_seeds_to_use = 25

for i in range(0, num_of_files):
    file = open('tasks_dual_' + str(i), "w")
    for j in range(num_steps):
        for z in range(i * 25, (i + 1) * 25):
            file.write("/global/scratch/alobos/DSPOnline2020/runAsSecPrice.py  " + str(j) + ' ' + str(z) + "\n")
    file.close()