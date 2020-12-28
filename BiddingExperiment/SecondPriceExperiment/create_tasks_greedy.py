num_of_files = 10
num_steps = 26
num_seeds_to_use = 10

for i in range(0, num_of_files):
    file = open('tasks_greedy_' + str(i), "w")
    for j in range(num_steps):
        for z in range(i * 10, (i + 1) * 10):
            file.write("/global/scratch/alobos/DSPOnline2020/runAsGreedy.py  " + str(j) + ' ' + str(z) + "\n")
    file.close()