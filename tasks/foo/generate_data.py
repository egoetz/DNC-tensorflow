import random


for i in range(0, 1000000):
    my_input = random.randint(0, 1000000000)
    my_input2 = my_input + 1
    my_output = my_input2 + 1
    file.write(f"{my_input} {my_input2} {my_output}\n")