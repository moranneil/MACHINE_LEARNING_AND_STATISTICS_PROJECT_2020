import algorithms as algorithms

speed_input = 25

pred = algorithms.receive_speed_from_webpage(speed_input)

print("The linear prediction is: ",pred)


npred = algorithms.receive_speed_from_webpage_Neural(speed_input)

print("The Neural prediction is =",npred)
