import algorithms as algorithms

speed_input = 20 

pred = algorithms.receive_speed_from_webpage(speed_input)

print("The linear prediction is: ",pred)


npred = algorithms.receive_speed_from_webpage_Neural(speed_input)

print("The Neural prediction is =",npred)


npred_c = algorithms.receive_speed_from_webpage_Neural_Clean(speed_input)

print("The Neural prediction on the Cleaned Dataset is =",npred_c)
