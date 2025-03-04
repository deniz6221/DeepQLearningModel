## Deep Q Learning Model Implementation

This project implements Deep Q Learning to train a robot. The task of the robot here is to push an object towards its goal.\
\
I initially created a network that contains convolutional layers and used snapshots of the enviroment as states. 
However this approach resulted in a very complex network, training took ages and the results were awful. The model can be found in file DQN.py \
Then I created a simple 6x64x8 multilayer perceptron as the DQN, the training was faster but it still took 10 hours to complete 5000 episodes. The model can be found in file DQN_Higher.ipynb \
The cumulative reward-episode plot can be found on cumulative_reward.png, rps-episode plot can be found on rps.png \
The robot didn't quite learn the actions its supposed to take with only 5000 episodes of learning but with my CPU this was the best I could do. The model can be tested with the test() function.
