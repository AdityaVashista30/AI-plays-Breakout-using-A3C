#Here we are combining main three neural networks: ANN.CNN & RNN with A3C

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights,std=1): #standard deviation as a parameter, inorder to set different weights for actors and critic
    output=torch.randn(weights.size())
    output*=std/torch.sqrt(output.pow(2).sum(1).expand_as(output))
    return output
    
# Initializing the weights of the neural network in an optimal way for the learning
def weights_init(m): #m=neural network object
    #we will use different if conditions to initializ weigths of convulational layers and fully connected layers
    classname=m.__class__.__name__ #will look for the type of connection in the object "m"
    if classname.find('Conv')!=-1: # if the connection is a convolution
        weight_shape=list(m.weights.data.size()) # list containing the shape of the weights in the object "m"
        fan_in=np.prod(weight_shape[1:4])
        fan_out=np.prod(weight_shape[2:4])*weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound; represent size of tensor weight 
        m.weights.data.uniform_(-w_bound, w_bound)# generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros
    elif classname.find('Linear') != -1: # if the connection is a full connection
    #DONE SEPERATELY AS CONVULATIONAL LAYERS HAVE MORE DIMENSIONS THAN CONNECTED LAYERS
        weight_shape=list(m.weights.data.size()) 
        fan_in=np.prod(weight_shape[1])
        fan_out=np.prod(weight_shape[0])
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound; represent size of tensor weight 
        m.weights.data.uniform_(-w_bound, w_bound)# generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0)
    

# Making the A3C brain
class ActorCritic(torch.nn.Module):
    def __init__(self,num_inputs,action_space):
        super(ActorCritic,self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1) # first convolution
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # second convolution
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # third convolution
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # fourth convolution
        self.conv5 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # fifth convolution
        self.lstm=nn.LSTMCell(32*3*3, 256)
        num_outputs = action_space.n # getting the number of possible actions
        self.fc1=nn.Linear(256,128)#hidden layer 1
        self.fc1=nn.Linear(128,256)#hidden layer 2
        self.critic_linear = nn.Linear(256, 1) # full connection of the critic: output = V(S)
        self.actor_linear = nn.Linear(256, num_outputs) # full connection of the actor: output = Q(S,A)
        self.fc1.weight.data=normalized_columns_initializer(self.fc1.weight.data,1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data=normalized_columns_initializer(self.fc2.weight.data,1)
        self.fc2.bias.data.fill_(0)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01) # setting the standard deviation of the actor tensor of weights to 0.01
        self.actor_linear.bias.data.fill_(0) # initializing the actor bias with zeros
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0) # setting the standard deviation of the critic tensor of weights to 0.01
        self.critic_linear.bias.data.fill_(0) # initializing the critic bias with zeros
        self.lstm.bias_ih.data.fill_(0) # initializing the lstm bias with zeros
        self.lstm.bias_hh.data.fill_(0) # initializing the lstm bias with zeros
        self.train() # setting the module in "train" mode to activate the dropouts and batchnorms
    
    def forward(self,inputs):
        inputs, (hx, cx) = inputs # getting separately the input images to the tuple (hidden states, cell states)
        x = F.elu(self.conv1(inputs)) # forward propagating the signal from the input images to the 1st convolutional layer
        x = F.elu(self.conv2(x)) # forward propagating the signal from the 1st convolutional layer to the 2nd convolutional layer
        x = F.elu(self.conv3(x)) # forward propagating the signal from the 2nd convolutional layer to the 3rd convolutional layer
        x = F.elu(self.conv4(x)) # forward propagating the signal from the 3rd convolutional layer to the 4th convolutional layer
        x = F.elu(self.conv5(x)) # forward propagating the signal from the 4th convolutional layer to the 5th convolutional layer
        x = x.view(-1, 32 * 3 * 3) # flattening the last convolutional layer into this 1D vector x
        hx, cx = self.lstm(x, (hx, cx)) # the LSTM takes as input x and the old hidden & cell states and ouputs the new hidden & cell states
        x = hx # getting the useful output, which are the hidden states (principle of the LSTM)
        x=F.elu(self.fc1(x))
        x=F.elu(self.fc2(x))
        return self.critic_linear(x), self.actor_linear(x), (hx, cx) # returning the output of the critic (V(S)), the output of the actor (Q(S,A)), and the new hidden & cell states ((hx, cx))
