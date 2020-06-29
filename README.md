# AI-plays-Breakout-using-A3C
It is an Artificial Intelligence project where a computer learns to play Breakout game using the A3C model and using environment space from Open AI Gym.

The A3C model is made up of a combination of CNN, RNN, and ANN layers having 2 outputs. The model has 5 Convolutional Layers followed by LSTM (RNN) layer, followed by 2 hidden ANN layers and final two output layers: actor and critic.
The CNN layers are used by AI to visualize the environment/game; LSTM is used to keep memory and track of game movements; ANN layers to analyze the situation and predict the suitable outputs.


The project consists of following files:

    Model: Class file for creating functions of AI model used in training and testing
    myOptimizer: function file for creating an suitable customized optimizer for model learning
    eniviornment: to establish and create the enviornment of Breakout Game  from openAI gym. The file also gives list of all suitable outputs that the AI can perform.
    Testing & Train: Cretaing,training and testing AI for the job
    Main: Main file to execute and store results in videos. The videos consist of AI playing Breakout at different time of training and testing.

The test videos are stored in test folder.
