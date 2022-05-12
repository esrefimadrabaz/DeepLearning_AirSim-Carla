# DeepLearning_AirSim-Carla
*** needs refactoring and some commenting ***
A deep learning preparing and testing environment. The data is the Carla Simulator's public testing data. It is roughly about 10GB's so it is fairly easy to find it.
configure.py script has a few functions for visualizing, preparing the data and then cooking it. It is not automated so you need to change it to use.
Configure_data() function in configure.py edits the Carla dataset so it looks more like a dataset that AirSim generated. This could be skipped, you could modify the data in the testing environment according to the environment.
The script then saves the data as panda DataFrame and calls the function to cook it, cooking proccess saves the data as h5py. This cooking and training proccess is influenced a lot by AirSim's documentation and examples. Explanations can be found in the original repo.
Some modifications were done to cooking and training scripts. I have modified the model and made it closer to Nvidia's end to end deep learning model for Autonomous cars to see how it does and thought it could do better as AirSim's model was developed for more "off-road" conditions.
The training and testing environment I used was more urban and city-like.


