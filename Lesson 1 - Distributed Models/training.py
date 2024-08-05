"""
We are going to import the utils1.py which they have provided us for easily training the data
It includes 
- datasets module : which allows use to import datasets like MNIST dataset from MNIST_DATA folder
- SimpleModel : a Neural Network implemented in Pytorch with 2 fully connected layers
- include_digits() and exclude_digits : Helper functions to simulate the redundancy of the data in distributed environments
- evaluate() -> which helps to evaluate the model

I am going to modify the plot_distribution() function to allow us to save the plots in the plots folder
I am going to save the trained models in the models folder
"""

from utils.utils1 import *
import torch


# Create a training dataset 
print("Creating a Training dataset..")
trainset = datasets.MNIST(
    "./MNIST_data/", download=True, train=True, transform=transform
)
print("Creation Complete.")

# Lets simulate the different datasets that might be available in real world (datasets with missing data, extra data, etc).
print("Simulating behavior of distributed systems..")
total_length = len(trainset)
print("Total length of Training data : "+ str(total_length))
split_size = total_length // 3
torch.manual_seed(42)
part1, part2, part3 = random_split(trainset, [split_size] * 3)
print("Dividing the total training data into 3 parts randomly of size : " + str(split_size) + " to simulate distributed datasets")


print("Excluding some digits from each dataset to simulate the missing data in local training")
part1 = exclude_digits(part1, excluded_digits=[1, 3, 7])
part2 = exclude_digits(part2, excluded_digits=[2, 5, 8])
part3 = exclude_digits(part3, excluded_digits=[4, 6, 9])

print("Saving the distribution of the dataset in each part..")
save_plot_distribution(part1, "Part 1", "plots/model1_distribution.png")
save_plot_distribution(part2, "Part 2", "plots/model2_distribution.png")
save_plot_distribution(part3, "Part 3", "plots/model3_distribution.png")
print("Plots are saved in ./plots/ directory")

print("Creating simple NN Models and training them on their respective dataset")

print("Training Model 1...")
model1 = SimpleModel()
train_model(model1, part1)
torch.save(model1.state_dict(), 'models/model1.pth')
print("Model 1 saved in ./models/ directory")

print("Training Model 2...")
model2 = SimpleModel()
train_model(model2, part2)
torch.save(model2.state_dict(), 'models/model2.pth')
print("Model 2 saved in ./models/ directory")

print("Training Model 3...")
model3 = SimpleModel()
train_model(model3, part3)
torch.save(model3.state_dict(), 'models/model3.pth')
print("Model 3 saved in ./models/ directory")
