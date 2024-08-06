from src.util.utils2 import *
import torch

print("Running setup.py ..")

############## TRAINING SETUP ##############

# Create the training dataset and save them in the ./training/ folder for future use

print("Creating MNIST dataset for training..")
trainset = datasets.MNIST(
    "./MNIST_data/", download=True, train=True, transform=transform
)

total_length = len(trainset)
print("Creating Complete. Size of training dataset = " + str(total_length))


split_size = total_length // 3
print("Dividing the total training data into 3 parts randomly of size : " + str(split_size) + " to simulate distributed datasets")
torch.manual_seed(42)
part1, part2, part3 = random_split(trainset, [split_size] * 3)


print("Excluding some digits from each dataset to simulate the missing data in local training")
part1 = exclude_digits(part1, excluded_digits=[1, 3, 7])
part2 = exclude_digits(part2, excluded_digits=[2, 5, 8])
part3 = exclude_digits(part3, excluded_digits=[4, 6, 9])

train_sets = [part1, part2, part3]

print("Exporting datasets to ./training/ directory.. ")
torch.save(trainset, './training/trainset.pt')
torch.save(part1, './training/model1_trainset.pt')
torch.save(part2, './training/model2_trainset.pt')
torch.save(part3, './training/model3_trainset.pt')
print("Export complete")

print("Saving the distribution of the dataset in each part..")
save_plot_distribution(part1, "Part 1", "plots/distribution/model1_distribution.png")
save_plot_distribution(part2, "Part 2", "plots/distribution/model2_distribution.png")
save_plot_distribution(part3, "Part 3", "plots/distribution/model3_distribution.png")
print("Plots are saved in ./plots/distribution/ directory")

############## END TRAINING SETUP ##############

############## TESTING SETUP ##############

# Create the testing dataset and save them in the ./testing/ folder for future use

print("Creating MNIST dataset for testing..")
testset = datasets.MNIST(
    "./MNIST_data/", download=True, train=False, transform=transform
)
print("Test set created of length : " + str(len(testset)))

print("Creating specialized dataset for all models which contain the respective missing values from their training...")
testset_for_model1 = include_digits(testset, included_digits=[1,3,7]);
testset_for_model2 = include_digits(testset, included_digits=[2,5,8]);
testset_for_model3 = include_digits(testset, included_digits=[4,6,9]);
print("Specialized datasets created")

print("Exporting datasets to ./testing/ directory.. ")
torch.save(testset, './testing/testset.pt')
torch.save(testset_for_model1, './testing/model1_testset.pt')
torch.save(testset_for_model2, './testing/model2_testset.pt')
torch.save(testset_for_model3, './testing/model3_testset.pt')
print("Export complete")


############## END TESTING SETUP ##############


print("Setup Complete")