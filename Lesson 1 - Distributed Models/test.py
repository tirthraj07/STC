from utils.utils1 import *

print("Testing Accuracy of Models")

print("Loading Model 1..")
model1 = SimpleModel()
model1.load_state_dict(torch.load('./models/model1.pth'))
print("Model 1 Loaded.")

print("Loading Model 2..")
model2 = SimpleModel()
model2.load_state_dict(torch.load('./models/model2.pth'))
print("Model 1 Loaded.")

print("Loading Model 3..")
model3 = SimpleModel()
model3.load_state_dict(torch.load('./models/model3.pth'))
print("Model 1 Loaded.")

print("Creating a Test set for checking accuracy..")
testset = datasets.MNIST(
    "./MNIST_data/", download=False, train=False, transform=transform
)
print("Test set created of length : " + str(len(testset)))

print("Checking accuracy of model 1..")
average_loss1, accuracy1 = evaluate_model(model1, testset)
print(
    f"Model 1-> Test Accuracy on all digits: {accuracy1:.4f}, "
)

print("Checking accuracy of model 2..")
average_loss2, accuracy2 = evaluate_model(model2, testset)
print(
    f"Model 2-> Test Accuracy on all digits: {accuracy2:.4f}, "
)

print("Checking accuracy of model 3..")
average_loss3, accuracy3 = evaluate_model(model3, testset)
print(
    f"Model 3-> Test Accuracy on all digits: {accuracy3:.4f}, "
)

print("Creating specialized dataset for all models which contain the respective missing values from their training...")
testset_for_model1 = include_digits(testset, included_digits=[1,3,7]);
testset_for_model2 = include_digits(testset, included_digits=[2,5,8]);
testset_for_model3 = include_digits(testset, included_digits=[4,6,9]);
print("Specialized datasets created")

print("Checking accuracy of model 1..")
average_loss1, accuracy1_on_137 = evaluate_model(model1, testset_for_model1)
print(
    f"Test Accuracy on [1,3,7]: {accuracy1_on_137:.4f}"
)

print("Checking accuracy of model 2..")
average_loss2, accuracy2_on_258 = evaluate_model(model2, testset_for_model2)
print(
    f"Test Accuracy on [2,5,8]: {accuracy2_on_258:.4f}"
)

print("Checking accuracy of model 3..")
average_loss3, accuracy3_on_469 = evaluate_model(model3, testset_for_model3)
print(
    f"Test Accuracy on [4,6,9]: {accuracy3_on_469:.4f}"
)

print("Simulation Complete")