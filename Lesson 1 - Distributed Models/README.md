## How To Start

Step 1: Install python virtual environment
```bash
pip install virtualenv
```
Step 2: Create a virtual environment in the Lesson 1 folder and activate
```bash
cd '.\Lesson 1 - Distributed Models\'
virtualenv env
cd .\env\
.\Scripts\activate
cd ..
```

Step 3: Install the requirements from requirements.txt file
```bash
pip install -r requirements.txt
```

Step 4: Create `models` and `plots` directory if not exists
```bash
mkdir plots,models
```

Step 5: Run `training.py`
```bash
python training.py
```

Step 6: Run `test.py`
```bash
python test.py
```

### Ideal Output
```
Testing Accuracy of Models
Loading Model 1..
Model 1 Loaded.
Loading Model 2..
Model 1 Loaded.
Loading Model 3..
Model 1 Loaded.
Creating a Test set for checking accuracy..
Test set created of length : 10000
Checking accuracy of model 1..
Model 1-> Test Accuracy on all digits: 0.6570, 
Checking accuracy of model 2..
Model 2-> Test Accuracy on all digits: 0.6872, 
Checking accuracy of model 3..
Model 3-> Test Accuracy on all digits: 0.6848, 
Creating specialized dataset for all models which contain the respective missing values from their training...
Specialized datasets created
Checking accuracy of model 1..
Test Accuracy on [1,3,7]: 0.0000
Checking accuracy of model 2..
Test Accuracy on [2,5,8]: 0.0000
Checking accuracy of model 3..
Test Accuracy on [4,6,9]: 0.0000
Simulation Complete
```