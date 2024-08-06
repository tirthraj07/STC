## Setup Lesson 2

Step 1: Install python virtual environment
```bash
pip install virtualenv
```
Step 2: Create a virtual environment in the Lesson 1 folder and activate
```bash
cd '.\Lesson 2 - Global Model with Federated Learning\'
virtualenv env
cd .\env\
.\Scripts\activate
cd ..
```

Step 3: Install the requirements from requirements.txt file
```bash
pip install -r requirements.txt
```

Step 4: Create necessary directories if not exists
```bash
mkdir plots/distribution,plots/confusion_matrix,models/global,training,testing
```

Step 5: Run `setup.py`
```bash
python setup.py
```
This will create training and testing datasets for our simulation

Step 6: Start 4 separate terminals terminals and run the following

`terminal 1`
```powershell
cd src/
python server.py
```
`terminal 2`
```powershell
cd src/
python client0.py
```
`terminal 3`
```powershell
cd src/
python client1.py
```
`terminal 4`
```powershell
cd src/
python client2.py
```

## Expected Output of server.py
```bash
INFO :      Starting Flower server, config: num_rounds=3, no round_timeout
INFO :      Flower ECE: gRPC server running (3 rounds), SSL is disabled
INFO :      [INIT]
INFO :      Using initial global parameters provided by strategy
INFO :      Evaluating initial global parameters
INFO :      test accuracy on all digits: 0.1323
INFO :      test accuracy on [1,3,7]: 0.3290
INFO :      test accuracy on [2,5,8]: 0.0794
INFO :      test accuracy on [4,6,9]: 0.0166
INFO :
INFO :      [ROUND 1]
INFO :      configure_fit: strategy sampled 3 clients (out of 3)
INFO :      aggregate_fit: received 3 results and 0 failures
INFO :      test accuracy on all digits: 0.8639
INFO :      test accuracy on [1,3,7]: 0.8862
INFO :      test accuracy on [2,5,8]: 0.7923
INFO :      test accuracy on [4,6,9]: 0.8650
INFO :      configure_evaluate: no clients selected, skipping evaluation
INFO :
INFO :      [ROUND 2]
INFO :      configure_fit: strategy sampled 3 clients (out of 3)
INFO :      aggregate_fit: received 3 results and 0 failures
INFO :      test accuracy on all digits: 0.9522
INFO :      test accuracy on [1,3,7]: 0.9672
INFO :      test accuracy on [2,5,8]: 0.9362
INFO :      test accuracy on [4,6,9]: 0.9376
INFO :      configure_evaluate: no clients selected, skipping evaluation
INFO :
INFO :      [ROUND 3]
INFO :      configure_fit: strategy sampled 3 clients (out of 3)
INFO :      aggregate_fit: received 3 results and 0 failures
INFO :      test accuracy on all digits: 0.9594
INFO :      test accuracy on [1,3,7]: 0.9660
INFO :      test accuracy on [2,5,8]: 0.9472
INFO :      test accuracy on [4,6,9]: 0.9525
INFO :      configure_evaluate: no clients selected, skipping evaluation
INFO :
INFO :      [SUMMARY]
INFO :      Run finished 3 round(s) in 110.99s
```
