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
mkdir plots models
```

Step 5: Run `training.py`
```bash
python training.py
```

Step 6: Run `test.py`
```bash
python test.py
```