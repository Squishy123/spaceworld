# spaceworld
World-Based Model Learning using LunarLander-v2


# Installation
## On Linux: 
### Install Conda 
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html 

### Create Conda Environment and Install System Dependencies
```
sudo make install_sys
```

### Activate and Install Pip Dependencies
```
conda activate spaceworld
make install_dep
```

# Running the Project
## Run the Main Script to start training an agent
```
python src/main.py
```