## Variational Autoencoder
This repository contains code and ideas about Variational Autoencoders. 
The current code contains an implementation of Variational Bayesian Autoencoder, and all ideas and mathematical foundations will be published in slides and in a report.



### How to use this repository:
1. Create venv

First of all we create a virtual Environment, activate it and install the requirements. If you use Pycharm you cann also add an interpreter, select "virtual environment" and create it. Afterwards you just need to execute the last two commands.
```
cd variational-autoencoder
python -m venv venv/ 
venv/Scripts/activate.bat # For windows users: venv\Scripts\activate.bat (in anaconda prompt)
python -m pip install -r requirements.txt
```
Depending on your IDE you can select the just created venv as Python Interpreter.

2. Tensorboard
```
cd variational-autoencoder
tensorboard --logdir=logs_tensorboard
#### or if we use Ligtning Module
tensorboard --logdir=lightning_logs/  
```

https://pytorch.org/get-started/locally/