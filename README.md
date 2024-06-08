## Fed2PKD: Bridging Model Diversity in Federated Learning via Two-Pronged Knowledge Distillation
## Requirments
This code requires the following:
* Python 3.6 or greater
* PyTorch 1.6 or greater
* Torchvision
* Numpy 1.18.5

## Data Preparation
* Download train and test datasets manually from the given links, or they will use the defalt links in torchvision.
* Experiments are run on MNIST, FEMNIST and CIFAR10.
http://yann.lecun.com/exdb/mnist/
https://s3.amazonaws.com/nist-srd/SD19/by_class.zip
http://www.cs.toronto.edu/∼kriz/cifar.html

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To train the Fed2PKD on MNIST/CIFAR10/CIFAR100 with n=7, k=500 std=2 tau=0.07 under statistical heterogeneous setting:
```
python federated_edit_main.py --mode model_heter --dataset mnist --num_classes 10 --num_users 20 --ways 7 --shots 500 --optimizer sgd --local_bs 32 --stdev 2 --rounds 300 --train_shots_max 510 --ldc 0.9 --ld 0.9 --tau 0.07 --k 20
```



You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'cifar10', 'cifar100'
* ```--num_classes:```  Default: 10. Options: 10, 10, 100
* ```--seed:```     Random Seed. Default set to 1234.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--momentum:```       Learning rate set to 0.5 by default.
* ```--local_bs:```  Local batch size set to 32 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.


#### Federated Parameters
* ```--num_users:```Number of users. Default is 20.
* ```--ways:```      Average number of local classes. Default is 5.
* ```--shots:```      Average number of samples for each local class. Default is 500.
* ```--test_shots:```      Average number of test samples for each local class. Default is 15.
* ```--ldc:```      Weight of L_ckd. Default is 0.9.
* ```--ld:```      Weight of L_gkd. Default is 0.9.
* ```--stdev:```     Standard deviation. Default is 2.
* ```--train_ep:``` Number of local training epochs in each user. Default is 1.
* ```--tau:```      Value of tau. Default is 0.07.
* ```--k:```      Number of S_t. Default is 50.
* ```--rounds:```   Value of total training epochs. Default is 300.
