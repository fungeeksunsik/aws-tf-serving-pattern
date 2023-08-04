# demo-svhn-classifier

TL;DR: Tensorflow2 offers roughly 3 code styles of model implementation: Sequential, Functional and Subclassing. This project utilizes first style for implementing [SVHN dataset](http://ufldl.stanford.edu/housenumbers/) image classifier, whose level of abstraction is highest among the three. 

--- 

In any frameworks, high-level API sacrifices flexibility for implementation and makes overall code black-box if components are not well understood. However, purpose of it is to reduce amount of code implementation and therefore controlling overall system complexity. 

This can also be applied when implementing neural network program in Tensorflow, since this framework provides APIs with wide spectrum of abstraction. For example, when execution logic which contains fixed number of linear steps(i.e. not iterative or autoregressive), this can be simply implemented by Keras `Sequential` class. Also, APIs for monitoring and controlling training process are prepared within Keras `callbacks` API. 

Although there are APIs with lower level of abstraction to implement exactly same components, this project implements image classifiers using high-level Keras APIs to demonstrate the convenience and simplicity of them. Classifier with MLP, CNN architecture will be trained independently, and each of them will be saved with set of weights which provided best validation accuracy. After training classifiers, they can be examined with test data using Streamlit demo in local environment.

## Environment

This application is scripted and tested on M2 Apple Silicon machine.

```shell
conda env create
conda activate svhn
```

### Execute

To train and save models, execute `main.py` script. Series of logs provided by custom logger and Tensorflow will be displayed on console during the implemented process. Trained model will be saved in directory specified in `LOCAL_DIR` variable of the script.

```shell
python main.py
```

To try out trained models, execute Streamlit demo as following.

```shell
streamlit run demo.py
```