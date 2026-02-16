The dataset used for this task was the BCCD Dataset (link: https://github.com/Shenggan/BCCD_Dataset); a small-scale dataset for blood cells detection. I used this dataset because it involves detecting three distinct classes—Red Blood Cells (RBC), White Blood Cells (WBC), 
and Platelets—within complex biological images.

Here are short descriptions about most of the files used in this assessment:
* The model_factory.py defines both the baseline and stronger models used for this assessment. The names of the models are:
1. baseline model: R-CNN ResNet50 FPN
2. Stronger model: Faster R-CNN model with a MobileNetV3-Large FPN backbone

* The config.py file has all the hyperparameters used for the training phase of this assessment.

* The evaluate.py file was used for creating the predicted bounding boxes for an input image and comparing it with the ground truth bounding boxes, showing the labels alongside each bounding box.


* The data_loader.py file was responsible for the parsing of the xml annotations from the BCCD dataset, preprocessing the images and then
mapping classes from the XML to numerical IDs that the model can understand.


Run the app.py by using this command in the windows powershell terminal: 
"start http://127.0.0.1:8000/docs; uvicorn app:app --reload"

For the training side of the code, use this command: "python3 train.py"

The model weights for both the baseline and stronger models can be downloaded from this link: https://drive.google.com/drive/folders/1MtRWg51N63TUpoikbGJz6WUtghTNbcuU?usp=sharing
Make sure to place the .pth files in the same directory as the rest of the python files uploaded in this repository.

Note: With the code files there is a file called "ML_AI_Engineer_Assessment.ipynb". This file was used for the sake of training the baseline and stronger models with a GPU for faster training time. This is a supplementary file and has the same code as the rest of the python files.

Note: steps for running the model and testing images on the web app frontend are given in the video link attached here: 
https://drive.google.com/file/d/1FJgJVQpYUvYuzbKrIAGVr3WzwGnxU9WT/view?usp=sharing

Happy testing!
