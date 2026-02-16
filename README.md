The dataset used for this task was the BCCD Dataset (link: https://github.com/Shenggan/BCCD_Dataset); a small-scale dataset for blood cells detection. I used this dataset because it involves detecting three distinct classes—Red Blood Cells (RBC), White Blood Cells (WBC), 
and Platelets—within complex biological images.

Run the app.py by using this command in the windows powershell terminal: 
"start http://127.0.0.1:8000/docs; uvicorn app:app --reload"

For the training side of the code, use this command: "python3 train.py"

The model weights for both the baseline and stronger models can be downloaded from this link: https://drive.google.com/drive/folders/1MtRWg51N63TUpoikbGJz6WUtghTNbcuU?usp=sharing
Make sure to place the .pth files in the same directory as the rest of the python files uploaded in this repository.

Note: With the code files there is a file called "ML_AI_Engineer_Assessment.ipynb". This file was used for the sake of training the baseline and stronger models with a GPU for faster training time. This is a supplementary file and has the same code as the rest of the python files.

Note: steps for running the model and testing images on the web app frontend are given in the video link attached here: https://drive.google.com/file/d/1c48--SXOJbHidtUcPux17lKJP72Bx6Pr/view?usp=sharing


Happy testing!
