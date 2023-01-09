# xAutoML-Project2
**Explainable automated framework for predicting the risk of major adverse cardiac event (MACE)**

Project 2 for the 'Explainable Automated Machine Learning' (LTAT.02.023) course

## Team
Dmitri Rozgonjuk <br>
Lisanna Lehes <br>
Marilin Moor <br>
Allan Mitt <br>
Jure Vito Srovin 

## Project Objective
This project aims to use AutoML to predict clinical outcomes using imaging and clinical variables. The imaging modality of interest is positron emission tomography (PET). The outcome of interest to predict is major adverse cardiac event (MACE) with heart failure. The more specific goals are (1) finding the best machine learning pipelines (using the [TPOT framework](https://github.com/EpistasisLab/tpot)) for models based on two datasets with best weighted F1-scores, and (2) applying interpretability techniques to provide insights into the black-box models in order to explain the major drivers of predictions on a global as well as local level.

## Project Workflow
TBW - will have a Canva flowchart here.


## Files and Directories
- `tpot_models/`: directory that includes the python files for TPOT models (results)
  - `tpot_X1.py`: results for Model 1 (less features)
  - `tpot_X2.py`: results for Model 2 (more features)
- `README.md`: the present file that includes the project meta-information
- `autosklearn_approach.ipynb`: a notebook where we initially tried to implement the `auto-sklearn` approach; not used in the final solution.
- `interpretability.ipynb`: a notebook that includes the interpreatibility part; pre-requisite is the existence of TPOT model files
- `requirements.txt`: python packages for installing
- `tpot_approach.ipynb`: a notebook with the TPOT implementation as the automated ML approach

## How to Run
TBW

## Project Solution
TBW

## Summary and Conclusions
TBW
