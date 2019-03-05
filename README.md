# Deep Learning Practical
This repository contains exercieses that accompany the lecture Deep Learning held by Prof. Hammer at Bielefeld University in February 2019.

## Setup
Our exercises come in the form of Jupyter Notebooks. The packages required to run them are listed in `environment.yml`. We recommended using Anaconda to create a virtual environment in which to run the notebooks. Here is how we suggest you set up Anaconda:

1. Install Anaconda if you do not have it installed already:
    - Download the installer from https://www.anaconda.com/distribution/
    - Execute the installer in a terminal, e.g. `bash Anaconda3-2018.12-Linux-x86_64.sh`.
    - Follow the instructions. Installing to your home is recommended and the default. You should *not* initialize Anaconda in your `.bashrc`, which is also the default. You should *not* install Microsoft VSCode.
2. Run conda with the `activate` argument, e.g. `~/anaconda3/bin/conda activate`.
3. This will ask you to enable conda as such: `echo ". /home/username/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc`. Do that.
4. Open a new terminal.
5. Navigate to this project's directory.
6. Create the environment: `conda env create`. This automatically finds and uses `environment.yml`, which creates an environment called `deep` with everything you need for our exercieses.

## Running
In this project's directory, activate the environment and launch jupyter:
```
conda activate deep
jupyter-notebook
```

## Report
Please use the `exercise_sheet_template.tex` to generate your report. Your report is due on *Friday, March 15th, 10am* as single-page PDF to [aschulz@techfak.uni-bielefeld.de](mailto:aschulz@techfak.uni-bielefeld.de). Please start your e-mail subject with `[Deep Learning]`.
