## A small exercise using Named Entity Recognition and Classification (NERC)

This is a small repository that contains code to run Named Entity Recognition 
and Classification using the [spaCy](https://spacy.io/) library.

spaCy is a very useful and complete Natural Language Processing (NLP) library 
written in Python. It is easy to use and does a good job hiding the complexities of 
a full NLP processing pipeline.

The purpose of this code is **purely educational** and has no other practical use.


### Content of this README:

This readme file describes this educational activity and the content of the repository,
and it is arranged in the following way:

  - Description of the practice
  - The code and the data
  - How to install the requirements
  - How to launch the code using the console
  
  
### Description of the practice

What we are going to do is very simple.
No programming is involved, but feel free to have a look at the code.
If you are not familiar with Python or with programming in general, you may feel a bit 
confused at the beginning. But don't worry. The code is simple, and to some extent, 
well documented. And you don't need to understand it to launch the examples.

The practice is divided in two parts.

The first part is about using a NERC tool (the one that spaCy provides us) to analyze 
some text documents. We will make use of the default NERC models that spaCy has, already
trained for several languages, and ready to use.
The code is prepared to use these default spaCy models in three languages: English (en),
French (fr) and Spanish (es).

The second part is about training our own NERC model for a custom task. The default models
detect things like PERSON (PER), LOCATION (LOC), etc. But a named entity can be anything
of interest, depending on our objectives.
Using some train/test data provided in this repository (in a suitable format), we will
launch a training process, again using spaCy.
The trained model will address a different type of entities.

The training process is usually a long process (hours, days...). Fortunately spaCy is
fast enough and our dataset is small, so it should take only a few minutes.
During the training the [F-score](https://en.wikipedia.org/wiki/F1_score)
 metric will be monitored, and incremental versions
of the model will be saved the this metric improves.

Once the train ends (you can stop it at any time pressing Ctrl+C), we will use this 
custom model on some input texts to see what happens.


### The code

The code contained in this repository is basically a wrapping around spaCy functionality
and some helper functions (to convert data formats, launch console scripts, evaluate...).

There are two packages, one for running the NERC, which simply contains some logic to
parse the console arguments and use it to run the NERC with spaCy.
This part also prints some information to console (and optionally to a file) about the
number/types of entities detected. It also generates an HTML file with the detected 
entities highlighted. This HTML can be opened with a regular web browser like Firefox.

There is another package for the NERC training part. It contains more files because the
training involves more processes.

There is a data conversion file, that contains some logic
to convert the input data to a format suitable for spaCy. The logic is specifically created
to work for the input data in this practice, and for spaCy. Different data formats or 
a library other than spaCy, would require another specific data conversion.

There is also code for evaluation during training. This is important, and that is the
reason why we need evaluation data (called development set) during training. Otherwise
we wouldn't know if our training is actually working as expected and teaching something
to our new model. It is perfectly possible to have a model training for hours and achieve
nothing. We need some way to assess how good the model is doing, and if it is improving
from time to time during the training process.

And of course there is a file that contains the logic related to the training process 
itself. However, the core of the training is hidden by spaCy.
This way we do not need to know all the detail, only provide it with the data and some
extra parameters, and spaCy does the hard work.

### The data

For the first part of this practice there is no data.
You can open a web browser, a copy some text from a newspaper, or even write something
of your own into a file.

For the second part, in order to train a new model, we need some specific data.
In particular we need a train set and a development set. They are stored in this 
repository [here](data).
Note that there is a file called test_PLAIN.txt, generated for just this practice, that
contains a plain text version of the test.txt file, so it can be directly analyzed once 
we have trained a model.

This dataset is entitled:

**Materials Science Named Entity Recognition: train/development/test sets**

More information and references 
[HERE](https://figshare.com/articles/Materials_Science_Named_Entity_Recognition_train_development_test_sets/8184428/1)

It is described by their authors as:

*Training, development and test sets for supervised named entity recognition for materials science.
The data is labelled using the IOB annotation scheme. 
There exist 7 entity tags: material (MAT), sample descriptor (DSC), symmetry/phase label (SPL), property (PRO), application (APL), synthesis method (SMT), and characterization method (CMT), along with the outside tag (O).
The data consists of 800 hand-labelled materials science abstracts. 
The data has an 80-10-10 split, giving 640 abstracts in the training set, 80 in the development set, and 80 in the test set.*

Labelling a training set is a process that is usually done by hand, so it is a difficult and
time-consuming task. We can use this dataset here because its license is 
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).


### How to install the requirements

Get the code of this repository downloading if or cloning it with git.

Before you start, you need to have Python 3.7 (or newer) installed in your system.

To check if you have Python installed, and which version, open a console and run:

```bash
python --version
```

You should see something like:

```
Python 3.7.2
```

Or a similar version
(if you have recently installed Python, your version will probably be 3.8.x)

Using the console go to the root of this repository (nerc_practice).

It is advisable (though not mandatory) to create a Python virtual environment to avoid
conflicts with other libraries.


Run:
```
python -m venv .env
```

This will create a virtual environment inside a folder named *.env*

You need to activate the virtual environment before start working with it.

In Windows:
```
.env\Scripts\activate
```

In Linux:

```
source .env/bin/activate
```

You will see that your prompt (the left part of the console) now indicates you the name
of the virtual environment.

Now few more steps to install the dependencies. Ensure that you continue in the root folder
of the repository (the one that contains the requirements.txt file):

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download es_core_news_sm
```

The first step install all the dependencies (including spaCy).
The other three steps install the spaCy models for English, French and Spanish.

If you see no errors in the console, you are good to go.


### How to launch the code using the console

In order to launch the scripts, do the following (assuming you are still 
in the root folder):

```
python -m part1_use_a_nerc.run_nerc
```

You should see the following in the console

```
usage: run_nerc.py [-h] --file FILE [--lang {en,fr,es}] [--output OUTPUT]
                   [--custom_model CUSTOM_MODEL] [--top_n TOP_N]
run_nerc.py: error: the following arguments are required: --file

```

This is ok, only that you need to provide some extra parameters, like the file you 
want to analyze. In order to see a more exhaustive information about the parameters
for this command, run:

```
python -m part1_use_a_nerc.run_nerc -h
```

It will show you the help:

```
usage: run_nerc.py [-h] --file FILE [--lang {en,fr,es}] [--output OUTPUT]
                   [--custom_model CUSTOM_MODEL] [--top_n TOP_N]

Perform NERC over a file content.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Path to the file to be processed
  --lang {en,fr,es}     Language of the input
  --output OUTPUT       Optional path to file to write the results
  --custom_model CUSTOM_MODEL
                        Path to a custom model you have trained or downloaded
                        from elsewhere
  --top_n TOP_N         Top N entities to print in the output report

```

Example of usage:

```
python -m part1_use_a_nerc.run_nerc --file /path/to/file --lang fr
```

The next (and last!) command is for training:

```
python -m part2_train_custom_nerc.run_nerc_train -h

usage: run_nerc_train.py [-h] --train_data TRAIN_DATA --dev_data DEV_DATA
                         [--lang {en,fr,es}] [--num_epochs NUM_EPOCHS]
                         --output_dir OUTPUT_DIR [--model_name MODEL_NAME]

Perform NERC over a file content.

optional arguments:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA
                        Path to the training data
  --dev_data DEV_DATA   Path to the development data for evaluation
  --lang {en,fr,es}     Base language to train the model from (to create an
                        spaCy blank model)
  --num_epochs NUM_EPOCHS
                        Number of epochs (full training set loops)
  --output_dir OUTPUT_DIR
                        Path to the folder to store the trained models
  --model_name MODEL_NAME
                        Name for the model to be trained (will be used as part
                        of the name of the stored model

```

Example of usage:

```
python -m part2_train_custom_nerc.run_nerc_train --train_data data/train.txt --dev_data data/dev.txt --lang en
```

