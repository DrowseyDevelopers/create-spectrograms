# create-spectrograms
##### Team: DrowseyDevelopers
##### Members: Michael Covarrubias, Juan Lozano, Arik Yueh

We call our team the 'Drowsey Developers' as we are a team on a quest
to see if we can create machine learning models that can classify
Electroencephalography (EEG) data, to either represent the
'focused', 'unfocused' or 'drowsey' attention state.


create-spectrograms repo used to generate spectrograms from EEG data.
```
    Module to take in .mat MatLab files and generate spectrogram images via Short Time Fourier Transform
         ----------          ------------------------------          --------------------
        | Data.mat |    ->  | Short-Time Fourier Transform |    ->  | Spectrogram Images |
         ----------          ------------------------------          --------------------
```

**Input:** Matlab Files
```
create-spectrogram/data/*.mat
```

**Output:** Spectrogram files generated in directory **output**
```
create-spectrogram/output/
```

## Clone Repository
We have another repository as a submodule, therefore we need to clone
recursively. Thus:
```
git clone https://github.com/DrowseyDevelopers/create-spectrograms.git --recursive
```

## Software Requirements
There are a few python packages that should be installed in order to get
the software running. Packages can be installed via **pip3**
```
scipy==1.4.1
matplotlib==3.1.1
numpy==1.17.3
pandas==0.25.2
```

## Generate Spectrogram Images
You need to make sure to have the **data** repository at **create-spectrogram/data**.
If so to get spectrogram image files run command:
```
python3 __main__.py
```

## Data Columns -> Channel Mapping
MatLab Columns
```
5 -> F7
6 -> F3
9 -> P7
10 -> O1
11 -> O2
12 -> P8
17 -> AF4
```

Python
```
4 -> F7
5 -> F3
8 -> P7
9 -> O1
10 -> O2
11 -> P8
16 -> AF4
```

columns
5, 6, 9, 10, 11, 12, 17


