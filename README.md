# Shelter-Animals-Outcome-Kaggle
Every year, approximately 7.6 million companion animals end up in US shelters. Many animals are given up as unwanted by their owners, while others are picked up after getting lost or taken out of cruelty situations. Many of these animals find forever families to take them home, but just as many are not so lucky. 2.7 million dogs and cats are euthanized in the US every year.
Using a dataset of intake information including breed, color, sex, and age from the Austin Animal Center, we're asking Kagglers to predict the outcome for each animal.

We also believe this dataset can help us understand trends in animal outcomes. These insights could help shelters focus their energy on specific animals who need a little extra help finding a new home. We encourage you to publish your insights on Scripts so they are publicly accessible.


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
sample_submission.csv
test.csv
train.csv

In [2]:
import pandas as pd 
import numpy as np
import seaborn as snb
import matplotlib.pyplot as plt
%matplotlib inline

all_animals=pd.read_csv('../input/train.csv')
In [3]:
#our next important step is to identify the type of all_animals we have in our dataset
snb.countplot(all_animals.AnimalType, palette='Set1')

#So from the barchart distribution we can see that the number of dogs are 4500 more than the number of cats. 
Out[3]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f0744598f60>

In [4]:
# we can also see the sex of the all_animals in our dataset
#to have a better undestanding of our animals groups.
snb.countplot(all_animals.SexuponOutcome, palette='Set1')

#the output shows most of the animals are Neutered Male and very few are Unknown
Out[4]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f0745268ba8>

In [5]:
# Furthermore we can also see the distribution of the age outcome of the all_animals in our dataset 
snb.countplot(all_animals.AgeuponOutcome, palette='Set1')
Out[5]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f074521ff98>

In [6]:
# so in the above 2 bar charts we observe two important informations that we need to separate for clear visualization
# first we have the information whether the animal is Male or Female
#And second we have whether the animal is neureted, spayed or intact.
#So sepatating these information is helpful for visualization and manupulation
In [7]:
def new_sex(m):
    m = str(m)
    if m.find('Male') >= 0:
        return 'male'
    if m.find('Female') >= 0: 
        return 'female'
    return 'unknown'
def new_neutered(m):
    m = str(m)
    if m.find('Spayed') >= 0: 
        return 'neutered'
    if m.find('Neutered') >= 0: 
        return 'neutered'
    if m.find('Intact') >= 0: 
        return 'intact'
    return 'unknown'
In [8]:
all_animals['Sex'] = all_animals.SexuponOutcome.apply(new_sex)
all_animals['Neutered'] = all_animals.SexuponOutcome.apply(new_neutered)
f, (am1, am2) = plt.subplots(1, 2, figsize=(15, 4))
snb.countplot(all_animals.Sex, palette='Set1', ax=am1)
snb.countplot(all_animals.Neutered, palette='Set1', ax=am2)
Out[8]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f07450d1860>

In [9]:
#Now we have a good picture of the the male and female distribution as well as the neutered and intact. thus the number of Male and Female animals are almost equal
#However the nauereted animals are more than twice to intact animals. 

#But beside the above independent variables in our dataset, we have also another variable Breed. And we want to see if this vatiable have also significant influence on the outcome of the animals.
In [10]:
def Newcol_Breed(m):
    m=str(m)
    if m.find('Mix')>=0:
        return 'mixed_animal'
    else:
        return 'not_Mixed_animal'
all_animals['Mix']=all_animals.Breed.apply(Newcol_Breed)
snb.countplot(all_animals.Mix, palette='Set1')
Out[10]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f0745060710>

In [11]:
#So what we can observe is that most of the animals are mixed but only 5000 animals are not mixed

#lets see the influence of different independent variables on the final outcome
In [12]:
f, (ax1, ax2)=plt.subplots(1, 2, figsize=(15, 4))
snb.countplot(data=all_animals, x='OutcomeType', hue='Sex', palette='Set1', ax=ax1)
snb.countplot(data=all_animals, x='Sex', hue='OutcomeType', palette='Set1', ax=ax2)

#Here we can clearly see that most of the male and female animals have higher adoption rate.
Out[12]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f0744fc7a90>

In [13]:
f, (ax1, ax2)=plt.subplots(1, 2, figsize=(15, 4))
snb.countplot(data=all_animals, x='OutcomeType', hue='AnimalType', palette='Set1', ax=ax1)
snb.countplot(data=all_animals, x='AnimalType', hue='OutcomeType', palette='Set1', ax=ax2)

#But here interestingly enough most of the dogs in our dataset  have the highest probablity of return to their owners
#However the number of cats transfer is higher than the dogs, enevthough they have relatively lower adoption rate than their counterpart dogs. 
Out[13]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f0744ec6908>

In [14]:
f, (ax1, ax2)=plt.subplots(1, 2, figsize=(15,4))
snb.countplot(data=all_animals, x='OutcomeType', hue='Neutered', palette='Set1',ax=ax1)
snb.countplot(data=all_animals, x='Neutered', hue='OutcomeType', palette='Set1', ax=ax2)

#Another interesting outcome is almost all of the neutered animals have the highest probablity if adoption
#on the otherhand intact animals have higher transfer rate than the neutered animals. 
Out[14]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f0744daf668>

In [15]:
#And what about the Breed animals, lets see how does mixed animals affect the ourcome of the animals

f, (ax1, ax2)=plt.subplots(1, 2, figsize=(15,4))
snb.countplot(data=all_animals, x='OutcomeType', hue='Mix', palette='Set1', ax=ax1)
snb.countplot(data=all_animals, x='Mix', hue='OutcomeType', palette='Set1', ax=ax2)

#the graph below shows that Mixed animlas have the highest chance of adoption and transfer, 
#while non mixed animals have have the lowest chance of adoption and transfer
Out[15]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f0744cac5c0>

In [16]:
#Age is another independent variable that we have in our dataset. Age may also have some significance on the outcome and lets see it
In [17]:
# Here age is given in different in months and years, so our first step should be converting them all to the same measure
def years_of_age(y):
    y = str(y)
    if y == 'nan': 
        return 0
    AgeOfAnimals = int(y.split()[0])
    if y.find('year') > -1: 
        return AgeOfAnimals 
    if y.find('month')> -1: 
        return AgeOfAnimals / 12.
    if y.find('week')> -1: 
        return AgeOfAnimals / 52.
    if y.find('day')> -1: 
        return AgeOfAnimals / 365.
    else: 
        return 0
In [18]:
all_animals['AnimalAge'] = all_animals.AgeuponOutcome.apply(years_of_age)
snb.distplot(all_animals.AnimalAge, bins = 20, kde=False)
Out[18]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f0744c2d550>

In [19]:
#Hence the aobe graph shows that most of the animals in the shelter have 0 and 1.5 yrs

#But does this have an effect on the outcome? lets see it.
In [20]:
def age_division(y):
    if y <=2.5:
        return 'Young'
    elif y>2.5 and y<8:
        return 'Young Adult'
    else:
        return 'Old'
all_animals['CatagoryAge'] = all_animals.AnimalAge.apply(age_division)
In [21]:
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
snb.countplot(data=all_animals, x='OutcomeType', hue='CatagoryAge', ax=ax1)
snb.countplot(data=all_animals, x = 'CatagoryAge', hue='OutcomeType', ax=ax2)
Out[21]:
<matplotlib.axes._subplots.AxesSubplot at 0x7f0744b24080>

In [22]:
# Finally in the above two output chats indicate that young animals of dogs and cats have higher probablity of aadoption that the other types of animals
#on the other hand older dogs and cats have the lowest probablity of adoption or transfer
