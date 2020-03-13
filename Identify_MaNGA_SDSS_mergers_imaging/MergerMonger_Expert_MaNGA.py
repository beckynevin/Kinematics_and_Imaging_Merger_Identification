'''
~~~
MergerMonger_Expert classifies a provided .txt file of MaNGA galaxies with predictors
using the posterior probability distributions from another .txt file that has
the imaging predictors from the input simulated galaxies (these are used to construct the LDA).
It does a very similar thing to MergerMonger_Novice but produces prior probabilities instead
of just a binary classification.
My goal is to make the classification threshold adjustable to have cleaner samples of mergers.
(Currently, it is just set at p_merg = 0.5)
~~~
'''

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import itertools
from sklearn import preprocessing
import os
import sklearn.metrics as metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.signal import argrelextrema
from astropy.io import fits
import matplotlib.colors as colors



def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a)
                  if smallest == element]

os.chdir(os.path.expanduser('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/Identify_MaNGA_SDSS_mergers_imaging'))

# A lot of the ML parts of this code are from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, target_names, title, cmap=plt.cm.Blues):
    sns.set_style("dark")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    target_names=['Nonmerger','Merger']
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True label', size=20)
    plt.xlabel('Predicted label', size=20)
    
    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

# The first step is to load up the prepared file that has the predictors of simulated galaxies
feature_dict = {i:label for i,label in zip(
                range(14),
                  ('Counter',
                  'Image',
                  'class label',
                  'Myr',
                  'Viewpoint',
                '# Bulges',
                   'Sep',
                   'Flux Ratio',
                  'Gini',
                  'M20',
                  'Concentration (C)',
                  'Asymmetry (A)',
                  'Clumpiness (S)',
                  'Sersic N',
                  'Shape Asymmetry (A_S)'))}


add_noise='no'

# For major mergers the priors are [0.9,0.1] for minor mergers they are [0.7,0.3]
# It's also possible to adjust these as a way of making a more selective classification,
# i.e., this basically moves the intercept term in the LDA
priors_list=[[0.9,0.1]]#[0.75,0.25]]
plt.clf()
missclass_list=[]

# It would be possible to adjust this classification by providing a new table
# Here I'm using the combined major merger classification, I could also do this with the minor merger files
df = pd.io.parsers.read_csv(filepath_or_buffer='LDA_prep_predictors_all_combined_major.txt',header=[0],sep='\t')

# Not sure why, but I always need to add Shape Assy as a separate column
df.columns = [l for i,l in sorted(feature_dict.items())] + ['Shape Asymmetry']

df.dropna(how="all", inplace=True) # to drop the empty line at file-end




        
myr=[]
myr_non=[]
for j in range(len(df)):
    if df[['class label']].values[j][0]==0.0:
        myr_non.append(df[['Myr']].values[j][0])
    else:
        myr.append(df[['Myr']].values[j][0])

myr_non=sorted(list(set(myr_non)))
myr=sorted(list(set(myr)))
    
    
    
    
    


# There has got to be a better way to do this,
# Here I'm just creating a bunch of different functions
# to calculate all of the different cross-terms.
def gini_m20(row):
    return row['Gini']*row['M20']
def gini_C(row):
    return row['Gini']*row['Concentration (C)']
def gini_A(row):
    return row['Gini']*row['Asymmetry (A)']
def gini_S(row):
    return row['Gini']*row['Clumpiness (S)']
def gini_n(row):
    return row['Gini']*row['Sersic N']
def gini_A_S(row):
    return row['Gini']*row['Shape Asymmetry']

def M20_C(row):
    return row['M20']*row['Concentration (C)']
def M20_A(row):
    return row['M20']*row['Asymmetry (A)']
def M20_S(row):
    return row['M20']*row['Clumpiness (S)']
def M20_n(row):
    return row['M20']*row['Sersic N']
def M20_A_S(row):
    return row['M20']*row['Shape Asymmetry']

def C_A(row):
    return row['Concentration (C)']*row['Asymmetry (A)']
def C_S(row):
    return row['Concentration (C)']*row['Clumpiness (S)']
def C_n(row):
    return row['Concentration (C)']*row['Sersic N']
def C_A_S(row):
    return row['Concentration (C)']*row['Shape Asymmetry']


def A_S(row):
    return row['Asymmetry (A)']*row['Clumpiness (S)']
def A_n(row):
    return row['Asymmetry (A)']*row['Sersic N']
def A_A_S(row):
    return row['Asymmetry (A)']*row['Shape Asymmetry']

def S_n(row):
    return row['Clumpiness (S)']*row['Sersic N']
def S_A_S(row):
    return row['Clumpiness (S)']*row['Shape Asymmetry']

def n_A_S(row):
    return row['Sersic N']*row['Shape Asymmetry']

df['Gini*M20'] = df.apply(gini_m20,axis=1)
df['Gini*C'] = df.apply(gini_C,axis=1)
df['Gini*A'] = df.apply(gini_A,axis=1)
df['Gini*S'] = df.apply(gini_S,axis=1)
df['Gini*n'] = df.apply(gini_n,axis=1)
df['Gini*A_S'] = df.apply(gini_A_S,axis=1)

df['M20*C'] = df.apply(M20_C,axis=1)
df['M20*A'] = df.apply(M20_A,axis=1)
df['M20*S'] = df.apply(M20_S,axis=1)
df['M20*n'] = df.apply(M20_n,axis=1)
df['M20*A_S'] = df.apply(M20_A_S,axis=1)

df['C*A'] = df.apply(C_A,axis=1)
df['C*S'] = df.apply(C_S,axis=1)
df['C*n'] = df.apply(C_n,axis=1)
df['C*A_S'] = df.apply(C_A_S,axis=1)

df['A*S'] = df.apply(A_S,axis=1)
df['A*n'] = df.apply(A_n,axis=1)
df['A*A_S'] = df.apply(A_A_S,axis=1)

df['S*n'] = df.apply(S_n,axis=1)
df['S*A_S'] = df.apply(S_A_S,axis=1)

df['n*A_S'] = df.apply(n_A_S,axis=1)


# These are lists of all the different possible combinations of terms,
# The LDA's k-fold cross-validation will select which ones are needed.

ct_1=['Gini','Gini','Gini','Gini','Gini','Gini',
  'M20','M20','M20','M20','M20',
  'Concentration (C)','Concentration (C)','Concentration (C)','Concentration (C)',
  'Asymmetry (A)','Asymmetry (A)','Asymmetry (A)',
  'Clumpiness (S)','Clumpiness (S)',
  'Sersic N']
ct_2=['M20','Concentration (C)','Asymmetry (A)', 'Clumpiness (S)','Sersic N','Shape Asymmetry',
  'Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry',
  'Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry',
  'Clumpiness (S)','Sersic N','Shape Asymmetry',
  'Sersic N','Shape Asymmetry',
  'Shape Asymmetry']
term=['Gini*M20','Gini*C','Gini*A','Gini*S','Gini*n','Gini*A_S',
        'M20*C','M20*A','M20*S', 'M20*n', 'M20*A_S',
        'C*A','C*S','C*n','C*A_S',
       'A*S','A*n','A*A_S',
        'S*n','S*A_S',
       'n*A_S']




inputs=['Gini','M20','Concentration (C)','Asymmetry (A)','Clumpiness (S)','Sersic N','Shape Asymmetry',
        'Gini*M20','Gini*C','Gini*A','Gini*S','Gini*n','Gini*A_S',
        'M20*C','M20*A','M20*S', 'M20*n', 'M20*A_S', 
        'C*A','C*S','C*n','C*A_S',
       'A*S','A*n','A*A_S',
        'S*n','S*A_S',
       'n*A_S']
       


OG_length=len(inputs)

prev_input=[]
prev_input_here=[]
missclass=[]
missclass_e=[]
num_comps=[]
list_coef=[]
list_coef_std=[]
list_inter=[]
list_inter_std=[]
list_master=[]
list_master_confusion=[]
list_classes=[]
list_std_scale=[]
list_sklearn=[]
list_mean_non=[]

# It is important to set up the k-fold outside of the loop
kf = StratifiedKFold(n_splits=10, random_state=True, shuffle=True)

# Its time to begin the LDA!
# This code will go through and select which predictors are important for the classifcation.
# It will also determine their coefficients and associated uncertainties.
# The way this works is it starts with nothing (no predictors) selected and then tries to add one term
# at a time, going through all possible ones and comparing their misclassification errors.
# Then, if adding a term does not significantly improve the number of misclassifications,
# it does not add another term.
# If it selects a cross-term that contains two terms,
# I've required it to also include those two terms individually in the classification for
# interpretation's sake.
for o in range(len(inputs)):
    # So for each possible predictor in the input list, go through and
    # determine which added one minimizes the misclassification error and select that term
    coef_mean=[]
    coef_std=[]
    inter_mean=[]
    inter_std=[]
    coef_mean_std=[]
    accuracy=[]
    accuracy_e=[]
    inputs_this_step=[]
    confusion_master_this_step=[]
    master_this_step=[]
    classes_this_step=[]
    std_scale_this_step=[]
    sklearn_this_step=[]
    mean_non_this_step=[]



    #Now inputs is changing and you need to go through and choose a variable
    for k in range(len(inputs)):#Search through every one

        prev_input.append(inputs[k])
        
        # inputs_here contains the list of predictors that minimize the misclassification
        inputs_here=[]
        inputs_here.append(inputs[k])

        # If inputs[k] is a cross term and the prev_input doesn't contain it, add it:
        for m in range(len(term)):
            if inputs[k]==term[m]:
                #then you are going to search every term of prev_input and see if it is there
                #for n in range(len(prev_input)):
                if ct_1[m] not in prev_input:
                    prev_input.append(ct_1[m])

                    inputs_here.append(ct_1[m])
                if ct_2[m] not in prev_input:
                    prev_input.append(ct_2[m])
                    inputs_here.append(ct_2[m])


        #print('inputs heading into LDA',prev_input)

        X = df[prev_input].values







        y = df['class label'].values

        
        std_scale = preprocessing.StandardScaler().fit(X)
        
        
        std_scale_this_step.append(std_scale)
        # It helps to scale all the data points for each predictor to have a mean
        # of 0 and a std of 1:
        X = std_scale.transform(X)


        enc = LabelEncoder()
        label_encoder = enc.fit(y)
        y = label_encoder.transform(y) + 1


        label_dict = {1: 'NonMerger', 2: 'Merger'}
        

        # Time to initialize the LDA with the given priors:
        sklearn_lda = LDA(priors=priors_list[0], store_covariance=True)#store_covariance=False

        
        # This actually runs the LDA:
        X_lda_sklearn = sklearn_lda.fit_transform(X, y)
        sklearn_this_step.append(sklearn_lda)
        dec = sklearn_lda.score(X,y)
        prob = sklearn_lda.predict_proba(X)

        coef = sklearn_lda.coef_






        # Okay but to get the errors associated with the LDA coefficients, split up the dataset k times
        # I chose k=10:
        kf.get_n_splits(X, y)



        coef_list=[]
        inter_list=[]
        classes_list=[]
        confusion_master=[]
        y_test_master=[]
        pred_master=[]
        single_prediction=[]
        #sklearn_list=[]
        mean_non_list=[]
        count=0
        # Now for each of the k splits, run LDA individually to determine some estimate
        # of how much error is associated with these coefficients:
        for train_index, test_index in kf.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


            sklearn_lda = LDA( priors=priors_list[0],store_covariance=True)#store_covariance=False
            #sklearn_list.append(sklearn_lda)


            X_lda_sklearn = sklearn_lda.fit_transform(X_train, y_train)
            coef = sklearn_lda.coef_
            inter = sklearn_lda.intercept_

            # store the intercepts and coefficients for the LDA
            inter_list.append(inter)
            coef_list.append(coef)
            pred = sklearn_lda.predict(X_test)
            classes_list.append(sklearn_lda.classes_)
            confusion_master.append(confusion_matrix(pred,y_test))

            # This tells you the number of false negatives and false positives in the classification,
            # or the number of misclassifications
            single_prediction.append(confusion_matrix(pred,y_test)[1][0]+confusion_matrix(pred,y_test)[0][1])

            my_lists_none = []
            my_lists_merg = []


            for f in range(len(X_train)):
                if y_train[f]==1:
                    my_lists_none.append(X_lda_sklearn[f][0])
                    continue
                else:
                    my_lists_merg.append(X_lda_sklearn[f][0])

            
            mean_non_list.append(float((np.mean(my_lists_merg))+float(np.mean(my_lists_none)))/2)
            
        # Okay the misclassifications are contained here (the mean and error)
        accuracy.append(np.mean(single_prediction))#/(master[0][0]+master[1][0]+master[0][1]+master[1][1]))

        accuracy_e.append(np.std(single_prediction))
        inputs_this_step.append(np.array(prev_input))
        # Later, I will use the 'accuracy' and 'inputs_this_step' to determine the number of terms to use

              
        confusion_master_this_step.append(np.array((np.mean(confusion_master,axis=0)/np.sum(np.mean(confusion_master,axis=0))).transpose()))
        master_this_step.append(np.array(np.mean(confusion_master, axis=0).transpose()))
        #print('appending with this', np.array(prev_input))

        classes_this_step.append(np.array(classes_list))
        

        coef_mean.append(np.mean(coef_list, axis=0))
        coef_std.append(np.std(coef_list, axis=0))

        inter_mean.append(np.mean(inter_list, axis=0))
        inter_std.append(np.std(inter_list, axis=0))
        mean_non_this_step.append(np.mean(mean_non_list))

        # This removes the new input in order to go back and try all other possible inputs
        for m in range(len(inputs_here)):
            try:
                prev_input.remove(inputs_here[m])
            except ValueError:
                continue

    if accuracy_e[accuracy.index(min(accuracy))]<0.00001:
        break
   
    thing=(inputs_this_step[accuracy.index(min(accuracy))])
    prev_input_here.append(thing)

    # Going about finding the term that minimizes the misclassifications error (unfortunately I named this term 'accuracy', but its really a number of misclassifications, so you'll want to minimize it)
    first_A=min(accuracy)

    
    for m in range(len(thing)):

        prev_input.append(thing[m])

        try:
            # Remove it from the list of inputs so you don't try to add it again
            inputs.remove(thing[m])
        except ValueError:
            # A lot of the commented out stuff is from when I first built this code
            #print('~~~Runing into troubles')
            #print('inputs', inputs)
            #print('the thing to remove', thing[m])
            continue
    # Appending everything with the number of misclassifications and terms that were just selected.
    # Later I'll go back and grab the relevant index
    prev_input=list(set(prev_input))
    missclass.append(min(accuracy))
    #print('coef previous to selecting min', coef_mean)
    missclass_e.append(accuracy_e[accuracy.index(min(accuracy))])
    list_coef.append(coef_mean[accuracy.index(min(accuracy))])
    #print('coef list', coef_mean[accuracy.index(min(accuracy))])
    list_coef_std.append(coef_std[accuracy.index(min(accuracy))])

    list_inter.append(inter_mean[accuracy.index(min(accuracy))])
    list_inter_std.append(inter_std[accuracy.index(min(accuracy))])

    list_master.append(master_this_step[accuracy.index(min(accuracy))])
    list_master_confusion.append(confusion_master_this_step[accuracy.index(min(accuracy))])

    list_classes.append(classes_this_step[accuracy.index(min(accuracy))])

    list_std_scale.append(std_scale_this_step[accuracy.index(min(accuracy))])

    list_mean_non.append(mean_non_this_step[accuracy.index(min(accuracy))])

    num_comps.append(len(prev_input))#
    list_sklearn.append(sklearn_this_step[accuracy.index(min(accuracy))])
    #print('min index', accuracy.index(min(accuracy)))
    #if OG_length < len(:
    #    break

    if len(thing)==28:
        break

# Now figure out how many terms you want overall.
min_A=min(missclass)
min_comps=num_comps[missclass.index(min(missclass))]
for p in range(len(missclass)):
    missclass_list.append(missclass[p])
plt.plot(num_comps,missclass, color='black')
plt.scatter(num_comps,missclass, color='black')

plt.fill_between(num_comps,np.array(missclass)+np.array(missclass_e), np.array(missclass)-np.array(missclass_e), alpha=.5,color='pink')


min_index=locate_min(missclass)[1][0]


min_A=missclass[locate_min(missclass)[1][0]]
min_A_e=missclass_e[locate_min(missclass)[1][0]]
'''Now you need to use one standard error '''


for m in range(len(missclass)):
    if missclass[m] < (min_A+min_A_e):
        print('m',missclass[m],min_A+min_A_e)
        new_min_index=m
        break
    else:
        new_min_index=min_index

print('new min', new_min_index)
min_A=missclass[new_min_index]
min_A_e=missclass_e[new_min_index]
min_comps=num_comps[new_min_index]

plt.scatter(min_comps, min_A,marker='x', color='black', zorder=100)
plt.xlabel('Number of terms')
plt.ylabel('Number of Misclassifications')
plt.savefig('determine_n_comps.png', dpi=1000)

# Print out all the slected terms and coefficients
inputs_all=prev_input_here[new_min_index]#:new_min_index+1]
#print(prev_input)
print('terms before adding in necessary', inputs_all)
print('coefficients', list_coef[new_min_index])
print('coefficient std', list_coef_std[new_min_index])




print('missclass', missclass)
print('missclass_e', missclass_e)

print('intercept', list_inter[new_min_index])
print('intercept std', list_inter_std[new_min_index])

'''Now Im making the confusion matrix'''
print('list_master_confusion', list_master_confusion[new_min_index])
print('list_master', list_master[new_min_index])

print('standard scale min', list_std_scale[new_min_index])
print('mean of that', list_std_scale[new_min_index].mean_)
print('std of that', list_std_scale[new_min_index].var_)

print('decision boundary', list_mean_non[new_min_index])


# Option to make a confusion matrix:
'''plt.clf()
fig=plt.figure()#figsize=(6,6)
plot_confusion_matrix(list_master_confusion[new_min_index], sklearn_lda.classes_, title='Normalized Confusion Matrix')
plt.savefig('Confusion_matrix.pdf')'''
#This is from this website: http://www.science.smith.edu/~jcrouser/SDS293/labs/lab5-py.html
plt.clf()
sns.set_style("darkgrid")

master=list_master[new_min_index]

# This is how to emeasure accuracy, precision, etc...
print('~~~Accuracy~~~')
print((master[1][1]+master[0][0])/(master[0][0]+master[1][0]+master[0][1]+master[1][1]))
print('~~~Precision~~~')
print(master[1][1]/(master[0][1]+master[1][1]))#TP/(TP+FP)
print('~~~Recall~~~')
print(master[1][1]/(master[1][0]+master[1][1]))#TP/(TP+FN)
print('~~~F1~~~')
print((2*master[1][1])/(master[0][1]+master[1][0]+2*master[1][1]))#2TP/(2TP+FP+FN)







# This section of code is to create a file for use in the Novice version of the code
# It creates a .txt file that lists the coefficients that were selected, their
# LDA values and stds.
file=open('Inputs_MergerMonger_Novice.txt','w')
for j in range(len(inputs_all)):
    
    file.write((inputs_all[j])+'\t')
file.write('\n')



for j in range(len(list_coef[new_min_index])):
    
    file.write(str(list_coef[new_min_index][j])+'\t')
file.write('\n')
file.write(str(list_inter[new_min_index])+'\n')

for j in range(len(list_std_scale[new_min_index].mean_)):
    file.write(str((list_std_scale[new_min_index].mean_)[j])+'\t')
file.write('\n')

for j in range(len(list_std_scale[new_min_index].mean_)):
    file.write(str(np.sqrt(list_std_scale[new_min_index].var_)[j])+'\t')
file.write('\n')

print('intercepts', list_inter[new_min_index])
#print('DECISION BOUNDARY', mean_non)
file.write(str(list_mean_non[new_min_index])+'\n')
file.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This next section creates a number of beautiful contour plots,
# This is for assessing if things make sense based on prior
# knowledge of the way these imaging predictors behave.
# It should really be its own function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
X_list_here=df[inputs_all].values
#and standardize it
std_mean=[float(x) for x in list_std_scale[new_min_index].mean_]
std_std=[float(np.sqrt(x)) for x in list_std_scale[new_min_index].var_]
X_std_here=[]
for j in range(len(X_list_here)):
    X_std_here.append(list((X_list_here[j]-std_mean)/std_std))


#print(list_sklearn[new_min_index].predict(X_std_here))
LDA_ID=list_sklearn[new_min_index].predict(X_std_here)

nonmerg_gini=[]
nonmerg_m20=[]

merg_gini=[]
merg_m20=[]

nonmerg_gini_LDA=[]
nonmerg_m20_LDA=[]

merg_gini_LDA=[]
merg_m20_LDA=[]


nonmerg_C_LDA=[]
nonmerg_A_LDA=[]
nonmerg_S_LDA=[]
nonmerg_n_LDA=[]
nonmerg_A_S_LDA=[]

merg_C_LDA=[]
merg_A_LDA=[]
merg_S_LDA=[]
merg_n_LDA=[]
merg_A_S_LDA=[]

LDA_LDA_merg=[]
LDA_LDA_nonmerg=[]
coef = list_coef[new_min_index]
inter = list_inter[new_min_index]

# Go through and grab the values of all the predictors for the merging and nonmerging pop
# of simulated galaxies:
for j in range(len(df)):
    if df['class label'].values[j]==0:
        nonmerg_gini.append(df['Gini'].values[j])
        nonmerg_m20.append(df['M20'].values[j])
        # I think to get the LDA values themselves you'll need to grab the coefficients
        # and the intercept
        
        LDA_LDA_nonmerg.append(np.sum(coef*X_std_here[j]+inter))
    if df['class label'].values[j]==1:
        merg_gini.append(df['Gini'].values[j])
        merg_m20.append(df['M20'].values[j])
        LDA_LDA_merg.append(np.sum(coef*X_std_here[j]+inter))
    
    if LDA_ID[j]==1:#then its a nonmerger
        nonmerg_gini_LDA.append(df['Gini'].values[j])
        nonmerg_m20_LDA.append(df['M20'].values[j])
        nonmerg_C_LDA.append(df['Concentration (C)'].values[j])
        nonmerg_A_LDA.append(df['Asymmetry (A)'].values[j])
        nonmerg_S_LDA.append(df['Clumpiness (S)'].values[j])
        nonmerg_n_LDA.append(df['Sersic N'].values[j])
        nonmerg_A_S_LDA.append(df['Shape Asymmetry'].values[j])
        
    if LDA_ID[j]==2:#then its a nonmerger
        merg_gini_LDA.append(df['Gini'].values[j])
        merg_m20_LDA.append(df['M20'].values[j])
        merg_C_LDA.append(df['Concentration (C)'].values[j])
        merg_A_LDA.append(df['Asymmetry (A)'].values[j])
        merg_S_LDA.append(df['Clumpiness (S)'].values[j])
        merg_n_LDA.append(df['Sersic N'].values[j])
        merg_A_S_LDA.append(df['Shape Asymmetry'].values[j])
        
        

dashed_line_x=np.linspace(-0.5,-3,100)
dashed_line_y=[-0.14*x + 0.33 for x in dashed_line_x]



merg_m20_LDA=np.array(merg_m20_LDA)
merg_gini_LDA=np.array(merg_gini_LDA)
merg_C_LDA=np.array(merg_C_LDA)
merg_A_LDA=np.array(merg_A_LDA)
merg_S_LDA=np.array(merg_S_LDA)
merg_n_LDA=np.array(merg_n_LDA)
merg_A_S_LDA=np.array(merg_A_S_LDA)

nonmerg_m20_LDA=np.array(nonmerg_m20_LDA)
nonmerg_gini_LDA=np.array(nonmerg_gini_LDA)
nonmerg_C_LDA=np.array(nonmerg_C_LDA)
nonmerg_A_LDA=np.array(nonmerg_A_LDA)
nonmerg_S_LDA=np.array(nonmerg_S_LDA)
nonmerg_n_LDA=np.array(nonmerg_n_LDA)
nonmerg_A_S_LDA=np.array(nonmerg_A_S_LDA)



merg_m20=np.array(merg_m20)
merg_gini=np.array(merg_gini)
nonmerg_m20=np.array(nonmerg_m20)
nonmerg_gini=np.array(nonmerg_gini)


plt.clf()
fig=plt.figure()
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)


ax1=fig.add_subplot(121)
ax1.set_title('Mergers', loc='right')
ax1.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

#sns.kdeplot(merg_m20, merg_gini, cmap="Reds", shade=True,shade_lowest=False)# bw=.15)

sns.kdeplot(merg_m20_LDA, merg_gini_LDA, cmap="Reds_r", shade=True,shade_lowest=False)# bw=.15)
ax1.set_xlim([0,-3])
ax1.set_ylim([0.2,0.8])#ax1.set_ylim([0.3,0.8])
ax1.set_xlabel(r'M$_{20}$')
ax1.set_ylabel(r'Gini')
ax1.set_aspect(abs(3)/abs(0.6))

ax2=fig.add_subplot(122)
ax2.set_title('Nonmergers', loc='right')
ax2.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

#sns.kdeplot(nonmerg_m20, nonmerg_gini, cmap="Blues", shade=True,shade_lowest=False)
sns.kdeplot(nonmerg_m20_LDA, nonmerg_gini_LDA, cmap="Blues_r", shade=True,shade_lowest=False)

ax2.set_xlim([0,-3])
ax2.set_ylim([0.2,0.8])#ax1.set_ylim([0.3,0.8])
ax2.set_xlabel(r'M$_{20}$')
ax2.set_ylabel(r'Gini')
ax2.set_aspect(abs(3)/abs(0.6))

plt.savefig('gini_m20_contour_LDA.pdf')

STOP
plt.clf()
fig=plt.figure()
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)


ax1=fig.add_subplot(121)

ax1.set_xlim([0,1])
ax1.set_ylim([0,6])#ax1.set_ylim([0.3,0.8])
ax1.set_xlabel(r'A')
ax1.set_ylabel(r'C')
ax1.set_aspect(1/6)
ax1.set_title('Mergers', loc='right')
plt.axvline(x=0.35, ls='--', color='black')
sns.kdeplot(merg_A_LDA, merg_C_LDA, cmap="Reds_r", shade=True,shade_lowest=False)

ax2=fig.add_subplot(122)

ax2.set_xlim([0,1])
ax2.set_ylim([0,6])#ax1.set_ylim([0.3,0.8])
ax2.set_xlabel(r'A')
ax2.set_ylabel(r'C')
ax2.set_aspect(1/6)
ax2.set_title('Nonmergers', loc='right')
plt.axvline(x=0.35, ls='--', color='black')
sns.kdeplot(nonmerg_A_LDA, nonmerg_C_LDA, cmap="Blues_r", shade=True,shade_lowest=False)
plt.savefig('C_A_contour_LDA_major.pdf')



plt.clf()
plt.clf()
fig=plt.figure()
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)


ax1=fig.add_subplot(121)
sns.kdeplot(merg_S_LDA, merg_C_LDA, cmap="Reds_r", shade=True,shade_lowest=False)
ax1.set_xlim([0,1])
ax1.set_ylim([0,6])#ax1.set_ylim([0.3,0.8])
ax1.set_xlabel(r'S')
ax1.set_ylabel(r'C')
ax1.set_aspect(1/6)


#ax1.legend(loc='lower center',
#          ncol=2)
ax1.set_title('Mergers', loc='right')

ax2=fig.add_subplot(122)
sns.kdeplot(nonmerg_S_LDA, nonmerg_C_LDA, cmap="Blues_r", shade=True,shade_lowest=False)
ax2.set_xlim([0,1])
ax2.set_ylim([0,6])#ax1.set_ylim([0.3,0.8])
ax2.set_xlabel(r'S')
ax2.set_ylabel(r'C')
ax2.set_aspect(1/6)
ax2.set_title('Nonmergers', loc='right')
plt.savefig('S_C_contour_LDA.pdf')

plt.clf()

'''

Now for n-A_S plot

'''
plt.clf()
fig=plt.figure()
ax1=fig.add_subplot(121)


sns.kdeplot(merg_A_S_LDA, merg_n_LDA, cmap="Reds_r", shade=True,shade_lowest=False)


ax1.set_xlim([0,1])
ax1.set_ylim([0,4])#ax1.set_ylim([0.3,0.8])
ax1.set_xlabel(r'$A_S$')
ax1.set_ylabel(r'$n$')
ax1.set_aspect(1/4)

#ax1.legend(loc='lower center',
#          ncol=2)
ax1.set_title('Mergers', loc='right')

#ax1.annotate(str(NAME), xy=(0.03,1.05),xycoords='axes fraction',size=15)
plt.axvline(x=0.2, ls='--', color='black')


ax2=fig.add_subplot(122)
sns.kdeplot(nonmerg_A_S_LDA, nonmerg_n_LDA, cmap="Blues_r", shade=True,shade_lowest=False)


ax2.set_xlim([0,1])
ax2.set_ylim([0,4])#ax1.set_ylim([0.3,0.8])
ax2.set_xlabel(r'$A_S$')
ax2.set_ylabel(r'$n$')
ax2.set_aspect(1/4)
plt.axvline(x=0.2, ls='--', color='black')

#ax1.legend(loc='lower center',
#          ncol=2)
ax2.set_title('Nonmergers', loc='right')
plt.savefig('n_A_S_contour_LDA.pdf')




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now load in the MaNGA table (for real galaxies) and classify
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


feature_dict2 = {i:label for i,label in zip(
            range(12),
              ('Counter',
              'ID',
              'Merger?',
              '# Bulges',
               'Sep',
               'Flux Ratio',
              'Gini',
              'M20',
              'Concentration (C)',
              'Asymmetry (A)',
              'Clumpiness (S)',
              'Sersic N',
            'Shape Asymmetry'))}

df2 = pd.io.parsers.read_table(filepath_or_buffer='LDA_img_statmorph_MaNGA_mytable_troubleshoot.txt',header=[0],sep='\t')
#LDA_img_statmorph_Fu_mergers.txt
#LDA_img_statmorph_rando_MaNGA.txt
df2.columns = [l for i,l in sorted(feature_dict2.items())] + ['Shape Asymmetry']




df2.dropna(how="all", inplace=True) # to drop the empty line at file-end

df2['Gini*M20'] = df2.apply(gini_m20,axis=1)
df2['Gini*C'] = df2.apply(gini_C,axis=1)
df2['Gini*A'] = df2.apply(gini_A,axis=1)
df2['Gini*S'] = df2.apply(gini_S,axis=1)
df2['Gini*n'] = df2.apply(gini_n,axis=1)
df2['Gini*A_S'] = df2.apply(gini_A_S,axis=1)

df2['M20*C'] = df2.apply(M20_C,axis=1)
df2['M20*A'] = df2.apply(M20_A,axis=1)
df2['M20*S'] = df2.apply(M20_S,axis=1)
df2['M20*n'] = df2.apply(M20_n,axis=1)
df2['M20*A_S'] = df2.apply(M20_A_S,axis=1)

df2['C*A'] = df2.apply(C_A,axis=1)
df2['C*S'] = df2.apply(C_S,axis=1)
df2['C*n'] = df2.apply(C_n,axis=1)
df2['C*A_S'] = df2.apply(C_A_S,axis=1)

df2['A*S'] = df2.apply(A_S,axis=1)
df2['A*n'] = df2.apply(A_n,axis=1)
df2['A*A_S'] = df2.apply(A_A_S,axis=1)

df2['S*n'] = df2.apply(S_n,axis=1)
df2['S*A_S'] = df2.apply(S_A_S,axis=1)

df2['n*A_S'] = df2.apply(n_A_S,axis=1)

X_gal = df2[inputs_all].values
print('X_gal', X_gal)




std_mean=[float(x) for x in list_std_scale[new_min_index].mean_]
std_std=[float(np.sqrt(x)) for x in list_std_scale[new_min_index].var_]
X_std=[]
testing_C=[]
testing_A=[]
testing_Gini=[]
testing_M20=[]
testing_n=[]
testing_A_S=[]
for j in range(len(X_gal)):
    X_std.append(list((X_gal[j]-std_mean)/std_std))
    testing_Gini.append(df2['Gini'].values[j])
    testing_M20.append(df2['M20'].values[j])
    testing_C.append(df2['Concentration (C)'].values[j])
    testing_A.append(df2['Asymmetry (A)'].values[j])
    testing_n.append(df2['Sersic N'].values[j])
    testing_A_S.append(df2['Shape Asymmetry'].values[j])

# Make some quick histograms to see if the values make sense
plt.clf()
plt.hist(testing_Gini)
plt.savefig('hist_Gini.pdf')
plt.clf()
plt.hist(testing_M20)
plt.savefig('hist_M20.pdf')
plt.clf()
plt.hist(testing_C)
plt.savefig('hist_C.pdf')
plt.clf()
plt.hist(testing_A)
plt.savefig('hist_A.pdf')
plt.clf()
plt.hist(testing_n)
plt.savefig('hist_n.pdf')
plt.clf()
plt.hist(testing_A_S)
plt.savefig('hist_A_S.pdf')

print('what are we even dealing with?', list_sklearn[new_min_index])
STOP

# Use the LDA from the simulated galaxies to classify this new table:
print(list_sklearn[new_min_index].predict(X_std))
print(list_sklearn[new_min_index].predict_proba(X_std))
# Usually, I make the classifications this way:
classifications=list_sklearn[new_min_index].predict(X_std)

# But if you want a different cut-off, ie a more conservative classification
# that only calls galaxies merger if p_merg > 0.75, then use this:
classifications=[]
print('shape', np.shape(X_std), 'len', len(X_std))
for j in range(len(X_std)):
    #print(list_sklearn[new_min_index].predict_proba(X_std)[j][1])
    # This is where you set the new threshold
    if list_sklearn[new_min_index].predict_proba(X_std)[j][1] > 0.9:
        classifications.append(2)
    else:
        classifications.append(1)
        

merg_gini_LDA_out=[]
merg_m20_LDA_out=[]
merg_C_LDA_out=[]
merg_A_LDA_out=[]
merg_S_LDA_out=[]
merg_n_LDA_out=[]
merg_A_S_LDA_out=[]

nonmerg_gini_LDA_out=[]
nonmerg_m20_LDA_out=[]
nonmerg_C_LDA_out=[]
nonmerg_A_LDA_out=[]
nonmerg_S_LDA_out=[]
nonmerg_n_LDA_out=[]
nonmerg_A_S_LDA_out=[]

merg_name_list=[]
nonmerg_name_list=[]
LDA_value=[]
LDA_value_merg=[]
LDA_value_nonmerg=[]
# I went through and grabbed all the values of the merging and nonmerging
# MaNGA galaxies and used them to make comparative contour plots
for j in range(len(classifications)):
    if classifications[j]==2:#merger
        
        merg_gini_LDA_out.append(df2['Gini'].values[j])
        merg_m20_LDA_out.append(df2['M20'].values[j])
        merg_C_LDA_out.append(df2['Concentration (C)'].values[j])
        merg_A_LDA_out.append(df2['Asymmetry (A)'].values[j])
        merg_S_LDA_out.append(df2['Clumpiness (S)'].values[j])
        merg_n_LDA_out.append(df2['Sersic N'].values[j])
        merg_A_S_LDA_out.append(df2['Shape Asymmetry'].values[j])
        merg_name_list.append(df2['ID'].values[j])
        # The LDA value is the sum of all of the stardardized Xs multiplied by
        # the LDA coefficients added to the intercept term
        LDA_value.append(np.sum(coef*X_std[j]+inter))
        LDA_value_merg.append(np.sum(coef*X_std[j]+inter))
        
    if classifications[j]==1:#nonmerger
        #print(df2['ID'].values[j])
        #print(X_std[j])
        #print(np.sum(X_std[j]))
        nonmerg_gini_LDA_out.append(df2['Gini'].values[j])
        nonmerg_m20_LDA_out.append(df2['M20'].values[j])
        nonmerg_C_LDA_out.append(df2['Concentration (C)'].values[j])
        nonmerg_A_LDA_out.append(df2['Asymmetry (A)'].values[j])
        nonmerg_S_LDA_out.append(df2['Clumpiness (S)'].values[j])
        nonmerg_n_LDA_out.append(df2['Sersic N'].values[j])
        nonmerg_A_S_LDA_out.append(df2['Shape Asymmetry'].values[j])
        nonmerg_name_list.append(df2['ID'].values[j])
        LDA_value.append(np.sum(coef*X_std[j]+inter))
        LDA_value_nonmerg.append(np.sum(coef*X_std[j]+inter))
    


# Compare the LDA values of your population of MaNGA galaxies
# to that of the simulated galaxies:
min_LDA = min(min(LDA_LDA_nonmerg), min(LDA_value_nonmerg))
max_LDA = max(max(LDA_LDA_merg), max(LDA_value_merg))
print('min',min_LDA)
plt.clf()
fig=plt.figure()
ax0 = fig.add_subplot(211)
ax0.hist(LDA_value_nonmerg, label='MaNGA Nonmergers', alpha=0.5, bins=20)
ax0.hist(LDA_value_merg, label='MaNGA Mergers', alpha=0.5, bins=20)
ax0.set_xlim([min_LDA,max_LDA])
plt.legend()
ax1 = fig.add_subplot(212)
ax1.hist(LDA_LDA_nonmerg, label='LDA Nonmergers', alpha=0.5, bins=20)
ax1.hist(LDA_LDA_merg, label='LDA Mergers', alpha=0.5, bins=20)
ax1.set_xlim([min_LDA, max_LDA])
plt.legend()

plt.savefig('joint_hist_LDA.pdf')


plt.clf()
fig=plt.figure()
ax0 = fig.add_subplot(211)
ax0.hist(nonmerg_A_LDA_out, label='MaNGA Nonmergers', alpha=0.5)
ax0.hist(merg_A_LDA_out, label='MaNGA Mergers', alpha=0.5)
ax0.set_xlim([-0.5, 1.5])
plt.legend()
ax1 = fig.add_subplot(212)
ax1.hist(nonmerg_A_LDA, label='LDA Nonmergers', alpha=0.5)
ax1.hist(merg_A_LDA, label='LDA Mergers', alpha=0.5)
ax1.set_xlim([-0.5, 1.5])
plt.legend()
plt.title('Asymmetry Distributions')
plt.savefig('joint_hist_A.pdf')

plt.clf()
fig=plt.figure()
ax0 = fig.add_subplot(211)
ax0.hist(nonmerg_A_S_LDA_out, label='MaNGA Nonmergers', alpha=0.5)
ax0.hist(merg_A_S_LDA_out, label='MaNGA Mergers', alpha=0.5)
#ax0.set_xlim([-0.5, 1.5])
plt.legend()
ax1 = fig.add_subplot(212)
ax1.hist(nonmerg_A_S_LDA, label='LDA Nonmergers', alpha=0.5)
ax1.hist(merg_A_S_LDA, label='LDA Mergers', alpha=0.5)
#ax1.set_xlim([-0.5, 1.5])
plt.legend()
plt.title('Shape Asymmetry Distributions')
plt.savefig('joint_hist_A_S.pdf')

# Print the precentage of things that are merging and nonmerging
print('percent nonmerg',len(nonmerg_name_list)/(len(nonmerg_name_list)+len(merg_name_list)))
print('percent merg',len(merg_name_list)/(len(nonmerg_name_list)+len(merg_name_list)))

# Start making the contour kdplots with scattered values of the MaNGA galaxies superimpsed
plt.clf()
fig=plt.figure()
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)


ax1=fig.add_subplot(121)
ax1.set_title('Mergers', loc='right')
ax1.plot(dashed_line_x, dashed_line_y, ls='--', color='black')


#sns.kdeplot(merg_m20, merg_gini, cmap="Reds", shade=True,shade_lowest=False)# bw=.15)

sns.kdeplot(merg_m20_LDA, merg_gini_LDA, cmap="Reds_r", shade=True,shade_lowest=False)# bw=.15)
im1=ax1.scatter(merg_m20_LDA_out, merg_gini_LDA_out, color='red',edgecolors='black')
#for j in range(len(merg_name_list)):
#    if merg_name_list[j]=='8309-12701':
#        ax1.annotate(merg_name_list[j],(merg_m20_LDA_out[j], merg_gini_LDA_out[j]))

ax1.set_xlim([0,-3])
ax1.set_ylim([0.2,0.8])#ax1.set_ylim([0.3,0.8])
ax1.set_xlabel(r'M$_{20}$')
ax1.set_ylabel(r'Gini')
ax1.set_aspect(abs(3)/abs(0.6))

ax2=fig.add_subplot(122)
ax2.set_title('Nonmergers', loc='right')
ax2.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

#sns.kdeplot(nonmerg_m20, nonmerg_gini, cmap="Blues", shade=True,shade_lowest=False)
sns.kdeplot(nonmerg_m20_LDA, nonmerg_gini_LDA, cmap="Blues_r", shade=True,shade_lowest=False)
im2=ax2.scatter(nonmerg_m20_LDA_out, nonmerg_gini_LDA_out, color='blue',edgecolors='black')
#for j in range(len(nonmerg_gini_LDA_out)):
#    ax2.annotate(name_list[j],(nonmerg_m20_LDA_out[j], nonmerg_gini_LDA_out[j]))

ax2.set_xlim([0,-3])
ax2.set_ylim([0.2,0.8])#ax1.set_ylim([0.3,0.8])
ax2.set_xlabel(r'M$_{20}$')
ax2.set_ylabel(r'Gini')
ax2.set_aspect(abs(3)/abs(0.6))

plt.savefig('gini_m20_contour_SDSS_ellip_major.pdf')


plt.clf()
fig=plt.figure()
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)


ax1=fig.add_subplot(121)

ax1.set_xlim([-0.2,1])
ax1.set_ylim([0,6])#ax1.set_ylim([0.3,0.8])
ax1.set_xlabel(r'A')
ax1.set_ylabel(r'C')
ax1.set_aspect(1.2/6)
ax1.set_title('Mergers', loc='right')
plt.axvline(x=0.35, ls='--', color='black')
sns.kdeplot(merg_A_LDA, merg_C_LDA, cmap="Reds_r", shade=True,shade_lowest=False)
im1=ax1.scatter(merg_A_LDA_out, merg_C_LDA_out, color='red',edgecolors='black')
#for j in range(len(merg_name_list)):
#    if merg_name_list[j]=='8309-12701':
#        ax1.annotate(merg_name_list[j],(merg_A_LDA_out[j], merg_C_LDA_out[j]))

ax2=fig.add_subplot(122)

ax2.set_xlim([-0.2,1])
ax2.set_ylim([0,6])#ax1.set_ylim([0.3,0.8])
ax2.set_xlabel(r'A')
ax2.set_ylabel(r'C')
ax2.set_aspect(1.2/6)
ax2.set_title('Nonmergers', loc='right')
plt.axvline(x=0.35, ls='--', color='black')
sns.kdeplot(nonmerg_A_LDA, nonmerg_C_LDA, cmap="Blues_r", shade=True,shade_lowest=False)
im2=ax2.scatter(nonmerg_A_LDA_out, nonmerg_C_LDA_out, color='blue',edgecolors='black')
#for j in range(len(nonmerg_A_S_LDA_out)):
#    ax2.annotate(name_list[j],(nonmerg_A_LDA_out[j], nonmerg_C_LDA_out[j]))

plt.savefig('C_A_contour_SDSS_ellip_major.pdf')


plt.clf()
plt.clf()
fig=plt.figure()
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)


ax1=fig.add_subplot(121)
sns.kdeplot(merg_S_LDA, merg_C_LDA, cmap="Reds_r", shade=True,shade_lowest=False)
im1=ax1.scatter(merg_S_LDA_out, merg_C_LDA_out, color='red',edgecolors='black')
#for j in range(len(merg_name_list)):
#    if merg_name_list[j]=='8309-12701':
#        ax1.annotate(merg_name_list[j],(merg_S_LDA_out[j], merg_C_LDA_out[j]))

ax1.set_xlim([0,1])
ax1.set_ylim([0,6])#ax1.set_ylim([0.3,0.8])
ax1.set_xlabel(r'S')
ax1.set_ylabel(r'C')
ax1.set_aspect(1/6)


#ax1.legend(loc='lower center',
#          ncol=2)
ax1.set_title('Mergers', loc='right')

ax2=fig.add_subplot(122)
sns.kdeplot(nonmerg_S_LDA, nonmerg_C_LDA, cmap="Blues_r", shade=True,shade_lowest=False)
im2=ax2.scatter(nonmerg_S_LDA_out, nonmerg_C_LDA_out, color='blue',edgecolors='black')
#for j in range(len(nonmerg_S_LDA_out)):
#    ax2.annotate(name_list[j],(nonmerg_S_LDA_out[j], nonmerg_C_LDA_out[j]))

ax2.set_xlim([0,1])
ax2.set_ylim([0,6])#ax1.set_ylim([0.3,0.8])
ax2.set_xlabel(r'S')
ax2.set_ylabel(r'C')
ax2.set_aspect(1/6)
ax2.set_title('Nonmergers', loc='right')
plt.savefig('S_C_contour_SDSS_CS_major.pdf')

plt.clf()

'''

Now for n-A_S plot

'''
plt.clf()
fig=plt.figure()
ax1=fig.add_subplot(121)


sns.kdeplot(merg_A_S_LDA, merg_n_LDA, cmap="Reds_r", shade=True,shade_lowest=False)
im1=ax1.scatter(merg_A_S_LDA_out, merg_n_LDA_out, color='red',edgecolors='black')
#for j in range(len(merg_name_list)):
#    if merg_name_list[j]=='8309-12701':
#        ax1.annotate(merg_name_list[j],(merg_A_S_LDA_out[j], nonmerg_n_LDA_out[j]))

for j in range(len(merg_A_S_LDA_out)):
    ax1.annotate(merg_name_list[j],(merg_A_S_LDA_out[j], merg_n_LDA_out[j]))


ax1.set_xlim([0,1])
ax1.set_ylim([0,4])#ax1.set_ylim([0.3,0.8])
ax1.set_xlabel(r'$A_S$')
ax1.set_ylabel(r'$n$')
ax1.set_aspect(1/4)

#ax1.legend(loc='lower center',
#          ncol=2)
ax1.set_title('Mergers', loc='right')

#ax1.annotate(str(NAME), xy=(0.03,1.05),xycoords='axes fraction',size=15)
plt.axvline(x=0.2, ls='--', color='black')


ax2=fig.add_subplot(122)
sns.kdeplot(nonmerg_A_S_LDA, nonmerg_n_LDA, cmap="Blues_r", shade=True,shade_lowest=False)
im2=ax2.scatter(nonmerg_A_S_LDA_out, nonmerg_n_LDA_out, color='blue',edgecolors='black')


ax2.set_xlim([0,1])
ax2.set_ylim([0,4])#ax1.set_ylim([0.3,0.8])
ax2.set_xlabel(r'$A_S$')
ax2.set_ylabel(r'$n$')
plt.axvline(x=0.2, ls='--', color='black')
ax2.set_aspect(1/4)

#ax1.legend(loc='lower center',
#          ncol=2)
ax2.set_title('Nonmergers', loc='right')
plt.savefig('n_A_S_contour_SDSS_ellip_major.pdf')






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# I wrote this next section to go about making nice panel plots of the galaxies
# with their imaging predictor values and classifications
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


sns.set_style("white")
os.chdir(os.path.expanduser('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/Identify_MaNGA_SDSS_mergers_imaging'))
dapall=fits.open('/Users/beckynevin/CfA_Code/Kinematic_ML/dapall-v2_5_3-2.3.0.fits')
drpall=fits.open('/Users/beckynevin/Clone_Docs_old_mac/Backup_My_Book/My_Passport_backup/Kinematics_and_Imaging_Merger_Identification/drpall-v2_4_3.fits')
print(len(df2),int(np.sqrt(len(df2))))

'''First, an option for just plotting these individually'''

'''for p in range(len(df2)):#len(df2)):
    plt.clf()
    gal_id=df2[['ID']].values[p][0]
    gal_class=list_sklearn[new_min_index].predict(X_std)[p]
    if gal_class==2:
        gal_name='Merger'
    else:
        gal_name='Nonmerger'
    gal_prob=list_sklearn[new_min_index].predict_proba(X_std)[p]
    
    print(os.getcwd())
    im=fits.open('imaging/out_'+str(gal_id)+'.fits')
    camera_data=(im[1].data/0.005) 
    
    plt.imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(3), vmax=10**(6.5)), cmap='afmhot')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
    plt.annotate('GalaxyZoo Elliptical'+'\n'+r'p$_{\mathrm{merg, major}} = $'+str(round(gal_prob[1],2))+'\n'+'LDA '+gal_name, xycoords='axes fraction',xy=(0.05,0.8),color='white', size=20)#,#ha="center", va="top",
#                bbox=dict(boxstyle="round", fc="1.0"))
    plt.axis('off')
 #   plt.colorbar()
    plt.tight_layout()
    plt.savefig('SDSS_superclean_ellip_major_'+str(gal_id)+'.pdf')'''
    


# Now, this will be an option to make panel plots for each
# that show the values of all the predictors in terms of the distribution of
# simulated galaxies

def second_smallest(numbers):
    if (len(numbers)<2):
      return
    if ((len(numbers)==2)  and (numbers[0] == numbers[1]) ):
      return
    dup_items = set()
    uniq_items = []
    for x in numbers:
      if x not in dup_items:
        uniq_items.append(x)
        dup_items.add(x)
    uniq_items.sort()
    return  uniq_items[1]

for p in range(len(df2)):#len(df2)):
    plt.clf()
    gal_id=df2[['ID']].values[p][0]
    #gal_class=list_sklearn[new_min_index].predict(X_std)[p]
    # Again change this if you are doing a classification that is more conservative:
    gal_class=classifications[p]
    if gal_class==2:
        gal_name='Merger'
    else:
        gal_name='Nonmerger'
    gal_prob=list_sklearn[new_min_index].predict_proba(X_std)[p]
    
    # So for this galaxy, coef*X_std[p] + inter will give you what sums to create the LDA
    # My question is - is there a way to show the relative importance of this array
    LDA_array = coef*X_std[p] + inter
    #print(inputs_all)
    #print(LDA_array)
    
    # I was thinking, determine which terms are most important if its nonmerging these
    # will be the most negative terms to coef*X_std[p]
    sorted = np.sort(LDA_array)
    #print('sorted', sorted)
    if gal_class==2:
        #print('trying to find the index of this', list(LDA_array))
        #print('this is the value', sorted[0][-1])
        i_1 = np.where(LDA_array == sorted[0][-1])[1][0]
        i_2 = np.where(LDA_array == sorted[0][-2])[1][0]
        #print('this is the index', i_1, i_2)
    else:
        i_1 = np.where(LDA_array == sorted[0][0])[1][0]
        i_2 = np.where(LDA_array == sorted[0][1])[1][0]
        
    #print('first most important', inputs_all[i_1], LDA_array[0][i_1])
    #print('second most important', inputs_all[i_2], LDA_array[0][i_2])
    
    im=fits.open('imaging/out_'+str(gal_id)+'.fits')
    camera_data=(im[1].data/0.005)
    

    fig = plt.figure()
    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
    #plt.subplot(grid[0, 0])
    #plt.subplot(grid[0, 1:])
    #plt.subplot(grid[1, :2])
    #plt.subplot(grid[1, 2])
    ax0 = plt.subplot(grid[0,0])
    
    im0 = ax0.imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(3), vmax=10**(6.5)), cmap='afmhot')#vmin=10**(1), vmax=10**(4)norm=colors.LogNorm(vmin=10**(-1), vmax=10**(2.5))
    ax0.annotate(r'p$_{\mathrm{merg, major}} = $'+str(round(gal_prob[1],2))+'\n'+'LDA '+gal_name, xycoords='axes fraction',xy=(0.05,0.77),color='white', size=9)#,#ha="center", va="top",

    ax0.axis('off')
    
    ax3 = plt.subplot(grid[1,:2])
    #fig.add_subplot(234)
    ax3.hist(LDA_LDA_nonmerg, label='Nonmergers', alpha=0.5, bins=10)
    ax3.hist(LDA_LDA_merg, label='Mergers', alpha=0.5, bins=10)
    ax3.annotate('1st Coef = '+str(inputs_all[i_1])+' '+str(round(LDA_array[0][i_1],1)), xy=(0.05,0.9), xycoords='axes fraction')
    ax3.annotate('2nd Coef = '+str(inputs_all[i_2])+' '+str(round(LDA_array[0][i_2],1)), xy=(0.05,0.8), xycoords='axes fraction')
    
    #ax3.set_xlim([-50,50])
    if gal_class==2:
        ax3.axvline(x=np.sum(coef*X_std[p]+inter), color='red')
    else:
        ax3.axvline(x=np.sum(coef*X_std[p]+inter), color='blue')
    plt.legend(loc="lower right")

    ax1 = plt.subplot(grid[0,1])
    #fig.add_subplot(232)
    if gal_class==2:
        sns.kdeplot(nonmerg_m20_LDA, nonmerg_gini_LDA, cmap="Blues_r", shade=True,shade_lowest=False, alpha=0.25)
        sns.kdeplot(merg_m20_LDA, merg_gini_LDA, cmap="Reds_r", shade=True,shade_lowest=False, alpha=0.75)
    else:
        sns.kdeplot(merg_m20_LDA, merg_gini_LDA, cmap="Reds_r", shade=True,shade_lowest=False, alpha=0.25)
        sns.kdeplot(nonmerg_m20_LDA, nonmerg_gini_LDA, cmap="Blues_r", shade=True,shade_lowest=False, alpha=0.75)
    im1=ax1.scatter(df2['M20'].values[p], df2['Gini'].values[p], color='yellow',marker='*', s=25, zorder=100)
    dashed_line_x=np.linspace(-0.5,-3,100)
    dashed_line_y=[-0.14*x + 0.33 for x in dashed_line_x]

    ax1.plot(dashed_line_x, dashed_line_y, ls='--', color='black')

    ax1.set_xlim([0,-3])
    ax1.set_ylim([0.2,0.8])#ax1.set_ylim([0.3,0.8])
    ax1.set_xlabel(r'M$_{20}$')
    ax1.set_ylabel(r'Gini')
    ax1.set_aspect(abs(3)/abs(0.6))

    ax2 = plt.subplot(grid[0,2])
    #fig.add_subplot(233)
    ax2.set_xlim([-0.2,1])
    ax2.set_ylim([0,6])#ax1.set_ylim([0.3,0.8])
    ax2.set_xlabel(r'A')
    ax2.set_ylabel(r'C')
    ax2.set_aspect(1.2/6)
    plt.axvline(x=0.35, ls='--', color='black')
    if gal_class ==2:
        sns.kdeplot(nonmerg_A_LDA, nonmerg_C_LDA, cmap="Blues_r", shade=True,shade_lowest=False, alpha=0.25)
        sns.kdeplot(merg_A_LDA, merg_C_LDA, cmap="Reds_r", shade=True,shade_lowest=False, alpha=0.75)
    else:
        sns.kdeplot(merg_A_LDA, merg_C_LDA, cmap="Reds_r", shade=True,shade_lowest=False, alpha=0.25)
        sns.kdeplot(nonmerg_A_LDA, nonmerg_C_LDA, cmap="Blues_r", shade=True,shade_lowest=False, alpha=0.75)
    im2=ax2.scatter(df2['Asymmetry (A)'].values[p], df2['Concentration (C)'].values[p], color='yellow',marker='*', s=25, zorder=100)


    
    ax5=plt.subplot(grid[1,2])
    #fig.add_subplot(236)
    if gal_class ==2:
        sns.kdeplot(nonmerg_A_S_LDA, nonmerg_n_LDA, cmap="Blues_r", shade=True,shade_lowest=False, alpha=0.25)
        sns.kdeplot(merg_A_S_LDA, merg_n_LDA, cmap="Reds_r", shade=True,shade_lowest=False, alpha=0.75)
    else:
        sns.kdeplot(merg_A_S_LDA, merg_n_LDA, cmap="Reds_r", shade=True,shade_lowest=False, alpha=0.25)
        sns.kdeplot(nonmerg_A_S_LDA, nonmerg_n_LDA, cmap="Blues_r", shade=True,shade_lowest=False, alpha=0.75)
    im5=ax5.scatter(df2['Shape Asymmetry'].values[p], df2['Sersic N'].values[p], color='yellow',marker='*', s=25, zorder=100)


    ax5.set_xlim([0,1])
    ax5.set_ylim([0,4])#ax1.set_ylim([0.3,0.8])
    ax5.set_xlabel(r'$A_S$')
    ax5.set_ylabel(r'$n$')
    plt.axvline(x=0.2, ls='--', color='black')
    ax5.set_aspect(1/4)

    #plt.tight_layout()
    plt.savefig('panel_plots/Panel_'+str(gal_id)+'.png', dpi=500)
    plt.close()
    
    


plt.clf()
fig, ax = plt.subplots(4,4, figsize=(15, 15), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .001, wspace=.001)

ax = ax.ravel()
p_plot=0

for p in range(15):#len(df2)):
    
    gal_id=df2[['ID']].values[p][0]
    if gal_id=='3813':#8133-12704' or gal_id=='7992-12704' or gal_id=='8250-12701':
        continue
    else:
        #gal_class=list_sklearn[new_min_index].predict(X_std)[p]
        gal_class=classifications[p]
        if gal_class==2:
            gal_name='Merger'
        else:
            gal_name='Nonmerger'
        gal_prob=list_sklearn[new_min_index].predict_proba(X_std)[p]
    
   
    
    
      
        for j in range(len(drpall[1].data['PLATEIFU'])):
            if drpall[1].data['PLATEIFU'][j]==gal_id:
           
                redshift=drpall[1].data['NSA_Z'][j]
                mangaid=str(drpall[1].data['MANGAID'][j]  )
                designid=str(drpall[1].data['DESIGNID'][j])
    
        if int(designid) > 10000:
            dsgn_grp=(designid)[:3]
        else:
            dsgn_grp=(designid)[:2]
        try:
            im=fits.open('preim/preimage-'+mangaid+'.fits')
        except FileNotFoundError:
            os.chdir(os.path.expanduser('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/Identify_MaNGA_SDSS_mergers_imaging/preim'))
            '''Here is where it is necessary to know the SDSS data password and username'''
            os.system('wget --http-user=sdss --http-passwd=2.5-meters https://data.sdss.org/sas/mangawork/manga/preimaging/D00'+dsgn_grp+\
    'XX/'+designid+'/preimage-'+mangaid+'.fits.gz ')
            os.system('gunzip preimage-'+mangaid+'.fits.gz')
            os.chdir(os.path.expanduser('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/Identify_MaNGA_SDSS_mergers_imaging'))
        
    
        im=fits.open('preim/preimage-'+mangaid+'.fits')
    
        camera_data=(im[4].data/0.005) 
    
        im=ax[p_plot].imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(1), vmax=10**(4)), cmap='afmhot')
        ax[p_plot].annotate(str(gal_id)+'\n'+str(round(gal_prob[0],2))+' '+str(round(gal_prob[1],2))+'\n'+gal_name, xycoords='axes fraction',xy=(0.1,0.85),#ha="center", va="top",
                bbox=dict(facecolor='black',boxstyle="round", fc="1.0", edgecolor='white'))
        ax[p_plot].axis('off')
    
        os.chdir(os.path.expanduser('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/Identify_MaNGA_SDSS_mergers_imaging/preim'))
        os.system('rm preimage-'+mangaid+'.fits.gz')
        os.system('rm preimage-'+mangaid+'.fits')
        p_plot+=1

os.chdir(os.path.expanduser('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/Identify_MaNGA_SDSS_mergers_imaging'))
plt.savefig('MaNGA_rando_major.pdf')
    
