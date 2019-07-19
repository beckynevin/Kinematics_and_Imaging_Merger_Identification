'''
~~~
MergerMonger Expert classifies a provided .txt file of galaxies with predictors
using the posterior probability distributions from the input simulated galaxies
used to construct the LDA. 
It does a very similar thing to MergerMonger Novice but produces prior probabilities instead
of just a binary classification.
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

def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a)
                  if smallest == element]

os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/MergerMonger'))

   # from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
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
priors_list=[[0.9,0.1]]#[0.75,0.25]]
plt.clf()
missclass_list=[]


df = pd.io.parsers.read_table(filepath_or_buffer='LDA_prep_predictors_all_combined_major.txt',header=[0],sep='\t')
#was LDA_prep_predictors_all_combined.txt
#LDA_img_ratio_statmorph_fg3_m12_A_S.txt'
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
    
    
    
    
df.dropna(inplace=True) # to drop the empty line at file-end
    
    

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
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
#print(df)

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

kf = StratifiedKFold(n_splits=10, random_state=True, shuffle=True)

for o in range(len(inputs)):#len(inputs)-20):

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
        inputs_here=[]
        inputs_here.append(inputs[k])

        #print('starting input', prev_input,inputs[k])
        #if inputs[k] is a cross term and the prev_input doesn't contain it, add it:
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

        from sklearn import preprocessing



        std_scale = preprocessing.StandardScaler().fit(X)
        
        
        std_scale_this_step.append(std_scale)
        X = std_scale.transform(X)


        enc = LabelEncoder()
        label_encoder = enc.fit(y)
        y = label_encoder.transform(y) + 1


        label_dict = {1: 'NonMerger', 2: 'Merger'}
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

        # LDA
        sklearn_lda = LDA(priors=priors_list[0], store_covariance=True)#store_covariance=False

        sklearn_this_step.append(sklearn_lda)
        X_lda_sklearn = sklearn_lda.fit_transform(X, y)
        
        dec = sklearn_lda.score(X,y)
        prob = sklearn_lda.predict_proba(X)

        coef = sklearn_lda.coef_







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
        for train_index, test_index in kf.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


            sklearn_lda = LDA( priors=priors_list[0],store_covariance=True)#store_covariance=False
            #sklearn_list.append(sklearn_lda)


            X_lda_sklearn = sklearn_lda.fit_transform(X_train, y_train)
            coef = sklearn_lda.coef_
            inter = sklearn_lda.intercept_

            inter_list.append(inter)
            coef_list.append(coef)
            inter_list.append(inter)
            pred =sklearn_lda.predict(X_test)
            classes_list.append(sklearn_lda.classes_)
            confusion_master.append(confusion_matrix(pred,y_test))

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
            

        accuracy.append(np.mean(single_prediction))#/(master[0][0]+master[1][0]+master[0][1]+master[1][1]))

        accuracy_e.append(np.std(single_prediction))
        inputs_this_step.append(np.array(prev_input))

        confusion_master_this_step.append(np.array((np.mean(confusion_master,axis=0)/np.sum(np.mean(confusion_master,axis=0))).transpose()))
        master_this_step.append(np.array(np.mean(confusion_master, axis=0).transpose()))
        #print('appending with this', np.array(prev_input))

        classes_this_step.append(np.array(classes_list))
        #sklearn_this_step.append(sklearn_list)


        coef_mean.append(np.mean(coef_list, axis=0))
        coef_std.append(np.std(coef_list, axis=0))

        inter_mean.append(np.mean(inter_list, axis=0))
        inter_std.append(np.std(inter_list, axis=0))
        mean_non_this_step.append(np.mean(mean_non_list))

        #prev_input.remove(new_stuff)
        for m in range(len(inputs_here)):
            try:
                prev_input.remove(inputs_here[m])
            except ValueError:
                continue

    if accuracy_e[accuracy.index(min(accuracy))]<0.00001:
        break
    #print('all of inputs', inputs_this_step)
    #print('selecting the best model for this step', (inputs_this_step[accuracy.index(min(accuracy))]))

    thing=(inputs_this_step[accuracy.index(min(accuracy))])
    first_A=min(accuracy)

    prev_input_here.append(thing)

    for m in range(len(thing)):

        prev_input.append(thing[m])

        try:

            inputs.remove(thing[m])
        except ValueError:
            #print('~~~RUning into troubles')
            #print('inputs', inputs)
            #print('the thing to remove', thing[m])
            continue
    #print('the input now', inputs)
    #STOP
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

#print('these are your inputs',prev_input, prev_input_here)


#list_coef=[]
#list_coef_std=[]

min_A=min(missclass)
min_comps=num_comps[missclass.index(min(missclass))]
for p in range(len(missclass)):
    missclass_list.append(missclass[p])
plt.plot(num_comps,missclass, color='black')
plt.scatter(num_comps,missclass, color='black')

plt.fill_between(num_comps,np.array(missclass)+np.array(missclass_e), np.array(missclass)-np.array(missclass_e), alpha=.5,color='orange')

def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a) 
                  if smallest == element]

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



'''plt.clf()
fig=plt.figure()#figsize=(6,6)
plot_confusion_matrix(list_master_confusion[new_min_index], sklearn_lda.classes_, title='Normalized Confusion Matrix')
plt.savefig('Confusion_matrix.pdf')'''
#This is from this website: http://www.science.smith.edu/~jcrouser/SDS293/labs/lab5-py.html
plt.clf()
sns.set_style("darkgrid")

master=list_master[new_min_index]

print('~~~Accuracy~~~')
print((master[1][1]+master[0][0])/(master[0][0]+master[1][0]+master[0][1]+master[1][1]))
print('~~~Precision~~~')
print(master[1][1]/(master[0][1]+master[1][1]))#TP/(TP+FP)
print('~~~Recall~~~')
print(master[1][1]/(master[1][0]+master[1][1]))#TP/(TP+FN)
print('~~~F1~~~')
print((2*master[1][1])/(master[0][1]+master[1][0]+2*master[1][1]))#2TP/(2TP+FP+FN)







'''This is to create a file for use in the Novice version of the code'''
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

'''Now make a beautiful plot by first making contours from the sims'''
'''But also reclassify all of df to compare'''
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

for j in range(len(df)):
    if df['class label'].values[j]==0:
        nonmerg_gini.append(df['Gini'].values[j])
        nonmerg_m20.append(df['M20'].values[j])
    if df['class label'].values[j]==1:
        merg_gini.append(df['Gini'].values[j])
        merg_m20.append(df['M20'].values[j])
    
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

'''plt.clf()
fig=plt.figure()
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)
ax1=fig.add_subplot(121)
ax1.set_title('Merging')
ax1.plot(dashed_line_x, dashed_line_y, ls='--', color='black')
im1=ax1.scatter(merg_m20, merg_gini, color='red')
im2=ax1.scatter(merg_m20_LDA, merg_gini_LDA, color='pink')
ax1.set_xlim([0,-3])
ax1.set_ylim([0.2,0.8])#ax1.set_ylim([0.3,0.8])
ax1.set_xlabel(r'M$_{20}$')
ax1.set_ylabel(r'Gini')
ax1.set_aspect(abs(3)/abs(0.6))

ax2=fig.add_subplot(122)
ax2.set_title('Nonmerging')
ax2.plot(dashed_line_x, dashed_line_y, ls='--', color='black')
im1=ax2.scatter(nonmerg_m20, nonmerg_gini, color='blue')
im2=ax2.scatter(nonmerg_m20_LDA, nonmerg_gini_LDA, color='purple')
ax2.set_xlim([0,-3])
ax2.set_ylim([0.2,0.8])#ax1.set_ylim([0.3,0.8])
ax2.set_xlabel(r'M$_{20}$')
ax2.set_ylabel(r'Gini')
ax2.set_aspect(abs(3)/abs(0.6))

plt.savefig('gini_m20_lda_vs_OG_class.pdf')'''

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


STOP





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

df2 = pd.io.parsers.read_table(filepath_or_buffer='LDA_img_statmorph_SDSS_50_mergers.txt',header=[0],sep='\t')
#LDA_img_statmorph_Fu_mergers.txt
#LDA_img_statmorph_rando_MaNGA.txt
df2.columns = [l for i,l in sorted(feature_dict2.items())] + ['Shape Asymmetry']

print(df2)


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




#X_gal = std_scale.transform(X_gal)

print('np.shape', np.shape(list_std_scale[new_min_index].mean_))
print('np.shape', np.shape(list_std_scale[new_min_index].var_))
print('np.shape', np.shape(X_gal))



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

print(list_sklearn[new_min_index].predict(X_std))
print(list_sklearn[new_min_index].predict_proba(X_std))
classifications=list_sklearn[new_min_index].predict(X_std)
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
print('~~~~Mergers~~~~')
LDA_value=[]

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
        LDA_value.append(np.sum(X_std[j]))
        print(df2['ID'].values[j],df2['Gini'].values[j], df2['M20'].values[j],
              df2['Concentration (C)'].values[j],df2['Asymmetry (A)'].values[j],
              df2['Clumpiness (S)'].values[j],df2['Sersic N'].values[j],
              df2['Shape Asymmetry'].values[j])
        print(X_std[j],np.sum(X_std[j]))
        print('~~~~~~')
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
        LDA_value.append(np.sum(X_std[j]))
    
plt.clf()
plt.hist(LDA_value)
plt.savefig('hist_LDA.pdf')

 

print('percent nonmerg',len(nonmerg_name_list)/(len(nonmerg_name_list)+len(merg_name_list)))
print('percent merg',len(merg_name_list)/(len(nonmerg_name_list)+len(merg_name_list)))


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








print('finished')

'''Optional panel to plot the images of these things with their probabilities assigned'''
from astropy.io import fits
import os
import seaborn as sns
import matplotlib.colors as colors

sns.set_style("white")
os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/MergerMonger'))
drpall=fits.open('drpall-v2_4_3.fits')
print(len(df2),int(np.sqrt(len(df2))))

'''First, an option for just plotting these individually'''

for p in range(len(df2)):#len(df2)):
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
    plt.savefig('SDSS_superclean_ellip_major_'+str(gal_id)+'.pdf')
    

STOP
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
        gal_class=list_sklearn[new_min_index].predict(X_std)[p]
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
            os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/MergerMonger/preim'))
            '''Here is where it is necessary to know the SDSS data password and username'''
            os.system('wget --http-user=sdss --http-passwd=2.5-meters https://data.sdss.org/sas/mangawork/manga/preimaging/D00'+dsgn_grp+\
    'XX/'+designid+'/preimage-'+mangaid+'.fits.gz ')
            os.system('gunzip preimage-'+mangaid+'.fits.gz')
            os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/MergerMonger'))
        
    
        im=fits.open('preim/preimage-'+mangaid+'.fits')
    
        camera_data=(im[4].data/0.005) 
    
        im=ax[p_plot].imshow(np.abs(camera_data),norm=colors.LogNorm(vmin=10**(1), vmax=10**(4)), cmap='afmhot')
        ax[p_plot].annotate(str(gal_id)+'\n'+str(gal_prob)+'\n'+gal_name, xycoords='axes fraction',xy=(0.1,0.9),#ha="center", va="top",
                bbox=dict(boxstyle="round", fc="1.0"))
        ax[p_plot].axis('off')
    
        os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/MergerMonger/preim'))
        os.system('rm preimage-'+mangaid+'.fits.gz')
        os.system('rm preimage-'+mangaid+'.fits')
        p_plot+=1

os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/MergerMonger'))
plt.savefig('MaNGA_rando_major.pdf')
    
