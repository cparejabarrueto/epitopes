import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import f_classif, chi2, f_regression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_curve, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

cols=[*range(1, 75, 1)] #72 es clase
df = pd.read_csv('descriptors_class_influenza.csv', sep=',',usecols=cols)
df.head()

label_encoder = preprocessing.LabelEncoder() #Encode target labels with value between 0 and n_classes-1. Para valores str
df['Class']= label_encoder.fit_transform(df['Class']) #Fit label encoder and return encoded labels.
print(df['Class'])
df

count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(2))
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations");

feature_cols =['PorcTiny','PorcSmall','PorcAliphatic','PorcAromatic','PorcNonPolar','PorcPolar','PorcCharged','PorcBasic','PorcAcidic','at_index','at_boman','at_charge','at_pi','at_hmoment_alpha','at_hmoment_sheet','HelixBendPreference','SideChainSize','ExtendedStructurePreference','Hidrophobicity','DoubleBendPreference','PartialSpecificVolume','FlatExtendedPreference','OccurrenceInAlphaRegion','pKC','SurroundingHidrophobicity','Blosum1','Blosum2','Blosum3','Blosum4','Blosum5','Blosum6','Blosum7','Blosum8','Blosum9','Blosum10','MsWhim1','MsWhim2','MsWhim3','st1','st2','st3','st4','st5','st6','st7','st8','t1','t2','t3','t4','t5','z1','z2','z3','z4','z5','HydrophobicityIndex','AlphaAndTurnPropensities','BulkyProperties','CompositionalCharacteristicIndex','LocalFlexibility','ElectronicProperties']
X=df[feature_cols]
Y=df.Class
Y=df.Class
X_p=preprocessing.normalize(X)

df2=df.iloc[np.random.random_integers(256,4312,256),0:74] #0:62
df3=df.iloc[0:256,0:74] #0:62
frames = [df3, df2]
df4=pd.concat(frames)
Xrandom=df4[feature_cols]
Yrandom=df4.Class
X_prandom=preprocessing.normalize(Xrandom)
count_classes = Yrandom.value_counts()
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(2))
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations");

classifiers = [SVC(random_state=123, probability=True),
               GaussianNB(var_smoothing=1e-9),
               RandomForestClassifier(random_state=123,n_estimators=1000),
               DecisionTreeClassifier(random_state=123,min_samples_split=100)]
result_table = pd.DataFrame(columns=['classifiers','fpr','tpr','auc'])

graph_acc = []; Acc = [];Rec = [];Pre = [];acc = [];rec = [];pre = [];acc_std = [];rec_std = [];pre_std = []
Std = [];Mean = [];Fpr = [];Tpr = [];Auc = [];Auc2 = [];Auc3 = [];Auc4 = [];fc = [];mean_tpr=[]
W_range = list(range(0, 100))
X_range = list(range(0, 400, 100))
Y_range = list(range(0, 100,1))

for cls in classifiers: #por cada clasificador
    print(cls)
    for i in W_range:
        fpr_all=[]
        tpr_all=[]
        X_train, X_test, y_train, y_test = train_test_split(X_prandom,Yrandom, test_size=0.2)
        model = cls.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(y_test,y_pred)
        fpr_all.append(fpr)
        tpr_all.append(tpr)
        auc = roc_auc_score(y_test, y_pred)
        Auc.append(auc)
        Acc.append(metrics.accuracy_score(y_test, y_pred.round()))
        Rec.append(metrics.recall_score(y_test,y_pred.round(),average='binary',pos_label = 1))
        Pre.append(metrics.precision_score(y_test,y_pred.round(),average='binary',pos_label = 1))
        result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr,
                                        'tpr':tpr,
                                        'auc':auc}, ignore_index=True)
    Auc2.append(np.mean(Auc))
    Auc3.append(np.std(Auc))
    acc.append(np.mean(Acc))
    rec.append(np.mean(Rec))
    pre.append(np.mean(Pre))
    acc_std.append(np.std(Acc))
    rec_std.append(np.std(Rec))
    pre_std.append(np.std(Pre))
    Auc=[]
    Acc=[]
    Rec=[]
    Pre=[]   

print(acc,acc_std)
print(rec,rec_std)
print(pre,pre_std)

fpr_svm=result_table['fpr'][0:99]
fpr_nb =result_table['fpr'][100:199]
fpr_rf =result_table['fpr'][200:299]
fpr_dt =result_table['fpr'][300:399]
tpr_svm=result_table['tpr'][0:99]
tpr_nb =result_table['tpr'][100:199]
tpr_rf =result_table['tpr'][200:299]
tpr_dt =result_table['tpr'][300:399]

fpr_svm=fpr_svm.to_list()
fpr_nb =fpr_nb.to_list()
fpr_rf =fpr_rf.to_list()
fpr_dt =fpr_dt.to_list()
tpr_svm=tpr_svm.to_list()
tpr_nb =tpr_nb.to_list()
tpr_rf =tpr_rf.to_list()
tpr_dt =tpr_dt.to_list()

matrix=np.zeros((2000,100))
for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_svm:             
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_fpr_svm = np.mean(matrix,axis=1)
std_fpr_svm = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_svm:            
    col=0
    for elemento2 in elemento:    
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1    
    fila=fila+1 
mean_tpr_svm = np.mean(matrix,axis=1)
std_tpr_svm = np.std(matrix,axis=1)


for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_nb:           
    col=0
    for elemento2 in elemento:   
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_fpr_nb = np.mean(matrix,axis=1)
std_fpr_nb = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_nb:            
    col=0
    for elemento2 in elemento:    
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1    
    fila=fila+1 
mean_tpr_nb = np.mean(matrix,axis=1)
std_tpr_nb = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_rf:            
    col=0
    for elemento2 in elemento: 
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_fpr_rf = np.mean(matrix,axis=1)
std_fpr_rf = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_rf:           
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_tpr_rf = np.mean(matrix,axis=1)
std_tpr_rf = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_dt:            
    col=0
    for elemento2 in elemento:   
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_fpr_dt = np.mean(matrix,axis=1)
std_fpr_dt = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_dt:            
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_tpr_dt = np.mean(matrix,axis=1)
std_tpr_dt = np.std(matrix,axis=1)

plt.figure(figsize=(8,6))
clasif = ["SVM","NB","RF","DT"]
plt.plot(mean_fpr_rf, mean_tpr_rf, color='green', label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[2], Auc2[2],Auc3[2]))
plt.plot(mean_fpr_nb, mean_tpr_nb, color='orange',label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[1], Auc2[1],Auc3[1]))
plt.plot(mean_fpr_dt, mean_tpr_dt, color='red',   label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[3], Auc2[3],Auc3[3]))
plt.plot(mean_fpr_svm,mean_tpr_svm,color='blue',  label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[0], Auc2[0],Auc3[0]))
plt.plot([0,1], [0,1], color='grey', linestyle='--')
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')
result_table.set_index('classifiers', inplace=True)
plt.savefig('ROC_curve_NegEpi_script_r256-256_alldescriptors.png')
plt.show()

V_range = list(range(1, 101))
Acc = []
Rec = []
Pre = []
RF=RandomForestClassifier(random_state=123)
for k in V_range:
    X_train, X_test, y_train, y_test = train_test_split(X_p,Y, test_size=0.20)#,random_state=123)
    RF.fit(X_train,y_train)
    y_pred = RF.predict(X_test)
    Acc.append(metrics.accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test,y_pred))
    Rec.append(metrics.recall_score(y_test,y_pred,average='binary',pos_label = 1))
    Pre.append(metrics.precision_score(y_test,y_pred,average='binary',pos_label = 1))
print('---------------------------------------------------')
print(" Comportamiento promedio de Accuracy: ", np.mean(Acc),"STD",np.std(Acc))
print(" Comportamiento promedio de Recall: ", np.mean(Rec),"STD",np.std(Rec))
print(" Comportamiento promedio de Precisión: ", np.mean(Pre),"STD",np.std(Pre))

V_range = list(range(1, 101))
Acc = []
Rec = []
Pre = []
RF=RandomForestClassifier(random_state=123)
for k in V_range:
    X_train, X_test, y_train, y_test = train_test_split(X_prandom,Yrandom, test_size=0.20)#,random_state=123)
    RF.fit(X_train,y_train)
    y_pred = RF.predict(X_test)
    Acc.append(metrics.accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test,y_pred))
    Rec.append(metrics.recall_score(y_test,y_pred,average='binary',pos_label = 1))
    Pre.append(metrics.precision_score(y_test,y_pred,average='binary',pos_label = 1))
print('---------------------------------------------------')
print(" Comportamiento promedio de Accuracy: ", np.mean(Acc),"STD",np.std(Acc))
print(" Comportamiento promedio de Recall: ", np.mean(Rec),"STD",np.std(Rec))
print(" Comportamiento promedio de Precisión: ", np.mean(Pre),"STD",np.std(Pre))

V_range = list(range(1, 101))
t_range = list(range(0,4))
print(t_range)
t_velues=[10,100,1000,10000]
t_scores=[0]
for j in t_range:
    Acc = []
    Rec = []
    Pre = []
    print(j)
    print(t_velues[j])
    for k in V_range:
        X_train, X_test, y_train, y_test = train_test_split(X_prandom,Yrandom, test_size=0.20)#,random_state=123)
        RF=RandomForestClassifier(n_estimators=t_velues[j])#,class_weight="balanced")
        RF.fit(X_train,y_train)
        y_pred = RF.predict(X_test)
        Acc.append(metrics.accuracy_score(y_test, y_pred))
        Rec.append(metrics.recall_score(y_test,y_pred,average='binary',pos_label = 1))
        Pre.append(metrics.precision_score(y_test,y_pred,average='binary',pos_label = 1))
    t_scores.append(np.mean(Rec))
    print('---------------------------------------------------')
    print(" Accuracy: ", np.mean(Acc),"STD",np.std(Acc))
    print(" Recall: ", np.mean(Rec),"STD",np.std(Rec))
    print(" Precisión: ", np.mean(Pre),"STD",np.std(Pre))
    print('---------------------------------------------------')

print(t_scores)
print(t_velues)
t_velues3=['0','10','100','1000','10000']
plt.plot(t_velues3,t_scores)
plt.xlabel('N')
plt.ylabel('Recall')


V_range = list(range(0, 62))
feature_cols =['PorcTiny','PorcSmall','PorcAliphatic','PorcAromatic','PorcNonPolar','PorcPolar','PorcCharged','PorcBasic','PorcAcidic','at_index','at_boman','at_charge','at_pi','at_hmoment_alpha','at_hmoment_sheet','HelixBendPreference','SideChainSize','ExtendedStructurePreference','Hidrophobicity','DoubleBendPreference','PartialSpecificVolume','FlatExtendedPreference','OccurrenceInAlphaRegion','pKC','SurroundingHidrophobicity','Blosum1','Blosum2','Blosum3','Blosum4','Blosum5','Blosum6','Blosum7','Blosum8','Blosum9','Blosum10','MsWhim1','MsWhim2','MsWhim3','st1','st2','st3','st4','st5','st6','st7','st8','t1','t2','t3','t4','t5','z1','z2','z3','z4','z5','HydrophobicityIndex','AlphaAndTurnPropensities','BulkyProperties','CompositionalCharacteristicIndex','LocalFlexibility','ElectronicProperties']
X=df[feature_cols]
X_p=preprocessing.normalize(X)
asel = SelectKBest(f_classif, k='all')
asel.fit(X_prandom,Yrandom)
scores = asel.scores_[asel.get_support()]
total_f_reg=-np.log10(asel.pvalues_)
total_f_reg /= total_f_reg.max()
names_scores = list(zip(feature_cols, total_f_reg))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_reg_Scores'])

ns_df_sorted = ns_df.sort_values(['F_reg_Scores', 'Feat_names'], ascending = [False, True])

X_indices = np.arange(X.shape[-1])
plt.figure(figsize=(15, 9))
print(list(ns_df_sorted['F_reg_Scores']))
plt.xticks(X_indices,ns_df_sorted['Feat_names'],rotation=90)
plt.bar(X_indices, ns_df_sorted['F_reg_Scores'], label="Regression($-Log(p_{value})$)",width=0.5)
plt.title("ANOVA - Comparing attributes",fontsize='xx-large')
plt.xlabel("Number of attributes",fontsize='xx-large')
plt.ylabel("Univariate score ($-Log(p_{value})$)",fontsize='xx-large')

plt.subplots_adjust(bottom=0.3)
plt.margins(0.005)
plt.ylim(0, 1)
plt.savefig('ml-ANOVA.png')
plt.show()

print(ns_df_sorted['F_reg_Scores'][41])
print(list(ns_df_sorted['Feat_names']))

print(*ns_df_sorted['Feat_names'][0:20], sep = ", ")
print(*ns_df_sorted['Feat_names'], sep = "', '")
print(*ns_df_sorted['F_reg_Scores'], sep = ", ")
print(ns_df_sorted['Feat_names'])

graph_acc=[]
anova = ['st7', 'Blosum4', 't2', 'PartialSpecificVolume', 'AlphaAndTurnPropensities', 'at_boman', 'HelixBendPreference', 't3', 'st2', 'Blosum3', 'DoubleBendPreference', 'MsWhim3', 'PorcCharged', 'Blosum5', 'PorcSmall', 'z3', 'st3', 'CompositionalCharacteristicIndex', 'SideChainSize', 'PorcAcidic', 'z2', 'Hidrophobicity', 'at_hmoment_sheet', 'ElectronicProperties', 'pKC', 't5', 'z5', 'PorcAromatic', 'PorcPolar', 'Blosum8', 'OccurrenceInAlphaRegion', 'PorcBasic', 'at_index', 'PorcAliphatic', 'FlatExtendedPreference', 'at_hmoment_alpha', 'MsWhim2', 'BulkyProperties', 'Blosum7', 'z1', 'LocalFlexibility', 'PorcTiny', 't1', 'z4', 'Blosum10', 'st6', 'Blosum1', 'st8', 'MsWhim1', 't4', 'SurroundingHidrophobicity', 'Blosum6', 'st5', 'at_pi', 'Blosum2', 'HydrophobicityIndex', 'at_charge', 'ExtendedStructurePreference', 'PorcNonPolar', 'Blosum9', 'st1', 'st4']
V_range = range(0, 62) #
print(V_range) #62 1,63*
Acc = []
Rec = []
Pre = []
Std = []
Mean = []
fc = []
print(anova[0])
W_range = range(0, 100)
for k in V_range: #atributos
    fc.append(anova[k])
    print(df[fc].head(0))
    for l in W_range: 
        X_r=df4[fc]
        Y_r=df4.Class
        Y_r.value_counts()
        X_rn=preprocessing.normalize(X_r)
        X_train, X_test, y_train, y_test = train_test_split(X_rn,Y_r, test_size=0.2)#,random_state=123)
        RF = RandomForestClassifier(random_state=123,n_estimators=1000)
        RF.fit(X_train, y_train)
        y_pred = RF.predict(X_test)
        Acc.append(metrics.accuracy_score(y_test, y_pred))
        Rec.append(metrics.recall_score(y_test,y_pred,average='binary',pos_label = 1))
        Pre.append(metrics.precision_score(y_test,y_pred,average='binary',pos_label = 1))
    Mean.append(np.mean(Acc))
    Std.append(np.std(Acc))
rango=[]
for i in V_range:
    rango.append(i+1)
import matplotlib.pyplot as plot
plt.bar(rango,Mean,yerr=Std,color="green",width=0.7)
plt.ylabel('Accuracy')
plt.xlabel('Accumulated attributes')
plt.title('Attributes selection')
plt.savefig('Exactitud_segun_atributos_ANOVA_NegEpi_script_r256-256.png')
plt.show()

print(list(Mean))
print(list(ns_df_sorted['Feat_names']))

#ANOVA
classifiers = [SVC(random_state=123, probability=True),
               GaussianNB(var_smoothing=1e-9),
               RandomForestClassifier(random_state=123),
               DecisionTreeClassifier(random_state=123,min_samples_split=100)]
result_table = pd.DataFrame(columns=['classifiers','fpr','tpr','auc'])
feature_coll = ['st7', 'Blosum4', 't2', 'PartialSpecificVolume', 'AlphaAndTurnPropensities', 'at_boman', 'HelixBendPreference', 't3', 'st2', 'Blosum3', 'DoubleBendPreference', 'MsWhim3', 'PorcCharged', 'Blosum5', 'PorcSmall', 'z3', 'st3', 'CompositionalCharacteristicIndex', 'SideChainSize', 'PorcAcidic', 'z2', 'Hidrophobicity', 'at_hmoment_sheet', 'ElectronicProperties', 'pKC', 't5', 'z5', 'PorcAromatic', 'PorcPolar', 'Blosum8', 'OccurrenceInAlphaRegion', 'PorcBasic', 'at_index', 'PorcAliphatic', 'FlatExtendedPreference', 'at_hmoment_alpha', 'MsWhim2', 'BulkyProperties', 'Blosum7', 'z1', 'LocalFlexibility', 'PorcTiny', 't1', 'z4', 'Blosum10', 'st6', 'Blosum1', 'st8', 'MsWhim1', 't4', 'SurroundingHidrophobicity', 'Blosum6', 'st5', 'at_pi', 'Blosum2', 'HydrophobicityIndex', 'at_charge', 'ExtendedStructurePreference', 'PorcNonPolar', 'Blosum9', 'st1', 'st4']

graph_acc = [];Acc = [];Rec = [];Pre = [];acc = [];rec = [];
pre = [];acc_std = [];rec_std = [];pre_std = [];Std = [];
Mean = [];Fpr = [];Tpr = [];Auc = [];Auc2 = [];Auc3 = [];Auc4 = [];fc = [];mean_tpr=[];

W_range = list(range(0, 100))
X_range = list(range(0, 400, 100))
Y_range = list(range(0, 100,1))
for cls in classifiers: #por cada clasificador
    print(cls)
    for i in W_range:
        fpr_all=[]
        tpr_all=[]
        X_r=df4[feature_coll]
        Y_r=df4.Class
        Y_r.value_counts()
        X_rn=preprocessing.normalize(X_r)
        X_train, X_test, y_train, y_test = train_test_split(X_rn,Y_r, test_size=0.20)
        model = cls.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(y_test,y_pred)
        fpr_all.append(fpr)
        tpr_all.append(tpr)
        auc = roc_auc_score(y_test, y_pred)
        Auc.append(auc)
        Acc.append(metrics.accuracy_score(y_test, y_pred.round()))
        Rec.append(metrics.recall_score(y_test,y_pred.round(),average='binary',pos_label = 1))
        Pre.append(metrics.precision_score(y_test,y_pred.round(),average='binary',pos_label = 1))
        result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr,
                                        'tpr':tpr,
                                        'auc':auc}, ignore_index=True)
    Auc2.append(np.mean(Auc))
    Auc3.append(np.std(Auc))
    acc.append(np.mean(Acc))
    rec.append(np.mean(Rec))
    pre.append(np.mean(Pre))
    acc_std.append(np.std(Acc))
    rec_std.append(np.std(Rec))
    pre_std.append(np.std(Pre))
    Auc=[]
    Acc=[]
    Rec=[]
    Pre=[]   
##IMPRESION DE TABLA
print(acc,acc_std)
print(rec,rec_std)
print(pre,pre_std)
fpr_svm=result_table['fpr'][0:99]
fpr_nb =result_table['fpr'][100:199]
fpr_rf =result_table['fpr'][200:299]
fpr_dt =result_table['fpr'][300:399]
tpr_svm=result_table['tpr'][0:99]
tpr_nb =result_table['tpr'][100:199]
tpr_rf =result_table['tpr'][200:299]
tpr_dt =result_table['tpr'][300:399]

fpr_svm=fpr_svm.to_list()
fpr_nb =fpr_nb.to_list()
fpr_rf =fpr_rf.to_list()
fpr_dt =fpr_dt.to_list()
tpr_svm=tpr_svm.to_list()
tpr_nb =tpr_nb.to_list()
tpr_rf =tpr_rf.to_list()
tpr_dt =tpr_dt.to_list()

matrix=np.zeros((2000,100))
for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_svm:            
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_fpr_svm = np.mean(matrix,axis=1)
std_fpr_svm = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_svm:            
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_tpr_svm = np.mean(matrix,axis=1)
std_tpr_svm = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_nb:            
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_fpr_nb = np.mean(matrix,axis=1)
std_fpr_nb = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_nb:            
    col=0
    for elemento2 in elemento:    
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1    
    fila=fila+1 
mean_tpr_nb = np.mean(matrix,axis=1)
std_tpr_nb = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_rf:            
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_fpr_rf = np.mean(matrix,axis=1)
std_fpr_rf = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_rf:            
    col=0
    for elemento2 in elemento:   
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1    
    fila=fila+1 
mean_tpr_rf = np.mean(matrix,axis=1)
std_tpr_rf = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_dt:           
    col=0
    for elemento2 in elemento:   
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1  
    fila=fila+1 
mean_fpr_dt = np.mean(matrix,axis=1)
std_fpr_dt = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_dt:            
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1  
    fila=fila+1 
mean_tpr_dt = np.mean(matrix,axis=1)
std_tpr_dt = np.std(matrix,axis=1)
plt.figure(figsize=(8,6))
clasif = ["SVM","NB","RF","DT"]
plt.plot(mean_fpr_rf, mean_tpr_rf, color='green', label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[2], Auc2[2],Auc3[2]))
plt.plot(mean_fpr_nb, mean_tpr_nb, color='orange',label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[1], Auc2[1],Auc3[1]))
plt.plot(mean_fpr_dt, mean_tpr_dt, color='red',   label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[3], Auc2[3],Auc3[3]))
plt.plot(mean_fpr_svm,mean_tpr_svm,color='blue',  label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[0], Auc2[0],Auc3[0]))
plt.plot([0,1], [0,1], color='grey', linestyle='--')
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')
result_table.set_index('classifiers', inplace=True)
plt.savefig('ROC_curve_ANOVA-NegEpi_script-r256-256.png')
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X_prandom,Yrandom,test_size=0.2,random_state=123)
mutual_info = mutual_info_classif(X_train, y_train,random_state=123)

mutual_info = pd.Series(mutual_info)
mutual_info.index = feature_cols
mi=mutual_info.sort_values(ascending=False)

mutual_info.sort_values(ascending=False).plot.bar(figsize=(15, 9))

sel_five_cols = SelectKBest(mutual_info_classif, k='all')
mi=sel_five_cols.fit(X_train, y_train)
mutual_info.index[sel_five_cols.get_support()]
plt.subplots_adjust(bottom=0.35)

plt.title("Mutual Information - Comparing attributes",fontsize='xx-large')
plt.xlabel("Number of attributes",fontsize='xx-large')
plt.ylabel("Mutual Information Score",fontsize='xx-large')
plt.savefig('ml-features-MutualInfo_random256-256_ok.png',pad_inches=1.5)

print(list(mutual_info.sort_values(ascending=False).index[sel_five_cols.get_support()]))
print(list(mutual_info.sort_values(ascending=False)))
print(list(mutual_info.sort_values(ascending=False).index[sel_five_cols.get_support()][0:20]))

graph_acc=[]
V_range = list(range(0, 62)) 
W_range = list(range(0, 10))
Acc = []
Rec = []
Pre = []
Std = []
Mean = []
fc = []
mutual=list(mutual_info.sort_values(ascending=False).index[sel_five_cols.get_support()])
for k in V_range: #veces de atributo
    fc.append(mutual[k])
    print(df[fc].head(0))
    for l in W_range:
        X_r=df4[fc]
        Y_r=df4.Class
        Y_r.value_counts()
        X_rn=preprocessing.normalize(X_r)
        X_train, X_test, y_train, y_test = train_test_split(X_rn,Y_r, test_size=0.20,random_state=123)
        RF=RandomForestClassifier(random_state=123,n_estimators=1000)
        RF.fit(X_train,y_train)
        y_pred = RF.predict(X_test)
        Acc.append(metrics.accuracy_score(y_test, y_pred))
        Rec.append(metrics.recall_score(y_test,y_pred,average='binary',pos_label = 1))
        Pre.append(metrics.precision_score(y_test,y_pred,average='binary',pos_label = 1))
    #print(np.mean(Acc))
    Mean.append(np.mean(Acc))
    Std.append(np.std(Acc))
rango=[]
#print(graph_acc)
for i in V_range:
    rango.append(i+1)
    
import matplotlib.pyplot as plot

plt.bar(rango,Mean,yerr=Std,color="green",width=0.7)
plt.ylabel('Accuracy')
plt.xlabel('Accumulated attributes')
plt.title('Attributes selection')
plt.savefig('Exactitud_segun_atributos_MutualInfo_RF_significativos-random256-256.png')
plt.show()

print("'PorcAromatic', 'PorcBasic', 'SurroundingHidrophobicity', 'at_pi', 'PorcAcidic', 'z1', 'z2', 'at_boman', 't2'")
print( mutual_info.sort_values(ascending=False).index[sel_five_cols.get_support()][0:9])
print(list(mutual_info.sort_values(ascending=False)))

classifiers = [SVC(random_state=123, probability=True),
               GaussianNB(var_smoothing=1e-9),
               RandomForestClassifier(random_state=123),#,n_estimators=1000),#,n_estimators=100,class_weight="balanced"),
               DecisionTreeClassifier(random_state=123,min_samples_split=100)]
# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers','fpr','tpr','auc'])
##MUTUALINFO

graph_acc=[]
Acc = [];Rec = [];Pre = [];acc = [];rec = [];pre = [];acc_std = [];rec_std = [];pre_std = [];
Std = [];Mean = [];Fpr = [];Tpr = [];Auc = [];Auc2 = [];Auc3 = [];Auc4 = [];fc = [];
mean_tpr=[]

W_range = list(range(0, 100))
X_range = list(range(0, 400, 100))
Y_range = list(range(0, 100,1))
mutual=['t3', 'PorcAromatic', 'PorcBasic', 'PartialSpecificVolume', 'Blosum1', 'z1', 'st7', 'at_hmoment_sheet', 'PorcAcidic', 'z5', 'AlphaAndTurnPropensities', 'st5', 'PorcAliphatic', 'PorcPolar', 't1', 't2', 'PorcTiny', 'at_boman', 'Blosum9', 'PorcSmall']

for cls in classifiers: #por cada clasificador
    print(cls)
    for i in W_range:   #repeticiones
        fpr_all=[]
        tpr_all=[]
        X_r=df4[mutual]
        Y_r=df4.Class
        Y_r.value_counts()
        X_rn=preprocessing.normalize(X_r)
        X_train, X_test, y_train, y_test = train_test_split(X_rn,Y_r, test_size=0.20)#,random_state=123)
        model = cls.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(y_test,y_pred)
        fpr_all.append(fpr)
        tpr_all.append(tpr)
        auc = roc_auc_score(y_test, y_pred)
        Auc.append(auc)
        Acc.append(metrics.accuracy_score(y_test, y_pred.round()))
        Rec.append(metrics.recall_score(y_test,y_pred.round(),average='binary',pos_label = 1))
        Pre.append(metrics.precision_score(y_test,y_pred.round(),average='binary',pos_label = 1))
        result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr,
                                        'tpr':tpr,
                                        'auc':auc}, ignore_index=True)
    Auc2.append(np.mean(Auc))
    Auc3.append(np.std(Auc))
    acc.append(np.mean(Acc))
    rec.append(np.mean(Rec))
    pre.append(np.mean(Pre))
    acc_std.append(np.std(Acc))
    rec_std.append(np.std(Rec))
    pre_std.append(np.std(Pre))
    Auc=[]
    Acc=[]
    Rec=[]
    Pre=[]   

print(acc,acc_std)
print(rec,rec_std)
print(pre,pre_std)

fpr_svm=result_table['fpr'][0:99]
fpr_nb =result_table['fpr'][100:199]
fpr_rf =result_table['fpr'][200:299]
fpr_dt =result_table['fpr'][300:399]
tpr_svm=result_table['tpr'][0:99]
tpr_nb =result_table['tpr'][100:199]
tpr_rf =result_table['tpr'][200:299]
tpr_dt =result_table['tpr'][300:399]

fpr_svm=fpr_svm.to_list()
fpr_nb =fpr_nb.to_list()
fpr_rf =fpr_rf.to_list()
fpr_dt =fpr_dt.to_list()
tpr_svm=tpr_svm.to_list()
tpr_nb =tpr_nb.to_list()
tpr_rf =tpr_rf.to_list()
tpr_dt =tpr_dt.to_list()

matrix=np.zeros((200,100))
for i in np.arange(0,200,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_svm:            
    col=0
    for elemento2 in elemento:  
        if col < 200: #columnas
            matrix[col][fila]=elemento2
        col=col+1   
    fila=fila+1 
mean_fpr_svm = np.mean(matrix,axis=1)
std_fpr_svm = np.std(matrix,axis=1)

for i in np.arange(0,200,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_svm:            
    col=0
    for elemento2 in elemento:  
        if col < 200: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_tpr_svm = np.mean(matrix,axis=1)
std_tpr_svm = np.std(matrix,axis=1)

for i in np.arange(0,200,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_nb:            
    col=0
    for elemento2 in elemento:  
        if col < 200: #columnas
            matrix[col][fila]=elemento2
        col=col+1 
    fila=fila+1 
mean_fpr_nb = np.mean(matrix,axis=1)
std_fpr_nb = np.std(matrix,axis=1)

for i in np.arange(0,200,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_nb:            
    col=0
    for elemento2 in elemento:    
        if col < 200: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_tpr_nb = np.mean(matrix,axis=1)
std_tpr_nb = np.std(matrix,axis=1)

for i in np.arange(0,200,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_rf:            
    col=0
    for elemento2 in elemento:  
        if col < 200: #columnas
            matrix[col][fila]=elemento2
        col=col+1 
    fila=fila+1 
mean_fpr_rf = np.mean(matrix,axis=1)
std_fpr_rf = np.std(matrix,axis=1)

for i in np.arange(0,200,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_rf:            
    col=0
    for elemento2 in elemento:
        if col < 200: #columnas
            matrix[col][fila]=elemento2
        col=col+1  
    fila=fila+1 
mean_tpr_rf = np.mean(matrix,axis=1)
std_tpr_rf = np.std(matrix,axis=1)

for i in np.arange(0,200,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_dt:            
    col=0
    for elemento2 in elemento: 
        if col < 200: #columnas
            matrix[col][fila]=elemento2
        col=col+1   
    fila=fila+1 
mean_fpr_dt = np.mean(matrix,axis=1)
std_fpr_dt = np.std(matrix,axis=1)

for i in np.arange(0,200,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_dt:            
    col=0
    for elemento2 in elemento:
        if col < 200: #columnas
            matrix[col][fila]=elemento2
        col=col+1   
    fila=fila+1 
mean_tpr_dt = np.mean(matrix,axis=1)
std_tpr_dt = np.std(matrix,axis=1)
###########
plt.figure(figsize=(8,6))
clasif = ["SVM","NB","RF","DT"]
plt.plot(mean_fpr_rf, mean_tpr_rf, color='green', label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[2], Auc2[2],Auc3[2]))
plt.plot(mean_fpr_nb, mean_tpr_nb, color='orange',label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[1], Auc2[1],Auc3[1]))
plt.plot(mean_fpr_dt, mean_tpr_dt, color='red',   label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[3], Auc2[3],Auc3[3]))
plt.plot(mean_fpr_svm,mean_tpr_svm,color='blue',  label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[0], Auc2[0],Auc3[0]))
plt.plot([0,1], [0,1], color='grey', linestyle='--')
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')
result_table.set_index('classifiers', inplace=True)
plt.savefig('ROC_curve_MUTUAL-INFO_NegEpi_script-r256-256.png')
plt.show() 

print(list(ns_df_sorted['Feat_names']))
print(list(ns_df_sorted['F_reg_Scores']))

#MIX

classifiers = [SVC(random_state=123, probability=True),
               GaussianNB(),
               RandomForestClassifier(random_state=123),
               DecisionTreeClassifier(random_state=123,min_samples_split=50)]

result_table = pd.DataFrame(columns=['classifiers','fpr','tpr','auc'])

graph_acc=[]
Acc = [];Rec = [];Pre = [];acc = [];rec = [];pre = [];acc_std = [];rec_std = [];pre_std = [];
Std = [];Mean = [];Fpr = [];Tpr = [];Auc = [];Auc2 = [];Auc3 = [];Auc4 = [];fc = [];
mean_tpr=[]

W_range = range(0, 100)
X_range = range(0, 400, 100)
Y_range = range(0, 100,1)

fc3=['st7', 'Blosum4', 't2', 'PartialSpecificVolume', 'AlphaAndTurnPropensities', 'at_boman', 'HelixBendPreference', 't3', 'st2', 'Blosum3', 'DoubleBendPreference', 'MsWhim3', 'PorcCharged', 'Blosum5', 'PorcSmall', 'z3', 'st3', 'CompositionalCharacteristicIndex', 'SideChainSize', 'PorcAcidic','t3', 'PorcAromatic', 'PorcBasic', 'PartialSpecificVolume', 'Blosum1', 'z1', 'st7', 'at_hmoment_sheet', 'PorcAcidic', 'z5', 'AlphaAndTurnPropensities', 'st5', 'PorcAliphatic', 'PorcPolar', 't1', 't2', 'PorcTiny', 'at_boman', 'Blosum9', 'PorcSmall']for cls in classifiers: 
    print(cls)
    for i in W_range:  
        fpr_all=[]
        tpr_all=[]
        X_r=df4[fc3] 
        Y_r=df4.Class
        Y_r.value_counts()
        X_rn=preprocessing.normalize(X_r)
        X_train, X_test, y_train, y_test = train_test_split(X_rn,Y_r, test_size=0.20)#,random_state=123)
        model = cls.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(y_test,y_pred)
        fpr_all.append(fpr)
        tpr_all.append(tpr)
        auc = roc_auc_score(y_test, y_pred)
        Auc.append(auc)
        Acc.append(metrics.accuracy_score(y_test, y_pred.round()))
        Rec.append(metrics.recall_score(y_test,y_pred.round(),average='binary',pos_label = 1))
        Pre.append(metrics.precision_score(y_test,y_pred.round(),average='binary',pos_label = 1))
        result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr,
                                        'tpr':tpr,
                                        'auc':auc}, ignore_index=True)
    Auc2.append(np.mean(Auc))
    Auc3.append(np.std(Auc))
    acc.append(np.mean(Acc))
    rec.append(np.mean(Rec))
    pre.append(np.mean(Pre))
    acc_std.append(np.std(Acc))
    rec_std.append(np.std(Rec))
    pre_std.append(np.std(Pre))
    Auc=[]
    Acc=[]
    Rec=[]
    Pre=[]   

print(acc,acc_std)
print(rec,rec_std)
print(pre,pre_std)

fpr_svm=result_table['fpr'][0:99]
fpr_nb =result_table['fpr'][100:199]
fpr_rf =result_table['fpr'][200:299]
fpr_dt =result_table['fpr'][300:399]
tpr_svm=result_table['tpr'][0:99]
tpr_nb =result_table['tpr'][100:199]
tpr_rf =result_table['tpr'][200:299]
tpr_dt =result_table['tpr'][300:399]

fpr_svm=fpr_svm.to_list()
fpr_nb =fpr_nb.to_list()
fpr_rf =fpr_rf.to_list()
fpr_dt =fpr_dt.to_list()
tpr_svm=tpr_svm.to_list()
tpr_nb =tpr_nb.to_list()
tpr_rf =tpr_rf.to_list()
tpr_dt =tpr_dt.to_list()

matrix=np.zeros((2000,100))
for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_svm:            
    col=0
    for elemento2 in elemento:
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1    
    fila=fila+1 
mean_fpr_svm = np.mean(matrix,axis=1)
std_fpr_svm = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_svm:            
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1   
    fila=fila+1 
mean_tpr_svm = np.mean(matrix,axis=1)
std_tpr_svm = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_nb:            
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1    
    fila=fila+1 
mean_fpr_nb = np.mean(matrix,axis=1)
std_fpr_nb = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_nb:            
    col=0
    for elemento2 in elemento:    
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1    
    fila=fila+1 
mean_tpr_nb = np.mean(matrix,axis=1)
std_tpr_nb = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_rf:            
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1   
    fila=fila+1 
mean_fpr_rf = np.mean(matrix,axis=1)
std_fpr_rf = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_rf:            
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1    
    fila=fila+1 
mean_tpr_rf = np.mean(matrix,axis=1)
std_tpr_rf = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in fpr_dt:            
    col=0
    for elemento2 in elemento:  
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1    
    fila=fila+1 
mean_fpr_dt = np.mean(matrix,axis=1)
std_fpr_dt = np.std(matrix,axis=1)

for i in np.arange(0,2000,1):
    for j in np.arange(0,100,1):
        matrix[i][j]=1
        j=j+1
    i=i+1    
fila=0
for elemento in tpr_dt:            
    col=0
    for elemento2 in elemento:    
        if col < 2000: #columnas
            matrix[col][fila]=elemento2
        col=col+1     
    fila=fila+1 
mean_tpr_dt = np.mean(matrix,axis=1)
std_tpr_dt = np.std(matrix,axis=1)
###########
plt.figure(figsize=(8,6))
clasif = ["SVM","NB","RF","DT"]
plt.plot(mean_fpr_rf, mean_tpr_rf, color='green', label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[2], Auc2[2],Auc3[2]))
plt.plot(mean_fpr_nb, mean_tpr_nb, color='orange',label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[1], Auc2[1],Auc3[1]))
plt.plot(mean_fpr_dt, mean_tpr_dt, color='red',   label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[3], Auc2[3],Auc3[3]))
plt.plot(mean_fpr_svm,mean_tpr_svm,color='blue',  label="Mean ROC {}, AUC={:.2f}±{:.2f}".format(clasif[0], Auc2[0],Auc3[0]))
plt.plot([0,1], [0,1], color='grey', linestyle='--')
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')
result_table.set_index('classifiers', inplace=True)
plt.savefig('ROC_curve_MIX_NegEpi_script.png')
plt.show() 