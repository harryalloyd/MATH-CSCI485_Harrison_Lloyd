import pandas as pd  
from sklearn.metrics import r2_score  
from sklearn.feature_selection import RFE 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
import matplotlib.pyplot as plt  
from sklearn.datasets import load_diabetes  
 
diabetes_bundle =load_diabetes()
feature_frame =pd.DataFrame(diabetes_bundle.data,columns=diabetes_bundle.feature_names)  # features as dataframe
target_series =pd.Series(diabetes_bundle.target,name="target")  #target as series

# 80/20 split
train_features,test_features,train_target,test_target = train_test_split(
    feature_frame,target_series,test_size=0.2) 

model =LinearRegression()  
model.fit(train_features,train_target) 

baseline_pred =model.predict(test_features)  #predict on test data
baseline_r2 =r2_score(test_target,baseline_pred)  
print("task 2")
print(f"R^2 score: {baseline_r2:.4f}\n")  

feature_labels =list(feature_frame.columns)  

rfe_r2_scores,rfe_selected_sets,rfe_coef_maps ={},{},{}

for i in range(feature_frame.shape[1],0,-1):  # iterate backwards
    rfe_selector =RFE(estimator=LinearRegression(),n_features_to_select=i)  
    rfe_selector.fit(train_features,train_target)  #fit selector on training data

    selected_features =feature_frame.columns[rfe_selector.support_].tolist()  #selected features
    rfe_selected_sets[i] =selected_features 
    subset_model =LinearRegression()  
    subset_model.fit(train_features[selected_features],train_target)  #fit subset model

    subset_pred =subset_model.predict(test_features[selected_features])  #predict with subset model
    subset_r2 =r2_score(test_target,subset_pred)  #r2 for this k
    rfe_r2_scores[i] =subset_r2  

    rfe_coef_maps[i] =dict(zip(selected_features,subset_model.coef_))  #store coefficients

sorted_ks =sorted(rfe_r2_scores.keys(),reverse=True)  
k_column =[]  
r2_column =[]  
features_column =[]  

for i in sorted_ks:  
    k_column.append(i)  
    r2_column.append(rfe_r2_scores[i])  
    features_column.append(rfe_selected_sets[i])  

r2_table_frame =pd.DataFrame()  #create table
r2_table_frame["k"] =k_column  # fill k, r^2, and features
r2_table_frame["R^2"] =r2_column  
r2_table_frame["features"] =features_column
r2_table_frame.to_csv("rfe_r2_by_k.csv",index=False)

r2_threshold =0.01  # the r^2 improvement
feature_counts =sorted(rfe_r2_scores.keys())  
r2_improvements ={}  

for i in range(1,len(feature_counts)):  # calculate marginal r^2 gain for each added feature
    curr_k =feature_counts[i]  
    prev_k =feature_counts[i-1]  
    r2_improvements[curr_k] =rfe_r2_scores[curr_k]-rfe_r2_scores[prev_k]  

significant_ks =[]  #k values where r^2 improvement meets threshold
for i in r2_improvements:
    delta_value =r2_improvements[i]
    if delta_value >=r2_threshold:
        significant_ks.append(i)  

if len(significant_ks)>0:  
    optimal_k =max(significant_ks)  
else:  
    optimal_k =max(rfe_r2_scores,key=rfe_r2_scores.get)

print("Task 3.5")  
print(f"threshold={r2_threshold}")  
print(f"chosen k={optimal_k} with test R^2={rfe_r2_scores[optimal_k]:.4f}")  
print(f"the {optimal_k} features at k: {rfe_selected_sets[optimal_k]}") 

plot_ks =sorted(rfe_r2_scores.keys())  
plot_r2s =[]  
for k in plot_ks:
    plot_r2s.append(rfe_r2_scores[k])

plt.figure()  #create the visualization
plt.plot(plot_ks,plot_r2s)  
plt.xlabel("Number of k features")  
plt.ylabel("Test R^2")  
plt.tight_layout()  
plt.savefig("rfe_r2_vs_k.png")  
plt.close()

coef_table_frame =pd.DataFrame(0.0,index=sorted(rfe_coef_maps.keys(),reverse=True),columns=feature_labels)  # build table

for i in rfe_coef_maps:  #coefficient table with the k values
    for feat_name,coef_value in rfe_coef_maps[i].items():  
        coef_table_frame.loc[i,feat_name]=coef_value  

coef_table_frame.to_csv("rfe_coefficients_by_k.csv",index_label="k")  
