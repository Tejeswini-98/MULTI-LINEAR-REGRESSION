#!/usr/bin/env python
# coding: utf-8

# ### ======================== MUTLI LINEAR REGRESSION==================

# ### =================='50_Startups=================================

# In[1]:


import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np


# In[2]:


start=pd.read_csv('50_Startups.csv')
start


# In[3]:


sf=start.rename(columns={'R&D Spend':'rd','Administration':'ad','Marketing Spend':'ms'},inplace=False)
sf


# In[4]:


sd=sf.drop(['State'],axis=1)


# In[5]:


sd.info()


# In[6]:


sd.corr()


# In[7]:


sns.set_style(style='darkgrid')
sns.pairplot(sd)


# In[8]:


import statsmodels.formula.api as smf
model=smf.ols('Profit~rd+ad+ms',data=sd).fit()


# In[9]:


model.params


# In[10]:


print(model.tvalues,'\n',model.pvalues)


# In[11]:


(model.rsquared,model.rsquared_adj,model.aic)


# In[12]:


new_data=pd.DataFrame({'rd':165455,"ad":90000,"ms":300000},index=[1])


# In[13]:


model.predict(new_data)


# ### ========================TOYOTA CORALLA=============================

# In[14]:


import pandas as pd


# In[15]:


coralla=pd.read_csv("ToyotaCorolla.csv",encoding='unicode_escape')
print(coralla)


# In[16]:


ca=pd.DataFrame(data=coralla)
ca


# In[17]:


ca.shape


# In[18]:


ca.info()


# In[19]:


ca2=pd.concat([ca.iloc[:,2:4],ca.iloc[:,6:7],ca.iloc[:,8:9],ca.iloc[:,12:14],ca.iloc[:,15:18]],axis=1)
ca2


# In[20]:


ca3=ca2.rename({'Age_08_04':'Age','cc':"CC",'Quarterly_Tax':'QT'},axis=1)
ca3


# In[21]:


ca3[ca3.duplicated()]


# In[22]:


ca4=ca3.drop_duplicates().reset_index(drop=True)
ca4


# In[23]:


ca4.describe()


# In[24]:


ca4.corr()


# In[25]:


sns.set_style(style='darkgrid')
sns.pairplot(ca4)


# In[26]:


model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=ca4).fit()


# In[27]:


model.params


# In[28]:


model.tvalues,np.round (model.pvalues,5)


# In[29]:



model.rsquared,model.rsquared_adj


# In[30]:


slr_c=smf.ols('Price~CC',data=ca4).fit()
slr_c.tvalues,slr_c.pvalues


# In[31]:


slr_d=smf.ols('Price~Doors',data=ca4).fit()
slr_d.tvalues,slr_d.pvalues


# In[32]:


mlr_cd=smf.ols('Price~CC+Doors',data=ca4).fit()
mlr_cd.tvalues


# In[33]:


rsq_age=smf.ols("Age~KM+HP+CC+Doors+Gears+QT+Weight",data=ca4).fit().rsquared
vif_age=1/(1-rsq_age)
rsq_KM=smf.ols("KM~Age+HP+CC+Doors+Gears+QT+Weight",data=ca4).fit().rsquared
vif_KM=1/(1-rsq_KM)
rsq_HP=smf.ols("HP~Age+KM+CC+Doors+Gears+QT+Weight",data=ca4).fit().rsquared
vif_HP=1/(1-rsq_HP)
rsq_CC=smf.ols("CC~Age+HP+KM+Doors+Gears+QT+Weight",data=ca4).fit().rsquared
vif_CC=1/(1-rsq_CC)
rsq_Doors=smf.ols("Doors~Age+HP+CC+KM+Gears+QT+Weight",data=ca4).fit().rsquared
vif_Doors=1/(1-rsq_Doors)
rsq_Gears=smf.ols("Gears~Age+HP+CC+Doors+KM+QT+Weight",data=ca4).fit().rsquared
vif_Gears=1/(1-rsq_Gears)
rsq_QT=smf.ols("QT~Age+HP+CC+Doors+Gears+KM+Weight",data=ca4).fit().rsquared
vif_QT=1/(1-rsq_QT)
rsq_Weight=smf.ols("Weight~Age+HP+CC+Doors+Gears+QT+KM",data=ca4).fit().rsquared
vif_Weight=1/(1-rsq_Weight)


# In[34]:


d1={'Variable':["Age","KM","HP","CC","Doors",'Gears','QT','Weight'],
   'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_Doors,vif_Gears,vif_QT,vif_Weight]}
Vif_df=pd.DataFrame(d1)
Vif_df


# In[35]:


sm.qqplot(model.resid,line="q")
plot.title("normal Q-Q plot of residuals")
plot.show()


# In[36]:


list(np.where(model.resid>6000))


# In[37]:


list(np.where(model.resid<-6000))


# In[38]:


def standard_values(vals):return(vals-vals.mean())/vals.std()


# In[39]:


plot.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plot.title("Residual plot")
plot.xlabel("standardized fitted values")
plot.ylabel('standardized residual values ')
plot.show()


# In[40]:


fig=plot.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Age',fig=fig)
plot.show()


# In[41]:


fig=plot.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'KM',fig=fig)
plot.show()


# In[42]:


fig=plot.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'HP',fig=fig)
plot.show()


# In[43]:


fig=plot.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'CC',fig=fig)
plot.show()


# In[44]:


fig=plot.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Doors',fig=fig)
plot.show()


# In[45]:


fig=plot.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'QT',fig=fig)
plot.show()


# In[46]:


fig=plot.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Weight',fig=fig)
plot.show()


# In[47]:


(c,_)=model.get_influence().cooks_distance
c


# In[48]:


fig=plot.figure(figsize=(20,7))
plot.stem(np.arange(len(ca4)),np.round(c,3))
plot.xlabel('Row Index')
plot.ylabel('Cooks Distance')
plot.show()


# In[49]:


np.argmax(c),np.max(c)


# In[50]:


fig,ax=plot.subplots(figsize=(20,20))
fig=influence_plot(model,ax=ax)


# In[51]:


k=ca4.shape[1]
n=ca4.shape[0]
leverage_cutoff=(3*(k+1))/n
leverage_cutoff


# In[52]:


ca4[ca4.index.isin([80])]


# In[53]:


ca_new=ca4.copy()
ca_new


# In[54]:


ca5=ca_new.drop(ca_new.index[[80]],axis=0).reset_index(drop=True)
ca5


# In[55]:


while np.max(c)>0.5:
    model=smf.ols("Price~Age+KM+HP+CC+Doors+Gears+QT+Weight",data=ca5).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c),np.max(c)
    ca5=ca5.drop(ca5.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    ca5
else:
    final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=ca5).fit()
    final_model.rsquared,final_model.aic
    print("thus model accuracy is improvred to",final_model.rsquared)


# In[56]:


if np.max(c)>0.5:
    model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=ca5).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c),np.max(c)
    ca5=ca5.drop(ca5.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    ca5
elif np.max(c)<0.5:
    final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=ca5).fit()
    final_model.rsquared,final_model.aic
    print("thus model accuracy is improvred to",final_model.rsquared) 


# In[57]:


final_model.rsquared


# In[58]:


ca5


# In[59]:


new_data=pd.DataFrame({'Age':12,"KM":4000,"HP":80,"CC":1300,"Doors":4,"Gears":5,"QT":69,"Weight":1012},index=[0])
new_data
                       


# In[60]:


final_model.predict(new_data)


# In[61]:


pred_y=final_model.predict(ca5)
pred_y


# In[ ]:




