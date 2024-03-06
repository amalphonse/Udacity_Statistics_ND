#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# Corresponding with this notebook is a slide deck where you will need to update all the portions in red.  Completing the notebook will provide all the results needed for the slides.  **Correctly completing the slides is a required part of the project.**
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Descriptive Statistics](#descriptive)
# - [Part II - Probability](#probability)
# - [Part III - Experimentation](#experimentation)
# - [Part IV - Algorithms](#algorithms)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='descriptive'></a>
# #### Part I - Descriptive Statistics
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(0)


# For each of the parts of question `1` notice links to [pandas documentation](https://pandas.pydata.org/) is provided to assist with answering the questions.  Though there are other ways you could solve the questions, the documentation is provided to assist you with one fast way to find the answer to each question.
# 
# 
# `1.a)` Now, read in the `ab_data.csv` data. Store it in `df`. Read in the dataset and take a look at the top few rows here. **This question is completed for you**:

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head()


# `b)` Use the below cell to find the number of rows in the dataset. [Helpful  Pandas Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html#pandas.DataFrame.shape)

# In[4]:


df.shape


# In[4]:


df[df['group']=='control'].count()[0]


# In[5]:


df[df['group']=='treatment'].count()[0]


# `c)` The proportion of users converted.  [Helpful  Pandas Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html)

# In[12]:


df[df['converted']==1].count()[0]


# In[14]:


df['converted'].mean()


# `d)` Do any of the rows have missing values? [Helpful Pandas Link One](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html) and [Helpful Pandas Link Two](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html)

# In[7]:


df.isnull().sum()


# `e)` How many customers are from each country? Build a bar chart to show the count of visits from each country.

# In[15]:


# number of visitors from each country - pull the necessary code from the next cell to provide just the counts
df['country'].value_counts()


# In[9]:


# bar chart of results - this part is done for you
df['country'].value_counts().plot(kind='bar');
plt.title('Number of Visits From Each Country');
plt.ylabel('Count of Visits');
plt.show();


# `f)` Recognize that all of your columns are of a **categorical data type** with the exception of one.  Which column is not **categorical**? [Helpful Pandas Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html)

# In[4]:


df.info()


# `g)` What are the possible values of the `converted` column?  Does it make sense that these values are the only possible values? Why or why not? 
# 
# **Here you can use one of the functions you used in an earlier question**.

# In[5]:


df['converted'].value_counts() #yes only 0 or 1 because either either they are converted to new page or not


# <a id='probability'></a>
# #### Part II - Probability
# 
# `1.` Now that you have had a chance to learn more about the dataset, let's look more at how different factors are related to `converting`.
# 
# `a)` What is the probability of an individual converting regardless of the page they receive or the country they are from? Simply, what is the chance of conversion in the dataset?

# In[11]:


df.query('converted==1').shape[0]/df.shape[0]


# `b)` Given that an individual was in the `control` group, what is the probability they converted? **This question is completed for you**

# In[16]:


df.query('group == "control"')['converted'].mean()


# `c)` Given that an individual was in the `treatment` group, what is the probability they converted?

# In[17]:


df.query('group == "treatment"')['converted'].mean()


# `d)` Do you see evidence that the treatment is related to higher `converted` rates?

# yes

# `e)` What is the probability that an individual was in the `treatment`?

# In[8]:


df.query('group == "treatment"').shape[0]/df.shape[0]


# `f)` What is the probability that an individual was from Canada `CA`?

# 1/3

# `g)` Given that an individual was in the `US`, what was the probability that they `converted`? **This question is completed for you**
# 
# $P(\text{converted} == 1|\text{country} ==\text{"US"})$
# 
# 

# In[18]:


df.query('country == "US"')['converted'].mean()


# `h)` Given that an individual was in the `UK`, what was the probability that they `converted`? 
# 
# $P(\text{converted} == 1|\text{country} ==\text{"UK"})$

# In[19]:


df.query('country == "UK"')['converted'].mean()


# `i)` Do you see evidence that the `converted` rate might differ from one country to the next?

# yes

# `j)` Consider the table below, fill in the conversion rates below to look at how conversion by country and treatment group vary.  The `US` column is done for you, and two methods for calculating the probabilities are shown - **COMPLETE THE REST OF THE TABLE**.  Does it appear that there could be an interaction between how country and treatment impact conversion?
# 
# These two values that are filled in can be written as:
# 
# $P(\text{converted} == 1|(\text{country} ==\text{"US" AND }\text{group} ==\text{"control"})) = 10.7\%$
# 
# $P(\text{converted} == 1|(\text{country} ==\text{"US" AND }\text{group} ==\text{"treatment"})) = 15.8\%$
# 
# |             | US          | UK          | CA          |
# | ----------- | ----------- | ----------- | ----------- |
# | Control     | 10.7%       |  %          |  %          |
# | Treatment   | 15.8%       |  %          |  %          |

# In[ ]:


# Method 1  - explicitly calculate each probability
print(df.query('country == "US" and group == "control" and converted == 1').shape[0]/df.query('country == "US" and group == "control"').shape[0]) 
print(df.query('country == "US" and group == "treatment" and converted == 1').shape[0]/df.query('country == "US" and group == "treatment"').shape[0])


# In[3]:


# Method 2 - quickly calculate using `groupby`
df.query('country == "US"').groupby('group')['converted'].mean()


# In[4]:


df.query('country == "UK"').groupby('group')['converted'].mean()


# In[5]:


df.query('country == "CA"').groupby('group')['converted'].mean()


# ##### Solution -- Complete the Table Here
# 
# |             | US          | UK          | CA          |
# | ----------- | ----------- | ----------- | ----------- |
# | Control     | 10.7%       |  %          |  %          |
# | Treatment   | 15.8%       |  %          |  %          |

# In[6]:


print(df.query('country == "US" and group == "control" and converted == 1').shape[0]/df.query('country == "US" and group == "control"').shape[0]) 
print(df.query('country == "US" and group == "treatment" and converted == 1').shape[0]/df.query('country == "US" and group == "treatment"').shape[0])
print(df.query('country == "UK" and group == "control" and converted == 1').shape[0]/df.query('country == "UK" and group == "control"').shape[0]) 
print(df.query('country == "UK" and group == "treatment" and converted == 1').shape[0]/df.query('country == "UK" and group == "treatment"').shape[0])
print(df.query('country == "CA" and group == "control" and converted == 1').shape[0]/df.query('country == "CA" and group == "control"').shape[0]) 
print(df.query('country == "CA" and group == "treatment" and converted == 1').shape[0]/df.query('country == "CA" and group == "treatment"').shape[0])


# <a id='experimentation'></a>
# ### Part III - Experimentation
# 
# `1.` Consider you need to make the decision just based on all the data provided.  If you want to assume that the control page is better unless the treatment page proves to be definitely better at a Type I error rate of 5%, you state your null and alternative hypotheses in terms of **$p_{control}$** and **$p_{treatment}$** as:  
# 
# $H_{0}: p_{control} >= p_{treatment}$
# 
# $H_{1}: p_{control} < p_{treatment}$
# 
# Which is equivalent to:
# 
# $H_{0}: p_{treatment} - p_{control} <= 0$
# 
# $H_{1}: p_{treatment} - p_{control} > 0$
# 
# 
# Where  
# * **$p_{control}$** is the `converted` rate for the control page
# * **$p_{treatment}$** `converted` rate for the treatment page
# 
# **Note for this experiment we are not looking at differences associated with country.**

# Assume under the null hypothesis, $p_{treatment}$ and $p_{control}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{treatment}$ and $p_{control}$ are equal. Furthermore, assume they are equal to the **converted** rate in `df` regardless of the page. **These are set in the first cell below.**<br><br>
# 
# * Use a sample size for each page equal to the ones in `df`. **These are also set below.**  <br><br>
# 
# * Perform the sampling distribution for the difference in `converted` between the two pages over 500 iterations of calculating an estimate from the null.  <br><br>
# 
# * Use the cells below to provide the necessary parts of this simulation.  
# 
# If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 4** in the classroom to make sure you are on the right track.<br><br>

# `a)` The **convert rate** for $p_{treatment}$ under the null.  The **convert rate** for $p_{control}$ under the null. The sample size for the `control` and the sample size for the `treatment` are from the original dataset. **All of these values are set below, and set the stage for the simulations you will run for the rest of this section.**

# In[6]:


p_control_treatment_null  = df['converted'].mean()
n_treatment = df.query('group == "treatment"').shape[0]
n_control = df.query('group == "control"').shape[0]


# In[4]:


n_treatment


# `b)` Use the results from part `a)` to simulate `n_treatment` transactions with a convert rate of `p_treatment_null`.  Store these $n_{treatment}$ 1's and 0's in a `list` of **treatment_converted**.  It should look something like the following (the 0's and and 1's **don't** need to be the same): 
# 
# `[0, 0, 1, 1, 0, ....]` 

# In[6]:


df.query('group == "treatment" and converted == 1')['converted']


# In[8]:


treatment_converted = df.query('group == "treatment" and converted == 1')['converted']


# In[13]:


treatment_converted.shape[0]


# In[14]:


treatment_converted.shape[0]/n_treatment


# `c)` Use the results from part `a)` to simulate `n_control` transactions with a convert rate of `p_control_null`.  Store these $n_{treatment}$ 1's and 0's in a `list` of **control_converted**.  It should look something like the following (the 0's and and 1's **don't** need to be exactly the same): 
# 
# `[0, 0, 1, 1, 0, ....]` 

# In[16]:


control_converted = df.query('group == "control" and converted == 1')['converted']


# In[17]:


control_converted.count()


# In[18]:


control_converted.shape[0]/n_control


# `d)` Find the estimate for $p_{treatment}$ - $p_{control}$ under the null using the simulated values from part `(b)` and `(c)`.

# In[10]:


treatment_converted.count()-control_converted.count()


# In[19]:


treatment_conversion=treatment_converted.shape[0]/n_treatment
control_conversion = control_converted.shape[0]/n_control
delta = treatment_conversion-control_conversion
delta


# `e)` Simulate 500 $p_{treatment}$ - $p_{control}$ values using this same process as `b)`- `d)` similarly to the one you calculated in parts **a. through g.** above.  Store all 500 values in an numpy array called **p_diffs**.  This array should look similar to the below **(the values will not match AND this will likely take a bit of time to run)**:
# 
# `[0.001, -0.003, 0.002, ...]`

# In[ ]:


#bootsamp = sample_data.sample(200, replace = True)
#coff_mean = bootsamp[bootsamp['drinks_coffee'] == True]['height'].mean()
#nocoff_mean = bootsamp[bootsamp['drinks_coffee'] == False]['height'].mean()
#diffs.append(coff_mean - nocoff_mean)


# In[20]:


p_diffs = []
for _ in range(500):
    # simulate the treatment and control converted arrays
    df_sample = df.sample(500, replace = True)
    treatment_converted_mean = df_sample.query('group == "treatment" and converted == 1')['converted'].mean()
    control_converted_mean = df_sample.query('group == "control" and converted == 1')['converted'].mean()
    # calculate p_treatment and p_control under the null
    # calculate the difference between p_treatment_null and p_control_null
    # add p_diff to the p_diffs array
    p_diffs.append(treatment_converted_mean-control_converted_mean)
p_diffs


# In[ ]:





# `f)` Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[12]:


p_diffs = pd.Series(p_diffs)
p_diffs.hist(bins=20)


# `g)` What proportion of the **p_diffs** are greater than the difference observed between `treatment` and `control` in `df`?

# the difference is 0

# `h)` In words, explain what you just computed in part `g)`  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages using our Type I error rate of 0.05?

# **Your Answer Here** 

# <a id='algorithms'></a>
# ### Part IV - Algorithms
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.  All the code needed for the modeling and results of the modeling for sections `b) - f)` have been completed for you. 
# 
# **You will need to complete sections `a)` and `g)`.**  
# 
# **Then use the code from `1.` to assist with the question `2.`   You should be able to modify the code to assist in answering each of question 2's parts.**<br><br>
# 
# `a)` Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Your Answer Here** Logistic Regression.

# The goal is to use **statsmodels** to fit the regression model you specified in part `a)` to see if there is a significant difference in conversion based on which page a customer receives.  
# 
# `b)` However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.
# 
# It may be helpful to look at the [get_dummies documentation](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) to encode the `ab_page` column.
# 
# Below you can see an example of the new columns that will need to be added (The order of columns is not important.): **This question is completed for you**
# 
# ##### Example DataFrame
# | intercept   | group       | ab_page     | converted   |
# | ----------- | ----------- | ----------- | ----------- |
# | 1           |  control    |  0          |  0          |
# | 1           |  treatment  |  1          |  0          |
# | 1           |  treatment  |  1          |  0          |
# | 1           |  control    |  0          |  0          |
# | 1           |  treatment  |  1          |  1          |
# | 1           |  treatment  |  1          |  1          |
# | 1           |  treatment  |  1          |  0          |
# | 1           |  control    |  0          |  1          |

# In[13]:


df['intercept'] = 1
df['ab_page'] = pd.get_dummies(df['group'])['treatment']
df.head()


# `c)`  Create your `X` matrix and `y` response column that will be passed to your model, where you are testing if there is a difference in `treatment` vs. `control`. **This question is completed for you**

# In[4]:


X = df[['intercept', 'ab_page']]
y = df['converted']


# `d)` Use **statsmodels** to import and fit your regression model on the `X` and `y` from part `c)`. 
# 
# You can find the [statsmodels documentation to assist with this exercise here](https://www.statsmodels.org/stable/discretemod.html).  **This question is completed for you**

# In[5]:


import statsmodels.api as sm

# Logit Model
logit_mod = sm.Logit(y, X)
logit_res = logit_mod.fit()


# `e)` Provide the summary of your model below. **This question is completed for you**

# In[6]:


print(logit_res.summary())


# `f)` What is the p-value associated with **ab_page**? Does it lead you to the same conclusion you drew in the **Experiment** section.

# **Your Answer Here.** p value is 0. Yes it does lead to the same conclusion.

# `2. a)` Now you will want to create two new columns as dummy variables for `US` and `UK`.  Again, use `get_dummies` to add these columns.  The dataframe you create should include at least the following columns (If both columns for `US` and `UK` are `0` this represents `CA`.  The order of rows and columns is not important for you to match - it is just to illustrate how columns should connect to one another.):
# 
# ##### Example DataFrame
# | intercept   | group       | ab_page     | converted   | country     |  US         | UK          |
# | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
# | 1           |  control    |  0          |  0          |  US         |  1          |  0          |
# | 1           |  treatment  |  1          |  0          |  UK         |  0          |  1          |
# | 1           |  treatment  |  1          |  0          |  US         |  1          |  0          |
# | 1           |  control    |  0          |  0          |  US         |  1          |  0          |
# | 1           |  treatment  |  1          |  1          |  CA         |  0          |  0          |
# | 1           |  treatment  |  1          |  1          |  UK         |  0          |  1          |
# | 1           |  treatment  |  1          |  0          |  US         |  1          |  0          |
# | 1           |  control    |  0          |  1          |  US         |  1          |  0          |

# In[8]:





# In[12]:


### Create the necessary dummy variables
df['intercept'] = 1
df['US'] = pd.get_dummies(df['country'])['US']
df['UK'] = pd.get_dummies(df['country'])['UK']
df.head()


# In[14]:


df.head()


# `b)`  Create your `X` matrix and `y` response column that will be passed to your model, where you are testing if there is 
# * a difference in `converted` between `treatment` vs. `control`
# * a difference in `converted` between `US`, `UK`, and `CA`

# In[15]:


X = df[['intercept', 'ab_page','US','UK']]
y = df['converted']


# `c)` Use **statsmodels** to import and fit your regression model on the `X` and `y` from part `b)`. 
# You can find the [statsmodels documentation to assist with this exercise here](https://www.statsmodels.org/stable/discretemod.html).

# In[16]:


import statsmodels.api as sm

# Logit Model
logit_mod = sm.Logit(y, X)
logit_res = logit_mod.fit()


# `d)` Provide the summary of your model below.

# In[17]:


logit_res.summary()


# `e)` What do the `p-values` associated with `US` and `UK` suggest in relation to how they impact `converted`? 

# **Your Answer Here** That in UK the conversion rate is staistically higher than in US.

# <a id='finalcheck'></a>
# ## Final Check!
# 
# Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# <a id='submission'></a>
# ## Submission
# 
# Please follow the directions in the classroom to submit this notebook, as well as your completed slides.

# In[ ]:




