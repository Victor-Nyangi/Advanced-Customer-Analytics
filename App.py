import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle#module that turns python objects into strings
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

sns.set()#Override the default matplotlib look with the seaborn one

from matplotlib.backends.backend_agg import RendererAgg
import streamlit as st

from sklearn.decomposition import PCA

scaler = pickle.load(open('scaler.pickle', 'rb'))
pca_model = pickle.load(open('pca.pickle', 'rb'))
kmeans_pca = pickle.load(open('kmeans_pca.pickle', 'rb'))

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

_lock = RendererAgg.lock


st.set_page_config(layout="wide")

st.title('''CUSTOMER ANALYTICS''')

st.markdown("""
The goal of this project is to use various Machine Learning techniques to get insight on a set of Customer data, understand customer behaviour and apply this information through marketing strategies.

The project is divided into 3 main steps Segmentation, Targeting and Positioning which make up the STP marketing model.
The STP marketing model is a familiar strategic approach in modern marketing. It is one of the most commonly applied marketing models in practice, with marketing leaders crediting it for efficient, streamlined communications practice.
It can ve applied to all areas of business and marketing activities.

- **Segmentation:** the process of dividing a population of customers into groups that share similar characteristics. Observations within the same group would have comparable and would respond similarly to different marketing activities.

- **Targeting:** the process of evaluating potential profits from each segment and deciding which segments to focus on for example, you can target one segment on TV and another online.

- **Positioning:** what product characteristics do the customers from a certain segment need? Shows how a product should be  presented to the customers and through what channel. This process has a framework of it's own known as **Marketing Mix**

Marketing Mix entails developing the best product or service and offering it at the right price through the right channel. 
This can be done by finding the purchase probability, brand choice probability and purchase quantity of a product.
This can be split into 4Ps:-
Product (Product features; Branding; Packaging), Price (Product cost, long term price changes), Promotion (Price reduction, display and feature), Place (Distribution: intensive, selective, exclusive)
""")
st.write('')

df_segmentation = pd.read_csv('segmentation_data.csv', index_col=0)

st.write('''
The dataset consists of information about the purchasing behavior of 2,000 individuals from a given area when entering a physical ‘FMCG’ store. All data has been collected through the loyalty cards they use at checkout. The data has been preprocessed and there are no missing values. 
In addition, the volume of the dataset has been restricted and anonymised to protect the privacy of the customers.

The variables are:-
- ID numerical Integer that shows a unique identificator of a customer.
- Sex that is categorical {0,1}	Biological sex (gender) of a customer. 0 for male, 1 for female.
- Marital status categorical{0,1} 0 for single and 1 for non-single (divorced / separated / married / widocolor = ('b', 'g', 'r', 'orange'))d)	
- Age of the customer in years
- Education	categorical	{0,1,2,3}, Level of education of the customer 0 for other / unknown, 1 for high school, 2 for university, 3 for graduate school	
- Income Self-reported annual income in US dollars of the customer.	
- Occupation categorical {0,1,2} Category or occupation of the customer, 0 for unemployed / unskilled, 1 for skilled employee / official, 2 for management / self-employed / highly qualified employee / officer	
- Settlement size categorical {0,1,2}, the size of the city that the customer lives in.	0 for small city, 1 for mid-sized city, 2 for big city
''')
st.write('''## Analyzing customer data''')
st.write('Looking at the data to gain some insight.')

st.write(df_segmentation.sample(n=7))

st.write('Descriptive statistics')
st.write(df_segmentation.describe())

st.write('')
row1_1, row1_2 = st.beta_columns((1,1))
with row1_1, _lock:
    fig = plt.figure(figsize=(12,10))
    s = sns.heatmap(df_segmentation.corr(),
                annot = True,#retain correlation coefficients
                cmap ='RdBu',
                vmin = - 1,
                vmax = 1)
    s.set_yticklabels(s.get_yticklabels(), rotation = 0, fontsize = 12)
    s.set_xticklabels(s.get_xticklabels(), rotation = 90, fontsize = 12)
    plt.title("Correlation heatmap")
    st.pyplot(fig)
with row1_2, _lock:
    st.write('')

with row1_2, _lock:
    st.write('''
    Correlation is a statistical measure that expresses the extent to which two variables are linearly related and shows possible linear association between two continuous variables.
    The blue regions show positive correlation, in other words when one variable increases the other variable increases and the red regions show negtive correlation, in other words when one variable decreases while the other decreases.

    The deeper the colour the stronger the positive or negative correlation.
    ''')

st.write('## **Segmentation**')
st.write('''
To achieve this, the step will involve:
- Standardizing data, so that all features have equal weight and avoid situations where Income would be considered much more important than Education for Instance. 
- Principal Components Analysis to reduce the dimensionality of the data by finding a subset of components while reserving variance/Information
- Hierarchical and flat clustering for dividing customers into groups. 
''')
segmentation_std = scaler.transform(df_segmentation)

pca = PCA(n_components = 3)
pca.fit(segmentation_std)
df_pca_comp = pd.DataFrame(data = pca.components_, columns= df_segmentation.columns.values, index = ['Component 1', 'Component 2', 'Component 3'])

st.write('I standardized the data using a standard scaler then transformed the data using a PCA model and set it to return 3 components ')
row2_1, row2_2 = st.beta_columns((1,1))
with row2_1, _lock:    
    fig = plt.figure(figsize=(12,9))
    sns.heatmap(df_pca_comp,
           vmin = -1,
           vmax = 1,
           cmap = 'RdBu',
           annot= True)
    plt.yticks([0, 1, 2],['Component 1', 'Component 2', 'Component 3'], rotation = 45, fontsize = 12)
    st.pyplot(fig)

with row2_2, _lock:
    st.markdown('''
    The 3 components represent the 7 variables being used and the heatmap to the right shows the the correlations between an original variable and a component.
    
    - Component 1 has a positive correlation on Age, Education, Income, Occupation, Settlement Size, this shows the **career focus** of an individual
    - Component 2 has a strong positive correlation towards Sex, Marital Status, and Education and a negative correlation to the career focused determinants, as such this shows the **education and life style** of an individual. 
    - Component 3 Age, Marital Status and Occupation are the most important determinants so this can be phrased as an individual's **work or life experience**
    ''')

scores_pca = pca_model.transform(segmentation_std)

df_segm_pca_kmeans = pd.concat([df_segmentation.reset_index(drop=True), pd.DataFrame(scores_pca)], axis = 1)
df_segm_pca_kmeans.columns.values[-3: ] = ['Component 1', 'Component 2', 'Component 3']
df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
df_segm_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment K-means PCA']).mean()
st.write('Using K-Means clustering model to group the data into 4 clusters')

df_segm_pca_kmeans_freq

df_segm_pca_kmeans_freq['Observations'] = df_segm_pca_kmeans[['Segment K-means PCA', 'Sex']].groupby(['Segment K-means PCA']).count()
df_segm_pca_kmeans_freq['Proportion of observations'] = df_segm_pca_kmeans_freq['Observations'] / df_segm_pca_kmeans_freq['Observations'].sum()
df_segm_pca_kmeans_freq = df_segm_pca_kmeans_freq.rename({0:'standard', 1:'career focused', 2:'fewer opportunities', 3:'well-off'})

st.write('### Qualitative Interpretation')
st.write('''
I determined that component 1 leans more towards **career** of an individual, component 2 leans more towards **Experience** and component 3 leans more towards the **experience** of an individual.
- Segment 0 is low on career, high on education and lifestyle and on experience which shows that this is the youngest segment and I label them as **standard**.
- Segment 1 is high on career but low on education and lifestyle and seems independent from experience so I label them as **career-focued**.
- Segment 2 is lowest career, education and life style but high value in experience so they have **fewer-opportunities**.
- Segment 3 has high value on career, experience and education and life style so they are **well-off**.
''')
df_segm_pca_kmeans_freq[['Component 1', 'Component 2', 'Component 3','Proportion of observations','Observations']]

df_segm_pca_kmeans['Legend'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0:'standard', 1:'career focused', 2:'fewer opportunities', 3:'well-off'})

row3_1, row3_2, row3_3 = st.beta_columns((1,1,1))
with row3_1, _lock:
    fig = plt.figure(figsize = (12,12))
    sns.scatterplot(data = df_segm_pca_kmeans, x= 'Component 1', y='Component 2', hue = df_segm_pca_kmeans['Legend'], palette = ['g','r','c','m'])
    plt.title('Clusters by PCA Components 1 and 2')
    st.pyplot(fig)

with row3_2, _lock:
    fig = plt.figure(figsize = (12,12))
    sns.scatterplot(data = df_segm_pca_kmeans, x= 'Component 1', y='Component 3', hue = df_segm_pca_kmeans['Legend'], palette = ['g','r','c','m'])
    plt.title('Clusters by PCA Components 1 and 3')
    st.pyplot(fig)

with row3_3, _lock:
    fig = plt.figure(figsize = (12,12))
    sns.scatterplot(data = df_segm_pca_kmeans, x= 'Component 2', y='Component 3', hue = df_segm_pca_kmeans['Legend'], palette = ['g','r','c','m'])
    plt.title('Clusters by PCA Components 2 and 3')
    st.pyplot(fig)

st.write('Now the data is segmented into various clusters')

st.write('## **Positioning**')
st.write('''
#### Marketing mix...
1. Purchase probability
2. Brand choice probability
3. Purchase quantity
''')
st.write('''
### Purchase Analytics
''')

st.write('''
This dataset consists of information about the purchases of chocolate candy bars of 500 individuals from a given area when entering a physical ‘FMCG’ store in the period of 2 years. All data has been collected through the loyalty cards they use at checkout. The data has been preprocessed and there are no missing values. In addition, the volume of the dataset has been restricted and anonymised to protect the privacy of the customers. 
The variables are:-
- ID numerical Integer that shows a unique identificator of a customer.
- Day when the customer has visited the store
- Incidence: 0 the customer has not purchased an item from the category of interest, 1	the customer has purchased an item from the category of interest 
- Brand categorical	{0,1,2,3,4,5} shows which brand the customer has purchased, 0 No brand was purchased 1,2,3,4,5 for the brand ID
- Quantity number of items bought by the customer from the product category of interest
- Last_Inc_Brand categorical {0,1,2,3,4,5} shows which brand the customer has purchased on their previous store visit 0	No brand was purchased 1,2,3,4,5 Brand ID
- Last_Inc_Quantity	number of items bought by the customer from the product category of interest during their previous store visit
- Price_1/2/3/4/5 numerical real Price of an item from Brand 1/2/3/4/5 on a particular day
- Promotion_1/2/3/4/5 categorical {0,1} Indicator whether Brand 1/2/3/4/5 was on promotion or not on a particular day, 0 there is no promotion, 1 There is promotion
- Sex that is categorical {0,1}	Biological sex (gender) of a customer. 0 for male, 1 for female.
- Marital status categorical{0,1} 0 for single and 1 for non-single (divorced / separated / married / widowed)	
- Age of the customer in years
- Education	categorical	{0,1,2,3}, Level of education of the customer 0 for other / unknown, 1 for high school, 2 for university, 3 for graduate school	
- Income Self-reported annual income in US dollars of the customer.	
- Occupation categorical {0,1,2} Category or occupation of the customer, 0 for unemployed / unskilled, 1 for skilled employee / official, 2 for management / self-employed / highly qualified employee / officer	
- Settlement size categorical {0,1,2}, the size of the city that the customer lives in.	0 for small city, 1 for mid-sized city, 2 for big city
''')

df_purchase = pd.read_csv('purchase data.csv')

st.write('''## Analyzing customer data''')
st.write('Looking at the data to gain some insight.')

st.write(df_purchase.sample(n=7))

st.write('''
I scale and transform using PCA and Kmeans saved models.
''')

features = df_purchase[['Sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement size']]
df_purchase_segm_std = scaler.transform(features)

df_purchase_segm_pca = pca_model.transform(df_purchase_segm_std)
purchase_segm_kmeans_pca = kmeans_pca.predict(df_purchase_segm_pca)

df_purchase_predictors = df_purchase.copy()

df_purchase_predictors['Segment'] = purchase_segm_kmeans_pca

st.write(f"There are {df_purchase.shape[0]} total visits by 500 customers")

temp1 = df_purchase_predictors[['ID', 'Incidence']].groupby(['ID'], as_index = False).count()
temp1 = temp1.set_index('ID')
temp1 = temp1.rename(columns = {'Incidence': 'N_visits'})
temp2 = df_purchase_predictors[['ID', 'Incidence']].groupby(['ID'], as_index = False).sum()
temp2 = temp2.set_index('ID')
temp2 = temp2.rename(columns = {'Incidence':'N_Purchases'})
temp3 = temp1.join(temp2)
temp3['Average_N_Purchases'] = temp3['N_Purchases']/temp3['N_visits']
temp4 = df_purchase_predictors[['ID', 'Segment']].groupby(['ID'], as_index = False).mean()
temp4 = temp4.set_index('ID')
df_purchase_descr = temp3.join(temp4)

segm_prop = df_purchase_descr[['N_Purchases', 'Segment']].groupby(['Segment']).count() / df_purchase_descr.shape[0]
segm_prop = segm_prop.rename(columns = {'N_Purchases': 'Segment Proportions'})
segments_mean = df_purchase_descr.groupby(['Segment']).mean()
segments_std = df_purchase_descr.groupby(['Segment']).std()

row4_1, row4_2 = st.beta_columns((1,1))
with row4_1, _lock:
    fig = plt.figure(figsize = (5,4))
    plt.pie(segm_prop['Segment Proportions'],
        labels = ['Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'],
        autopct = '%1.1f%%',
        colors = ('b', 'g', 'r', 'orange'))
    plt.title('Segment Proportions')
    st.pyplot(fig)

with row4_2, _lock:
    fig = plt.figure(figsize = (9,6))
    plt.bar(x = (0, 1, 2, 3),
        tick_label = ('Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'),
        height = segments_mean['N_visits'],
        yerr = segments_std['N_visits'],#y error
        color = ('b', 'g', 'r', 'orange'))
    plt.xlabel('Segment')
    plt.ylabel('Number of Store Visits')
    plt.title('Average Number of Store Visits by Segment')
    st.pyplot(fig)

row5_1, row5_2 = st.beta_columns((1,1))
with row5_1, _lock:
    fig = plt.figure(figsize = (9,6))
    plt.bar(x = (0, 1, 2, 3),
        tick_label = ('Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'),
        height = segments_mean['N_Purchases'],
        yerr = segments_std['N_Purchases'],#y error is sd
        color = ('b', 'g', 'r', 'orange'))
    plt.xlabel('Segment')
    plt.ylabel('Purchase Incidences')
    plt.title('Purchases Incidences Visits by Segment')
    st.pyplot(fig)

with row5_2, _lock:
    fig = plt.figure(figsize = (9,6))
    plt.bar(x = (0, 1, 2, 3),
        tick_label = ('Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'),
        height = segments_mean['Average_N_Purchases'],
        yerr = segments_std['Average_N_Purchases'],
        color = ('b', 'g', 'r', 'orange'))
    plt.xlabel('Segment')
    plt.ylabel('Average Number of Purchases')
    plt.title('Average number of Purchases by Segment')
    st.pyplot(fig)

st.write('''

The vertical line is the dispersion of data points/sd
Career focused high dispersion shows that the some groups have higher number of purchases than others
Basically despite having close to the same income, they spend it differently
Fewer-Opportunities is homogenous coz of less s.d.

''')

df_purchase_incidence = df_purchase_predictors[df_purchase_predictors['Incidence'] == 1]

brand_dummies = pd.get_dummies(df_purchase_incidence['Brand'], prefix = 'Brand', prefix_sep = '_')
brand_dummies['Segment'], brand_dummies['ID'] = df_purchase_incidence['Segment'], df_purchase_incidence['ID']

temp = brand_dummies.groupby(['ID'], as_index = True).mean()

mean_brand_choice = temp.groupby(['Segment'], as_index = True).mean()


row6_1, row6_2 = st.beta_columns((2,1))
with row6_1, _lock:
    st.subheader('**Average Brand Choice by Segment**')
    fig=plt.figure()
    sns.heatmap(mean_brand_choice,
            vmin = 0,
            vmax = 1,
            cmap = 'PuBu',
            annot = True)
    plt.yticks([0, 1, 2, 3], ['Standard', 'Career-Focused', 'Fewer-Opportunities', 'Well-Off'], rotation=45, fontsize=9)
    st.pyplot(fig)

with row6_2, _lock:
    st.write('')
    st.write('')
    st.write('''
    - Brands are in order of cost
    - Fewer opportunities prefer Brand_2, so price is not that of a big factor
    - Carreer focused prefer Brand 5, the most expensive maybe due to to luxury, so raise price of Brand 5
    - Well-Off buy brand 4 then brand 5
    - Standard(most homogenous) prefer brand 2 then 1 thus I could try and influence them to try different brands
    ''')


temp = df_purchase_predictors[df_purchase_predictors['Brand'] == 1]
temp.loc[:, 'Revenue Brand 1'] = temp['Price_1'] * temp['Quantity']
segments_brand_revenue = pd.DataFrame()
segments_brand_revenue[['Segment', 'Revenue Brand 1']] = temp[['Segment','Revenue Brand 1']].groupby(['Segment'], as_index = False).sum()

temp = df_purchase_predictors[df_purchase_predictors['Brand'] == 2]
temp.loc[:, 'Revenue Brand 2'] = temp['Price_2'] * temp['Quantity']
segments_brand_revenue[['Segment', 'Revenue Brand 2']] = temp[['Segment','Revenue Brand 2']].groupby(['Segment'], as_index = False).sum()

temp = df_purchase_predictors[df_purchase_predictors['Brand'] == 3]
temp.loc[:, 'Revenue Brand 3'] = temp['Price_3'] * temp['Quantity']
segments_brand_revenue[['Segment', 'Revenue Brand 3']] = temp[['Segment','Revenue Brand 3']].groupby(['Segment'], as_index = False).sum()


temp = df_purchase_predictors[df_purchase_predictors['Brand'] == 4]
temp.loc[:, 'Revenue Brand 4'] = temp['Price_4'] * temp['Quantity']
segments_brand_revenue[['Segment', 'Revenue Brand 4']] = temp[['Segment','Revenue Brand 4']].groupby(['Segment'], as_index = False).sum()

temp = df_purchase_predictors[df_purchase_predictors['Brand'] == 5]
temp.loc[:, 'Revenue Brand 5'] = temp['Price_5'] * temp['Quantity']
segments_brand_revenue[['Segment', 'Revenue Brand 5']] = temp[['Segment','Revenue Brand 5']].groupby(['Segment'], as_index = False).sum()

segments_brand_revenue['Total Revenue'] = (segments_brand_revenue['Revenue Brand 1'] +
                                           segments_brand_revenue['Revenue Brand 2'] +
                                           segments_brand_revenue['Revenue Brand 3'] +
                                           segments_brand_revenue['Revenue Brand 4'] +
                                           segments_brand_revenue['Revenue Brand 5'])

segments_brand_revenue['Segment Proportions'] = segm_prop['Segment Proportions']
segments_brand_revenue['Segment'] = segments_brand_revenue['Segment'].map({0:'Standard',
                                                                           1:'Career-Focused',
                                                                           2:'Fewer-Opportunities',
                                                                           3:'Well-Off'})
segments_brand_revenue = segments_brand_revenue.set_index(['Segment'])

st.write('Revenue per brand per cluster')
segments_brand_revenue
st.write('''
One can  sees that if brand 3 was to reduce its price, the Standard Segment might pivot towards it.
Well-Off seem to be loyal and not affected by price, so Brand 4 could increase its price.
''')
st.write('')
st.write("### **Purchase probability**")

segment_dummies = pd.get_dummies(purchase_segm_kmeans_pca, prefix = 'Segment',prefix_sep = '_')
df_purchase_predictors = pd.concat([df_purchase_predictors, segment_dummies], axis = 1)
df_pa = df_purchase_predictors

Y = df_pa['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa['Price_1'] +
                   df_pa['Price_2'] +
                   df_pa['Price_3'] +
                   df_pa['Price_4'] +
                   df_pa['Price_5']) / 5

model_purchase = LogisticRegression(solver ='sag')#sag is optimal for simple problems with large datasets
model_purchase.fit(X, Y)

st.write(f"Upon running a logistic regression to see the relation between the average price of a product and the purchase incidence, a coefficient of {model_purchase.coef_} showed that a decrease in price would lead to an increase in purchase probability")

st.write('Price elasticity measures how a variable of interest changes when the price changes.')
st.write('Price elasticity of purchase probability is the % change in purchase probability in response to a 1% change in price')
st.write('Own Price Elasticity is price elasticity with respect to the same product')
st.write('Cross Price Elasticity Price elasticity with respect to another product')

st.write('### Price Elasticity of Purchase Probability')


row7_1, row7_2 = st.beta_columns((1,1))
with row7_1, _lock:
    st.markdown('''
    Elasticity E = ((change in Pr(purchase))/Pr(purchase)) / ((change in Price)/Price)

    E = beta( i.e coefficient for the model) * price * (1-Pr(purchase))
    
    ''')
with row7_2, _lock:
    st.write(df_pa[['Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price_5']].describe())

price_range = np.arange(0.5, 3.5, 0.01)

df_price_range = pd.DataFrame(price_range)


Y_pr = model_purchase.predict_proba(df_price_range)
purchase_pr = Y_pr[:][:, 1]

pe = model_purchase.coef_[:, 0] * price_range * (1 - purchase_pr)
df_price_elasticities = pd.DataFrame(price_range)

df_price_elasticities = df_price_elasticities.rename(columns = {0: "Price_Point"})
df_price_elasticities['Mean_PE'] = pe
#pd.options.display.max_rows = None

row8_1, row8_2 = st.beta_columns((1,1))
with row8_1, _lock:
    st.write('Price Elasticity of Purchase Probability')
    fig = plt.figure(figsize= (9,6))
    plt.plot(price_range, pe, color='grey')
    plt.xlabel('Price')
    plt.ylabel('Elasticity')
    st.pyplot(fig)
with row8_2, _lock:
    st.write('')
    st.write('')
    st.write('')
    st.write('''
        ineslastic |E| < 1, elastic |E| > 1

        e.g. an increase in 1% at the price 1.1 would lead to a decrease in purchase probability by 0.69% so inelastic

        an increase in 1% at the price 1.5 would lead to an decrease in purchase probability by 1.7% so elastic

        As such for inelastic values, general recommendation is to increase the price since there wouldn't be much of a decrease in purchase probability whereas vice versa for elastic

        For prices lower than 1.25(inelasticity) I can increase the price but past 1.25 there's more to gain by reducing the price     
    ''')

st.write("### Purchase Probability by Segments")

df_pa_segment_3 = df_pa[df_pa['Segment'] == 3]

Y = df_pa_segment_3['Incidence']

X  = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_3['Price_1'] +
                   df_pa_segment_3['Price_2'] +
                   df_pa_segment_3['Price_3'] +
                   df_pa_segment_3['Price_4'] +
                   df_pa_segment_3['Price_5']) / 5


model_incidence_segment_3 = LogisticRegression(solver='sag')
model_incidence_segment_3.fit(X, Y)

#model_incidence_segment_3.coef_ # has a lower impact, and less elastic

Y_segment_3 = model_incidence_segment_3.predict_proba(df_price_range)
purchase_pr_segment_3 = Y_segment_3[:][:,1]
pe_segment_3 = model_incidence_segment_3.coef_[:, 0] * price_range * (1 - purchase_pr_segment_3)


df_price_elasticities['PE_segment_3'] = pe_segment_3


df_pa_segment_1 = df_pa[df_pa['Segment'] == 1]

Y = df_pa_segment_1['Incidence']

X  = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_1['Price_1'] +
                   df_pa_segment_1['Price_2'] +
                   df_pa_segment_1['Price_3'] +
                   df_pa_segment_1['Price_4'] +
                   df_pa_segment_1['Price_5']) / 5


model_incidence_segment_1 = LogisticRegression(solver='sag')
model_incidence_segment_1.fit(X, Y)

Y_segment_1 = model_incidence_segment_1.predict_proba(df_price_range)
purchase_pr_segment_1 = Y_segment_1[:][:,1]
pe_segment_1 = model_incidence_segment_1.coef_[:, 0] * price_range * (1 - purchase_pr_segment_1)

df_price_elasticities['PE_segment_1'] = pe_segment_1

df_pa_segment_2 = df_pa[df_pa['Segment'] == 2]

Y = df_pa_segment_2['Incidence']

X  = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_2['Price_1'] +
                   df_pa_segment_2['Price_2'] +
                   df_pa_segment_2['Price_3'] +
                   df_pa_segment_2['Price_4'] +
                   df_pa_segment_2['Price_5']) / 5

model_incidence_segment_2 = LogisticRegression(solver='sag')
model_incidence_segment_2.fit(X, Y)

Y_segment_2 = model_incidence_segment_2.predict_proba(df_price_range)
purchase_pr_segment_2 = Y_segment_2[:][:,1]
pe_segment_2 = model_incidence_segment_2.coef_[:, 0] * price_range * (1 - purchase_pr_segment_2)

df_price_elasticities['PE_segment_2'] = pe_segment_2


df_pa_segment_0 = df_pa[df_pa['Segment'] == 0]

Y = df_pa_segment_0['Incidence']

X  = pd.DataFrame()
X['Mean_Price'] = (df_pa_segment_0['Price_1'] +
                   df_pa_segment_0['Price_2'] +
                   df_pa_segment_0['Price_3'] +
                   df_pa_segment_0['Price_4'] +
                   df_pa_segment_0['Price_5']) / 5



model_incidence_segment_0 = LogisticRegression(solver='sag')
model_incidence_segment_0.fit(X, Y)


Y_segment_0 = model_incidence_segment_0.predict_proba(df_price_range)
purchase_pr_segment_0 = Y_segment_0[:][:,1]
pe_segment_0 = model_incidence_segment_0.coef_[:, 0] * price_range * (1 - purchase_pr_segment_0)

df_price_elasticities['PE_segment_0'] = pe_segment_0

row9_1, row9_2 = st.beta_columns((1,1))
with row9_1, _lock:
    st.write('Displaying all price elasticities of purchase probability on the same plot.')
    fig = plt.figure(figsize = (9,6))
    plt.plot(price_range, pe, color='grey', label="All")
    plt.plot(price_range, pe_segment_0, color='blue', label="Standard")
    plt.plot(price_range, pe_segment_1, color='green', label="Career-Focused")
    plt.plot(price_range, pe_segment_2, color='red', label="Fewer-Opportunities")
    plt.plot(price_range, pe_segment_3, color='orange', label="Well-Off")
    plt.xlabel('Price')
    plt.ylabel('Elasticity')
    plt.legend(loc='lower left')
    st.pyplot(fig)

with row9_2, _lock:
    st.write('')
    st.write('')
    st.write('''
    Career-focused segment are the least elastic when compared to the rest. So, their purchase probability elasticity is not as affected by price.
    
    Standard segment price elasticity seem to differ across price range. This may be due to the fact that
    the standard segment is least homogenous, discovered during descriptive analysis. It may be that the customers in this segment have different shopping habbits, which is why the
    customers start with being more elastic than average but then shift to being more inelastic than the average customer
    and indeed the Career-focused segment.


    ''')

st.write('''
    Fewer opportunities segment are is price sensitive compared to the mean
    The point of inelasticity limit is 1.39 which is 14 cents higher than the average turning point 
    Inelasticity, increase price between 0.5 and 1.39 and decrease them afterwards so as to target purchase prob for Career Focused
    Fewer opportunities segment are is price sensitive compared to the mean
    With an increase in price they become more and more elastic, much faster
    1.27 is the tipping point as such more inelasticity at lower points..
    Biggest cluster so maybe more inelastcity sophisticated
    Cluster enjoys the product such that a price increase in the low price range doesn't affect their willingness to buy
    More expensive it becomes, the less their will in buying
''')

st.write('')
st.write('### *Purchase Probability with Promotion Feature*')

Y = df_pa['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa['Price_1'] + 
                   df_pa['Price_2'] + 
                   df_pa['Price_3'] + 
                   df_pa['Price_4'] + 
                   df_pa['Price_5']) / 5


# Include a second promotion feature. I'd like to examine the effects of promotions on purchase probability.
# Calculate the average promotion rate across the five brands. Add the mean price for the brands.
X['Mean_Promotion'] = (df_pa['Promotion_1'] +
                       df_pa['Promotion_2'] +
                       df_pa['Promotion_3'] +
                       df_pa['Promotion_4'] +
                       df_pa['Promotion_5'] ) / 5

# The coefficient for promotion is positive. 
# Therefore, there is a positive relationship between promotion and purchase probability.
model_incidence_promotion = LogisticRegression(solver = 'sag')
model_incidence_promotion.fit(X, Y)

# ## Price Elasticity with Promotion

# We create a data frame on which our model will predict. We need to include A price and promotion feature.
# First, we'll include the price range as the price feature. Next, we'll include the promotion feature.
df_price_elasticity_promotion = pd.DataFrame(price_range)
df_price_elasticity_promotion = df_price_elasticity_promotion.rename(columns = {0: "Price_Range"})

# We'll calculate price elasticities of purchase probability when we assume there is a promotion across at each price points.
df_price_elasticity_promotion['Promotion'] = 1

# Purchase Probability with Promotion Model Prediction
Y_promotion = model_incidence_promotion.predict_proba(df_price_elasticity_promotion)

promo = Y_promotion[:, 1]
price_elasticity_promo = (model_incidence_promotion.coef_[:, 0] * price_range) * (1 - promo)


# Update master data to include elasticities of purchase probability with promotion feature
df_price_elasticities['Elasticity_Promotion_1'] = price_elasticity_promo


# ## Price Elasticity without Promotion

df_price_elasticity_promotion_no = pd.DataFrame(price_range)
df_price_elasticity_promotion_no = df_price_elasticity_promotion_no.rename(columns = {0: "Price_Range"})



# Promotion feature -No Promotion.
# We assume there aren't any promotional activities on any of the price points.
# We examine the elasticity of purchase probability when there isn't promotion.
df_price_elasticity_promotion_no['Promotion'] = 0

#Purchase Probability without Promotion Model Prediction
Y_no_promo = model_incidence_promotion.predict_proba(df_price_elasticity_promotion_no)


no_promo = Y_no_promo[: , 1]


price_elasticity_no_promo = model_incidence_promotion.coef_[:, 0] * price_range *(1- no_promo)

# Update master data frame to include purchase probability elasticities without promotion.
# We can now see the values with and without promotion and compare them for each price point in our price range.
df_price_elasticities['Elasticity_Promotion_0'] = price_elasticity_no_promo

plt.figure(figsize = (9, 6))
plt.plot(price_range, price_elasticity_no_promo)
plt.plot(price_range, price_elasticity_promo)
plt.xlabel('Price')
plt.ylabel('Elasticity')
plt.title('Price Elasticity of Purchase Probability with and without Promotion')

row10_1, row10_2 = st.beta_columns((1,1))
with row10_1, _lock:
    st.write('Displaying all price elasticities of purchase probability on the same plot.')
    fig = plt.figure(figsize = (9, 6))
    plt.plot(price_range, price_elasticity_no_promo, label="Without promotion")
    plt.plot(price_range, price_elasticity_promo, label="With promotion")
    plt.xlabel('Price')
    plt.ylabel('Elasticity')
    plt.legend(loc='lower left')
    st.pyplot(fig)

with row10_2, _lock:
    st.write('')
    st.write('')
    st.write('''
    Plot purchase elasticities with and without promotion side by side for comprarisson.
    Observe that the purchase probability elasticity of the customer is less elastic when there is promotion.
    This is an important insight for marketers, as according to our model people are more likely to buy a product if there is
    some promotional activity rather than purchase a product with the same price, when it isn't on promotion. 

    ''')
st.write('### **Brand choice**')

# Here we are interested in determining the brand choice of the customer. 
# Hence, we filter our data, to include only purchase occasion, when a purchase has occured. 
brand_choice = df_pa[df_pa['Incidence'] == 1]


# Our model will predict the brand.
Y = brand_choice['Brand']
# We predict based on the prices for the five brands.
features = ['Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price_5']
X = brand_choice[features]
# Brand Choice Model fit.
model_brand_choice = LogisticRegression(solver = 'sag', multi_class = 'multinomial')
model_brand_choice.fit(X, Y)

# Here are the coeffictients for the model. We have five brands and five features for the price. 

bc_coef = pd.DataFrame(model_brand_choice.coef_)

bc_coef = pd.DataFrame(np.transpose(model_brand_choice.coef_))
coefficients = ['Coef_Brand_1', 'Coef_Brand_2', 'Coef_Brand_3', 'Coef_Brand_4', 'Coef_Brand_5']
bc_coef.columns = [coefficients]
prices = ['Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price_5']
bc_coef.index = [prices]
bc_coef = bc_coef.round(2)

row11_1, row11_2 = st.beta_columns((1,1))
with row11_1, _lock:
    bc_coef

with row11_2, _lock:
    st.write('''
    I make some transformations on the coefficients data frame to increase readability.
    I transpose the data frame, to keep with the conventional representation of results.
    I add labels for the columns and the index, which represent the coefficients of the brands and prices, respectively. 
    ''')

st.write('')
st.write('### **Own and cross brand Price elsticity**')


# We want to calculate price elasticity of brand choice.
# Here we create a data frame with price columns, which our model will use to predict the brand choice probabilities.
df_own_brand_5 = pd.DataFrame(index = np.arange(price_range.size))
df_own_brand_5['Price_1'] = brand_choice['Price_1'].mean()
df_own_brand_5['Price_2'] = brand_choice['Price_2'].mean()
df_own_brand_5['Price_3'] = brand_choice['Price_3'].mean()
df_own_brand_5['Price_4'] = brand_choice['Price_4'].mean()
df_own_brand_5['Price_5'] = price_range

# Brand Choice Model prediction.
predict_brand_5 = model_brand_choice.predict_proba(df_own_brand_5)

# Our model returns the probabilities of choosing each of the 5 brands. 
# Since, we are interested in the probability for the fifth brand we need to obtain the last column located on position 4,
# as we're starting to count from 0.
pr_own_brand_5 = predict_brand_5[: ][:, 4]

# We're interested in choosing brand 5. 
# Therefore, the beta coefficient we require is that of the brand 5 coefficient and price 5.
beta5 = bc_coef.iloc[4, 4]

# Calculating price elasticities for brand choice without promotion. 
own_price_elasticity_brand_5 = beta5 * price_range * (1 - pr_own_brand_5)

# Adding the price elasticities to our master data frame. 
df_price_elasticities['Brand_5'] = own_price_elasticity_brand_5

# Plot elasticities of purchase probability for brand 5.
plt.figure(figsize = (9, 6))
plt.plot(price_range, own_price_elasticity_brand_5, color = 'grey')
plt.xlabel('Price 5')
plt.ylabel('Elasticity')
plt.title('Own Price Elasticity of Purchase Probability for Brand 5')


# ## Cross Price Elasticity Brand 5, Cross Brand 4


# We want to examine the effect of the changes in price of a competitor brand.
# As we've discussed in the lecture, the brand which comes closest to our own brand is brand 4. 
# Therefore, we need to examine changes in the price of this brand.
# Keep in mind, we could examine the cross price elasticities for any of the remaining brands, 
# we just need to update this data frame accordingly to contain the respective brand.
df_brand5_cross_brand4 = pd.DataFrame(index = np.arange(price_range.size))
df_brand5_cross_brand4['Price_1'] = brand_choice['Price_1'].mean()
df_brand5_cross_brand4['Price_2'] = brand_choice['Price_2'].mean()
df_brand5_cross_brand4['Price_3'] = brand_choice['Price_3'].mean()
df_brand5_cross_brand4['Price_4'] = price_range
df_brand5_cross_brand4['Price_5'] = brand_choice['Price_5'].mean()

predict_brand5_cross_brand4 = model_brand_choice.predict_proba(df_brand5_cross_brand4)

# As now we're interested in what the probability of choosing the competitor brand is, 
# we need to select the purchase probability for brand 4, contained in the 4th column with index 3. 
pr_brand_4 = predict_brand5_cross_brand4[:][:, 3]

# In order to calculate the cross brand price elasticity, we need to use the new formula we introduced in the lecture.
# The elasticity is equal to negative the price coefficient of the own brand multiplied by the price of the cross brand,
# further multiplied by the probability for choosing the cross brand.
brand5_cross_brand4_price_elasticity = -beta5 * price_range * pr_brand_4

# Update price elasticities data frame to include the cross price elasticities for brand 5 with respect to brand 4.
df_price_elasticities['Brand_5_Cross_Brand_4'] = brand5_cross_brand4_price_elasticity

# Here we examine the cross price elasticity of purchase probability for brand 5 with respect to brand 4.
# We observe they are positive. As the price of the competitor brand increases, 
# so does the probability for purchasing our own brand.
# Even though the elasticity starts to decrease from the 1.45 mark, it is still positive, 
# signalling that the increase in purchase probability for the own brand happens more slowly.

# ## Own and Cross-Price Elasticity by Segment

# We are interested in analysing the purchase probability for choosing brand 5 by segments.
# We filter our data to contain only purchase incidences of the third segment - Well-off.
brand_choice_s3 = df_pa[df_pa['Incidence'] == 1]
brand_choice_s3 = brand_choice_s3[brand_choice_s3['Segment'] == 3]

# Brand Choice Model estimation.
Y = brand_choice_s3['Brand']
brand_choice_s3 = pd.get_dummies(brand_choice_s3, columns=['Brand'], prefix = 'Brand', prefix_sep = '_')
X = brand_choice_s3[features]
model_brand_choice_s3 = LogisticRegression(solver = 'sag', multi_class = 'multinomial', max_iter = 300)
model_brand_choice_s3.fit(X, Y)

# Coefficients table for segment 3
bc_coef = pd.DataFrame(np.transpose(model_brand_choice_s3.coef_))
bc_coef.columns = [coefficients]
bc_coef.index = [prices]
bc_coef = bc_coef.round(2)

# Calculating own-brand price elasticity for brand 5 and the Well-off segment.
df_own_brand_5_s3 = pd.DataFrame(index = np.arange(price_range.size))
df_own_brand_5_s3['Price_1'] = brand_choice_s3['Price_1'].mean()
df_own_brand_5_s3['Price_2'] = brand_choice_s3['Price_2'].mean()
df_own_brand_5_s3['Price_3'] = brand_choice_s3['Price_3'].mean()
df_own_brand_5_s3['Price_4'] = brand_choice_s3['Price_4'].mean()
df_own_brand_5_s3['Price_5'] = price_range

predict_own_brand_5_s3 = model_brand_choice_s3.predict_proba(df_own_brand_5_s3)
pr_own_brand_5_s3 = predict_own_brand_5_s3[: ][: , 4]

own_price_elasticity_brand_5_s3 =  beta5 * price_range * (1 - pr_own_brand_5_s3)
df_price_elasticities['Brand 5 S3'] = own_price_elasticity_brand_5_s3


# Calculating cross-brand price elasticity for brand 5 with respect to brand 4 for the Well-off segment.
df_brand5_cross_brand4_s3 = pd.DataFrame(index = np.arange(price_range.size))
df_brand5_cross_brand4_s3['Price_1'] = brand_choice_s3['Price_1'].mean()
df_brand5_cross_brand4_s3['Price_2'] = brand_choice_s3['Price_2'].mean()
df_brand5_cross_brand4_s3['Price_3'] = brand_choice_s3['Price_3'].mean()
df_brand5_cross_brand4_s3['Price_4'] = price_range
df_brand5_cross_brand4_s3['Price_5'] = brand_choice_s3['Price_5'].mean()

predict_brand5_cross_brand4_s3 = model_brand_choice_s3.predict_proba(df_brand5_cross_brand4_s3)
pr_cross_brand_5_s3 = predict_brand5_cross_brand4_s3[: ][: , 3]

# Update master data frame to include the newly obtained cross-brand price elasticities.
brand5_cross_brand4_price_elasticity_s3 = -beta5 * price_range * pr_cross_brand_5_s3
df_price_elasticities['Brand_5_Cross_Brand_4_S3'] = brand5_cross_brand4_price_elasticity_s3

# Using a figure with axes we plot the own brand and cross-brand price elasticities for brand 5 cross brand 4 side by side.
fig, axs = plt.subplots(1, 2, figsize = (14, 4))
axs[0].plot(price_range, own_price_elasticity_brand_5_s3, color = 'orange')
axs[0].set_title('Brand 5 Segment Well-Off')
axs[0].set_xlabel('Price 5')

axs[1].plot(price_range, brand5_cross_brand4_price_elasticity_s3, color = 'orange')
axs[1].set_title('Cross Price Elasticity of Brand 5 wrt Brand 4 Segment Well-Off')
axs[1].set_xlabel('Price 4')

for ax in axs.flat:
    ax.set(ylabel = 'Elasticity')


# Here we are interesting in analysing the brand choice probability of the Standard segment.
# We filter our data, by selecting only purchases from segment 0.
brand_choice_s0 = df_pa[df_pa['Incidence'] == 1]
brand_choice_s0 = brand_choice_s0[brand_choice_s0['Segment'] == 0]


# Brand Choice Model estimation.
Y = brand_choice_s0['Brand']
brand_choice_s0 = pd.get_dummies(brand_choice_s0, columns=['Brand'], prefix = 'Brand', prefix_sep = '_')
X = brand_choice_s0[features]
model_brand_choice_s0 = LogisticRegression(solver = 'sag', multi_class = 'multinomial', max_iter = 200)
model_brand_choice_s0.fit(X, Y)

# Coefficients table segment 0.
bc_coef = pd.DataFrame(np.transpose(model_brand_choice_s0.coef_))
bc_coef.columns = [coefficients]
bc_coef.index = [prices]
bc_coef = bc_coef.round(2)


# Calculating own-brand price elasticity for brand 5 and the Standard segment.
df_own_brand_5_s0 = pd.DataFrame(index = np.arange(price_range.size))
df_own_brand_5_s0['Price_1'] = brand_choice_s0['Price_1'].mean()
df_own_brand_5_s0['Price_2'] = brand_choice_s0['Price_2'].mean()
df_own_brand_5_s0['Price_3'] = brand_choice_s0['Price_3'].mean()
df_own_brand_5_s0['Price_4'] = brand_choice_s0['Price_4'].mean()
df_own_brand_5_s0['Price_5'] = price_range

predict_own_brand_5_s0 = model_brand_choice_s0.predict_proba(df_own_brand_5_s0)
pr_own_brand_5_s0 = predict_own_brand_5_s0[: ][: , 4]

# Compute price elasticities and update master data frame.
# We'd like to include the elasticities for the segments in order from 0 to three, which is why we use insert() on position 10.
own_price_elasticity_brand_5_s0 =  beta5 * price_range * (1 - pr_own_brand_5_s0)
df_price_elasticities.insert(10, column = 'Brand 5 S0', value = own_price_elasticity_brand_5_s0)



# Calculating cross-brand price elasticity for brand 5 with respect to brand 4 for the Standard segment.
df_brand5_cross_brand4_s0 = pd.DataFrame(index = np.arange(price_range.size))
df_brand5_cross_brand4_s0['Price_1'] = brand_choice_s0['Price_1'].mean()
df_brand5_cross_brand4_s0['Price_2'] = brand_choice_s0['Price_2'].mean()
df_brand5_cross_brand4_s0['Price_3'] = brand_choice_s0['Price_3'].mean()
df_brand5_cross_brand4_s0['Price_4'] = price_range
df_brand5_cross_brand4_s0['Price_5'] = brand_choice_s0['Price_5'].mean()

predict_brand5_cross_brand4_s0 = model_brand_choice_s0.predict_proba(df_brand5_cross_brand4_s0)
pr_cross_brand_5_s0 = predict_brand5_cross_brand4_s0[: ][: , 3]

# Compute price elasticities and update master data frame.
# We need to use insert() on position 11, to save the price elasticities in the correct order.
brand5_cross_brand4_price_elasticity_s0 = -beta5 * price_range * pr_cross_brand_5_s0
df_price_elasticities.insert(11, column = 'Brand_5_Cross_Brand_4_S0', value = brand5_cross_brand4_price_elasticity_s0)


# Filter data by the Career-focused segment, which is the first segment.
brand_choice_s1 = df_pa[df_pa['Incidence'] == 1]
brand_choice_s1 = brand_choice_s1[brand_choice_s1['Segment'] == 1]


# Brand Choice Model estimation.
Y = brand_choice_s1['Brand']
brand_choice_s1 = pd.get_dummies(brand_choice_s1, columns=['Brand'], prefix = 'Brand', prefix_sep = '_')
X = brand_choice_s1[features]
model_brand_choice_s1 = LogisticRegression(solver = 'sag', multi_class = 'multinomial', max_iter = 300)
model_brand_choice_s1.fit(X, Y)

# Coefficients table segment 1
bc_coef = pd.DataFrame(np.transpose(model_brand_choice_s1.coef_))
bc_coef.columns = [coefficients]
bc_coef.index = [prices]
bc_coef = bc_coef.round(2)


# Calculating own-brand price elasticity for brand 5 and the Career-focused segment.
df_own_brand_5_s1 = pd.DataFrame(index = np.arange(price_range.size))
df_own_brand_5_s1['Price_1'] = brand_choice_s1['Price_1'].mean()
df_own_brand_5_s1['Price_2'] = brand_choice_s1['Price_2'].mean()
df_own_brand_5_s1['Price_3'] = brand_choice_s1['Price_3'].mean()
df_own_brand_5_s1['Price_4'] = brand_choice_s1['Price_4'].mean()
df_own_brand_5_s1['Price_5'] = price_range

predict_own_brand_5_s1 = model_brand_choice_s1.predict_proba(df_own_brand_5_s1)
pr_own_brand_5_s1 = predict_own_brand_5_s1[: ][: , 4]

#compute price elasticities and update data frame
own_price_elasticity_brand_5_s1 =  beta5 * price_range * (1 - pr_own_brand_5_s1)
df_price_elasticities.insert(12, column = 'Brand 5 S1', value = own_price_elasticity_brand_5_s1)


# Calculating cross-brand price elasticity for brand 5 with respect to brand 4 for the Career-focused segment.
df_brand5_cross_brand4_s1 = pd.DataFrame(index = np.arange(price_range.size))
df_brand5_cross_brand4_s1['Price_1'] = brand_choice_s1['Price_1'].mean()
df_brand5_cross_brand4_s1['Price_2'] = brand_choice_s1['Price_2'].mean()
df_brand5_cross_brand4_s1['Price_3'] = brand_choice_s1['Price_3'].mean()
df_brand5_cross_brand4_s1['Price_4'] = price_range
df_brand5_cross_brand4_s1['Price_5'] = brand_choice_s1['Price_5'].mean()

predict_brand5_cross_brand4_s1 = model_brand_choice_s1.predict_proba(df_brand5_cross_brand4_s1)
pr_cross_brand_5_s1 = predict_brand5_cross_brand4_s1[: ][: , 3]

brand5_cross_brand4_price_elasticity_s1 = -beta5 * price_range * pr_cross_brand_5_s1
df_price_elasticities.insert(13, column = 'Brand_5_Cross_Brand_4_S1', value = brand5_cross_brand4_price_elasticity_s1)

# Filter data, select only purchases from segment 2, which is the Fewer-Opportunities segment.
brand_choice_s2 = df_pa[df_pa['Incidence'] == 1]
brand_choice_s2 = brand_choice_s2[brand_choice_s2['Segment'] == 2]


# Brand Choice Model estimation.
Y = brand_choice_s2['Brand']
brand_choice_s2 = pd.get_dummies(brand_choice_s2, columns=['Brand'], prefix = 'Brand', prefix_sep = '_')
X = brand_choice_s2[features]
model_brand_choice_s2 = LogisticRegression(solver = 'sag', multi_class = 'multinomial', max_iter = 300)
model_brand_choice_s2.fit(X, Y)

# Coefficients table segment 2
bc_coef = pd.DataFrame(np.transpose(model_brand_choice_s2.coef_))
bc_coef.columns = [coefficients]
bc_coef.index = [prices]
bc_coef = bc_coef.round(2)

# Calculating own-brand price elasticity for brand 5 and the Fewer-opportunities segment.
df_own_brand_5_s2 = pd.DataFrame(index = np.arange(price_range.size))
df_own_brand_5_s2['Price_1'] = brand_choice_s2['Price_1'].mean()
df_own_brand_5_s2['Price_2'] = brand_choice_s2['Price_2'].mean()
df_own_brand_5_s2['Price_3'] = brand_choice_s2['Price_3'].mean()
df_own_brand_5_s2['Price_4'] = brand_choice_s2['Price_4'].mean()
df_own_brand_5_s2['Price_5'] = price_range

predict_own_brand_5_s2 = model_brand_choice_s2.predict_proba(df_own_brand_5_s2)
pr_own_brand_5_s2 = predict_own_brand_5_s2[: ][: , 4]

#compute price elasticities and update data frame
own_price_elasticity_brand_5_s2 =  beta5 * price_range * (1 - pr_own_brand_5_s2)
df_price_elasticities.insert(14, column = 'Brand 5 S2', value = own_price_elasticity_brand_5_s2)

# Calculating cross-brand price elasticity for brand 5 with respect to brand 4 for the Fewer-opportunities segment.
df_brand5_cross_brand4_s2 = pd.DataFrame(index = np.arange(price_range.size))
df_brand5_cross_brand4_s2['Price_1'] = brand_choice_s2['Price_1'].mean()
df_brand5_cross_brand4_s2['Price_2'] = brand_choice_s2['Price_2'].mean()
df_brand5_cross_brand4_s2['Price_3'] = brand_choice_s2['Price_3'].mean()
df_brand5_cross_brand4_s2['Price_4'] = price_range
df_brand5_cross_brand4_s2['Price_5'] = brand_choice_s2['Price_5'].mean()

predict_brand5_cross_brand4_s2 = model_brand_choice_s2.predict_proba(df_brand5_cross_brand4_s2)
pr_cross_brand_5_s2 = predict_brand5_cross_brand4_s2[: ][: , 3]

brand5_cross_brand4_price_elasticity_s2 = -beta5 * price_range * pr_cross_brand_5_s2
df_price_elasticities.insert(15, column = 'Brand_5_Cross_Brand_4_S2', value = brand5_cross_brand4_price_elasticity_s2)

fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 2, figsize = (11, 9), sharex = True)
ax1[0].plot(price_range, own_price_elasticity_brand_5, 'tab:grey')
ax1[0].set_title('Brand 5 Average Customer')
ax1[0].set_ylabel('Elasticity')
ax1[1].plot(price_range, brand5_cross_brand4_price_elasticity, 'tab:grey')
ax1[1].set_title('Cross Brand 4 Average Customer')

st.pyplot(fig)

ax2[0].plot(price_range, own_price_elasticity_brand_5_s0)
ax2[0].set_title('Brand 5 Segment Standard')
ax2[0].set_ylabel('Elasticity')
ax2[1].plot(price_range, brand5_cross_brand4_price_elasticity_s0)
ax2[1].set_title('Cross Brand 4 Segment Standard')
st.pyplot(fig)


ax3[0].plot(price_range, own_price_elasticity_brand_5_s1, 'tab:green')
ax3[0].set_title('Brand 5 Segment Career-Focused')
ax3[0].set_ylabel('Elasticity')
ax3[1].plot(price_range, brand5_cross_brand4_price_elasticity_s1, 'tab:green')
ax3[1].set_title('Cross Brand 4 Segment Career-Focused')
st.pyplot(fig)

ax4[0].plot(price_range, own_price_elasticity_brand_5_s2, 'tab:red')
ax4[0].set_title('Brand 5 Segment Fewer-Opportunities')
ax4[0].set_ylabel('Elasticity')
ax4[1].plot(price_range, brand5_cross_brand4_price_elasticity_s2, 'tab:red')
ax4[1].set_title('Cross Brand 4 Segment Fewer-Opportunities')
st.pyplot(fig)


ax5[0].plot(price_range, own_price_elasticity_brand_5_s3, 'tab:orange')
ax5[0].set_title('Brand 5 Segment Well-off')
ax5[0].set_xlabel('Price 5')
ax5[0].set_ylabel('Elasticity')
ax5[1].plot(price_range, brand5_cross_brand4_price_elasticity_s3, 'tab:orange')
ax5[1].set_title('Cross Brand 4 Segment Well-off')
ax5[1].set_xlabel('Price 4')

st.pyplot(fig)

st.write('''

Plot the own and cross brand price elasticities for the average customer and each of the four segments.
Observe differences and similiraties between the segments and examine their preference, when it comes to brand choice.
The two segments, which seem to be of most interested for the marketing team of brand 5, seem to be the Career-focused
and the Well-off. They are also the segments which purchase this brand most often. 
The Career-focused segment is the most inelastic and they are the most loyal segment. 
Based on our model, they do not seem to be that affected by price, therefore brand 5 could increase its price, 
without fear of significant loss of customers from this segment. 
The Well-off segment on the other hand, seems to be more elastic. They also purchase the competitor brand 4 most often.
In order to target this segment, our analysis signals, that price needs to be decreased. However, keep in mind 
that other factors aside from price might be influencing the purchase behaivour of this segment.

''')

st.write('')
st.write('### **Price Elasticity of Purchase Quantity**')

st.write('''
    I want to determine price elasticity of purchase quantity, also known as price elasticity of demand.
    I'm interested in purchase ocassion, where the purchased quantity is different from 0.
    Therefore, once again we filter our data to contain only shopping visits where the client has purchased at least one product.
''')
df_purchase_quantity = df_pa[df_pa['Incidence'] == 1]

# Create brand dummies, for each of the five brands.
df_purchase_quantity = pd.get_dummies(df_purchase_quantity, columns = ['Brand'], prefix = 'Brand', prefix_sep = '_')
st.write('The descriptive analysis of the purchase quantitiy data frame, shows that quantity ranges from 1 to 15 and has an average value of 2.8, which means that more often than not our customers buy more than 1 chocolate candy bar.')

#Find the price of the product that is chosen at this incidence
df_purchase_quantity['Price_Incidence'] = (df_purchase_quantity['Brand_1'] * df_purchase_quantity['Price_1'] +
                                           df_purchase_quantity['Brand_2'] * df_purchase_quantity['Price_2'] +
                                           df_purchase_quantity['Brand_3'] * df_purchase_quantity['Price_3'] +
                                           df_purchase_quantity['Brand_4'] * df_purchase_quantity['Price_4'] +
                                           df_purchase_quantity['Brand_5'] * df_purchase_quantity['Price_5'] )

df_purchase_quantity['Promotion_Incidence'] = (df_purchase_quantity['Brand_1'] * df_purchase_quantity['Promotion_1'] +
                                               df_purchase_quantity['Brand_2'] * df_purchase_quantity['Promotion_2'] +
                                               df_purchase_quantity['Brand_3'] * df_purchase_quantity['Promotion_3'] +
                                               df_purchase_quantity['Brand_4'] * df_purchase_quantity['Promotion_4'] +
                                               df_purchase_quantity['Brand_5'] * df_purchase_quantity['Promotion_5'] )

X = df_purchase_quantity[['Price_Incidence', 'Promotion_Incidence']]
Y = df_purchase_quantity['Quantity']

model_quantity = LinearRegression()
model_quantity.fit(X, Y)

# Linear Regression Model. The coefficients for price and promotion are both negative. 
# It appears that promotion reflects negatively on the purchase quantity of the average client, which is unexpected.
model_quantity.coef_

# We examine the price elasticity of purchase quantity with active promotional activities for each price point.
df_price_elasticity_quantity = pd.DataFrame(index = np.arange(price_range.size))
df_price_elasticity_quantity['Price_Incidence'] = price_range
df_price_elasticity_quantity['Promotion_Incidence'] = 1

beta_quantity = model_quantity.coef_[0]

predict_quantity = model_quantity.predict(df_price_elasticity_quantity)

# We calculate the price elasticity with our new formula. It is the beta coefficient for price multiplied by price
# and divided by the purchase quantity.
price_elasticity_quantity_promotion_yes = beta_quantity * price_range / predict_quantity

df_price_elasticities['PE_Quantity_Promotion_1'] = price_elasticity_quantity_promotion_yes

# Here we assume there are no promotinal activities active for the entire price range.
df_price_elasticity_quantity['Promotion_Incidence'] = 0
# Find the new predicted quantities.
predict_quantity = model_quantity.predict(df_price_elasticity_quantity)
# Calculate the new price elasticities.
price_elasticity_quantity_promotion_no = beta_quantity * price_range / predict_quantity
# Add the results to the master data frame.
df_price_elasticities['PE_Quantity_Promotion_0'] = price_elasticity_quantity_promotion_no

# Plot the two elasticities side by side. 
# We observe that the two elasticities are very close together for almost the entire price range.
# It appears that promotion does not appear to be a significant factor in the customers' decission 
# what quantity of chocolate candy bars to purchase.

row12_1, row12_2 = st.beta_columns((1,1))
with row12_1, _lock:    
    fig = plt.figure(figsize = (9, 6))
    plt.plot(price_range, price_elasticity_quantity_promotion_yes, color = 'orange')
    plt.plot(price_range, price_elasticity_quantity_promotion_no)
    plt.xlabel('Price')
    plt.ylabel('Elasticity')
    plt.title('Price Elasticity of Purchase Quantity with Promotion')
    st.pyplot(fig)

with row12_2, _lock:
    st.write('''
    Plotting the two elasticities side by side. 
    Observe that the two elasticities are very close together for almost the entire price range.
    It appears that promotion does not appear to be a significant factor in the customers' decision what quantity of chocolate candy bars to purchase.
    ''')



