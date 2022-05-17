# Basic libraries
import pandas as pd
import os

# Natural Language Processing
from sklearn.feature_extraction import _stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib



# r/TheOnion DataFrame
df_onion = pd.read_csv('./data/the_onion.csv')

# r/nottheonion DataFrame
df_not_onion = pd.read_csv('./data/not_onion.csv')

# 
# Show first 5 rows of df_onion
print("Shape:", df_onion.shape)
df_onion.head()

# 
# Show first 5 rows of df_not_onion
print("Shape:", df_not_onion.shape)
df_not_onion.head()

#  
# ### Data Cleaning Function
def clean_data(dataframe):

    # Drop duplicate rows
    dataframe.drop_duplicates(subset='title', inplace=True)
    
    # Remove punctation
    dataframe['title'] = dataframe['title'].str.replace('[^\w\s]',' ')

    # Remove numbers 
    dataframe['title'] = dataframe['title'].str.replace('[^A-Za-z]',' ')

    # Make sure any double-spaces are single 
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')
    dataframe['title'] = dataframe['title'].str.replace('  ',' ')

    # Transform all text to lowercase
    dataframe['title'] = dataframe['title'].str.lower()
    
    print("New shape:", dataframe.shape)
    return dataframe.head()

# 
# Call `clean_data(dataframe)` function
clean_data(df_onion)

# 
# Call `clean_data(dataframe)` function
clean_data(df_not_onion)

# 
# Create a DataFrame to check nulls
pd.DataFrame([df_onion.isnull().sum(),df_not_onion.isnull().sum()], index=["TheOnion","notheonion"]).T

#  
# ### Date Range of Scraped Posts

# 
# Convert Unix Timestamp to Datetime
df_onion['timestamp'] = pd.to_datetime(df_onion['timestamp'], unit='s')
df_not_onion['timestamp'] = pd.to_datetime(df_not_onion['timestamp'], unit='s')

# Show date-range of posts scraped from r/TheOnion and r/nottheonion
print("TheOnion start date:", df_onion['timestamp'].min())
print("TheOnion end date:", df_onion['timestamp'].max())
print("nottheonion start date:", df_not_onion['timestamp'].min())
print("nottheonion end date:", df_not_onion['timestamp'].max())


#  
# ### r/TheOnion: Most Active Authors
df_onion_authors = df_onion['author'].value_counts() 
df_onion_authors = df_onion_authors[df_onion_authors > 100].sort_values(ascending=False)

# Set y values: Authors 
df_onion_authors_index = list(df_onion_authors.index)

#  
# ### r/nottheonion: Most Active Authors
df_not_onion_authors = df_not_onion['author'].value_counts() 
df_not_onion_authors = df_not_onion_authors[df_not_onion_authors > 100].sort_values(ascending=False)

# Set y values: Authors
df_not_onion_authors_index = list(df_not_onion_authors.index)

#  
# ### r/TheOnion: Most Referenced Domains
df_onion_domain = df_onion['domain'].value_counts() 
df_onion_domain = df_onion_domain.sort_values(ascending=False).head(10)

# Set y values: Domains 
df_onion_domain_index = list(df_onion_domain.index)

#  
# ### r/nottheonion: Most Referenced Domains
df_nonion_domain = df_not_onion['domain'].value_counts()
df_nonion_domain = df_nonion_domain.sort_values(ascending=False).head(10)

# Set y values: Names of authors 
df_nonion_domain_index = list(df_nonion_domain.index)

#  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # Natural Language Processing (NLP)

#  
# ### Concatenate DataFrames
df = pd.concat([df_onion[['subreddit', 'title']], df_not_onion[['subreddit', 'title']]], axis=0)

#Reset the index
df = df.reset_index(drop=True)

#  
# ### Binarize Target `subreddit`  #  - `TheOnion`: 1     #  - `nottheonion`: 0
df["subreddit"] = df["subreddit"].map({"nottheonion": 0, "TheOnion": 1})

#  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # ## Apply `CountVectorizer()`
#  - `ngram_range = (1,1)`

# 
# Set variables to show TheOnion Titles
mask_on = df['subreddit'] == 1
df_onion_titles = df[mask_on]['title']

# Instantiate a CountVectorizer
cv1 = CountVectorizer(stop_words = 'english')

# Fit and transform the vectorizer on our corpus
onion_cvec = cv1.fit_transform(df_onion_titles)

# Convert onion_cvec into a DataFrame
onion_cvec_df = pd.DataFrame(onion_cvec.toarray(),
                   columns=cv1.get_feature_names_out())

# Inspect head of Onion Titles cvec
print("----------------------------------------")
print(onion_cvec_df.head(5))


#  
# ### Count Vectorize `df` where `subreddit` is `0`
#  - `ngram_range = (1,1)`

# 
# Set variables to show NotTheOnion Titles
mask_no = df['subreddit'] == 0
df_not_onion_titles = df[mask_no]['title']

# Instantiate a CountVectorizer
cv2 = CountVectorizer(stop_words = 'english')

# Fit and transform the vectorizer on our corpus
not_onion_cvec = cv2.fit_transform(df_not_onion_titles)

# Convert onion_cvec into a DataFrame
not_onion_cvec_df = pd.DataFrame(not_onion_cvec.toarray(),
                   columns=cv2.get_feature_names_out())

# Inspect head of Not Onion Titles cvec
print(not_onion_cvec_df.shape)

#  
# ### r/TheOnion: Top 5 Unigrams 

# 
# Set up variables to contain top 5 most used words in Onion
onion_wc = onion_cvec_df.sum(axis = 0)
onion_top_5 = onion_wc.sort_values(ascending=False).head(5)

#  
# ### r/nottheonion: Top 5 Unigrams

# 
# Set up variables to contain top 5 most used words in Onion
nonion_wc = not_onion_cvec_df.sum(axis = 0)
nonion_top_5 = nonion_wc.sort_values(ascending=False).head(5)

#  
# ## Common Unigrams between Top 5 in r/TheOnion & r/nottheonion

# 
# Create list of unique words in top five
not_onion_5_set = set(nonion_top_5.index)
onion_5_set = set(onion_top_5.index)

# Return common words
common_unigrams = onion_5_set.intersection(not_onion_5_set)
common_unigrams

#  
# ###  # # # # # # # #Count Vectorize `df` where `subreddit` is `1`
#  - `ngram_range = (2,2)`

# 
# Set variables to show TheOnion Titles
mask = df['subreddit'] == 1
df_onion_titles = df[mask]['title']

# Instantiate a CountVectorizer
cv = CountVectorizer(stop_words = 'english', ngram_range=(2,2))

# Fit and transform the vectorizer on our corpus
onion_cvec = cv.fit_transform(df_onion_titles)

# Convert onion_cvec into a DataFrame
onion_cvec_df = pd.DataFrame(onion_cvec.toarray(),
                   columns=cv.get_feature_names_out())



#  
# ### Count Vectorize `df` where `subreddit` is `0`
#  - `ngram_range = (2,2)`

# 
# Set variables to show NotTheOnion Titles
mask = df['subreddit'] == 0
df_not_onion_titles = df[mask]['title']

# Instantiate a CountVectorizer
cv = CountVectorizer(stop_words = 'english', ngram_range=(2,2))

# Fit and transform the vectorizer on our corpus
not_onion_cvec = cv.fit_transform(df_not_onion_titles)

# Convert onion_cvec into a DataFrame
not_onion_cvec_df = pd.DataFrame(not_onion_cvec.toarray(),
                   columns=cv.get_feature_names_out())


#  
# ### r/TheOnion: Top 5 Bigrams

# 
# Set up variables to contain top 5 most used bigrams in r/TheOnion
onion_wc = onion_cvec_df.sum(axis = 0)
onion_top_5 = onion_wc.sort_values(ascending=False).head(5)

#  
# ### r/nottheonion: Top 5 Bigrams

# 
# Set up variables to contain top 5 most used bigrams in r/nottheonion
nonion_wc = not_onion_cvec_df.sum(axis = 0)
nonion_top_5 = nonion_wc.sort_values(ascending=False).head(5)

#  
# ### Common Bigrams between Top 5 in r/TheOnion & r/nottheonion

# 
not_onion_5_list = set(nonion_top_5.index)
onion_5_list = set(onion_top_5.index)

# Return common words
common_bigrams = onion_5_list.intersection(not_onion_5_list)
common_bigrams

#  
# ### Create custom `stop_words` to include common frequent words
# Create lists 
custom = _stop_words.ENGLISH_STOP_WORDS
custom = list(custom)
common_unigrams = list(common_unigrams)
common_bigrams = list(common_bigrams)

# Append unigrams to list 
for i in common_unigrams:
    custom.append(i)

    
# Append bigrams to list 
for i in common_bigrams:
    split_words = i.split(" ")
    for word in split_words:
        custom.append(word)


# 
# Baseline score
df['subreddit'].value_counts(normalize=True)

#  
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Modeling

# ### Set `X` (predictor) and `y` (target) variables 
X = df['title']
y = df['subreddit']

#  
# ### Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=42,
                                                    stratify=y)


# Customize stop_words to include `onion` so that it doesn't appear
# in coefficients 

stop_words_onion = _stop_words.ENGLISH_STOP_WORDS
stop_words_onion = list(stop_words_onion)
stop_words_onion.append('onion')

# ### CountVectorizer & MultinomialNB
def CV_MultNb(alphaV,range):
    #Instantiate the classifier and vectorizer
    nb = MultinomialNB(alpha = alphaV)
    cvec = CountVectorizer(ngram_range= range)

    # Fit and transform the vectorizor
    cvec.fit(X_train)

    Xcvec_train = cvec.transform(X_train)
    Xcvec_test = cvec.transform(X_test)

    # Fit the classifier
    nb.fit(Xcvec_train,y_train)
    
    # Create the predictions for Y training data
    print("Score: ",nb.score(Xcvec_test, y_test))
    
    # Save the vectorizer
    vec_file = 'vectorizer.pkl'
    joblib.dump(cvec, open(vec_file, 'wb'))
    # Save the model    
    joblib.dump(nb, 'model_gs.pkl', compress = 1)
    print("Model CountVectorizer & MultinomialNB created")




# ### Model 4: TfidfVectorizer & MultinomialNB 
def tfIdf_MultNb(alphaV,maxdf,mindf,range): 
    #Instantiate the classifier and vectorizer
    nb = MultinomialNB(alpha = alphaV)
    cvec = TfidfVectorizer(max_df=maxdf,min_df=mindf,ngram_range= range)

    # Fit and transform the vectorizor
    cvec.fit(X_train)

    Xcvec_train = cvec.transform(X_train)
    Xcvec_test = cvec.transform(X_test)

    # Fit the classifier
    nb.fit(Xcvec_train,y_train)

    # Create the predictions for Y training data
    print("Score: ",nb.score(Xcvec_test, y_test))
    
    # Save the vectorizer
    vec_file = 'vectorizer.pkl'
    joblib.dump(cvec, open(vec_file, 'wb'))
    # Save the model    
    joblib.dump(nb, 'model_gs.pkl', compress = 1)
    print("Model TfidfVectorizer & MultinomialNB created")




# ### CountVectorizer & Logistic Regression
def CV_LR(ValC,stopword,range):

    #Instantiate the classifier and vectorizer
    lr = LogisticRegression(C = ValC, solver='liblinear')
    cvec = CountVectorizer(stop_words = stopword, ngram_range=range)

    # Fit and transform the vectorizor
    cvec.fit(X_train)

    Xcvec2_train = cvec.transform(X_train)
    Xcvec2_test = cvec.transform(X_test)

    # Fit the classifier
    lr.fit(Xcvec2_train,y_train)

    # Create the predictions for Y training data
    print("Score: ",lr.score(Xcvec2_test, y_test))
    
    # Save the vectorizer
    vec_file = 'vectorizer.pkl'
    joblib.dump(cvec, open(vec_file, 'wb'))
    # Save the model    
    joblib.dump(lr, 'model_gs.pkl', compress = 1)
    print("Model CountVectorizer & Logistic Regression created")


# ### TfidfVectorizer & Logistic Regression
def tfIdf_LR(ValC,maxdf,mindf,range): 
    #Instantiate the classifier and vectorizer
    lr = LogisticRegression(C = ValC, solver='liblinear')
    cvec = TfidfVectorizer(max_df=maxdf,min_df=mindf,ngram_range= range)

    # Fit and transform the vectorizor
    cvec.fit(X_train)

    Xcvec_train = cvec.transform(X_train)
    Xcvec_test = cvec.transform(X_test)

    # Fit the classifier
    lr.fit(Xcvec_train,y_train)

    # Create the predictions for Y training data
    print("Score: ",lr.score(Xcvec_test, y_test))
    
    # Save the vectorizer
    vec_file = 'vectorizer.pkl'
    joblib.dump(cvec, open(vec_file, 'wb'))
    # Save the model    
    joblib.dump(lr, 'model_gs.pkl', compress = 1)
    print("Model TfidfVectorizer & Logistic Regression created")



#
## # # # # # # # # # # # # # # # # create the model

#tfIdf_LR(1,0.75,5,(1,3))
#CV_LR(1,"None",(1,1))
#tfIdf_MultNb(0.1,0.75,4,(1,2))
#CV_MultNb(0.6,(1,3))


numMod=0
print("\n\nEntrez le numéro de modèle que vous voulez créer ? \n",
      "1 : CountVectorizer & MultinomialNB \n",
      "2 : TfidfVectorizer & MultinomialNB \n",
      "3 : CountVectorizer & Logistic Regression \n",
      "4 : TfidfVectorizer & Logistic Regression \n")

numMod = input("Modèle numéro: ")

if int(numMod)==1:
    CV_MultNb(0.6,(1,3))
elif int(numMod)==2:
    tfIdf_MultNb(0.1,0.75,4,(1,2))
elif int(numMod)==3:
    CV_LR(1,None,(1,1))
elif int(numMod)==4:
    tfIdf_LR(1,0.75,5,(1,3))
else :
    print("No model created")







''' a={'title': ["Mom With Arms Full Of Groceries Holds Baby By Scruff Of Neck With Mouth"]}
df_a1=pd.DataFrame(data=a)
clean_data(df_a1)
df_a1 = df_a1.reset_index(drop=True)
a1_titles = df_a1['title']
cv_a1= CountVectorizer(stop_words = 'english')
a1_cvec=cv_a1.fit_transform(a1_titles)
a1_cvec_df = pd.DataFrame(a1_cvec.toarray(),
                   columns=cv_a1.get_feature_names_out())

X_df_a1 = df_a1['title']
Xcvec_df_a1 = cvec.transform(X_df_a1)
preds1 = nb.predict(Xcvec_df_a1)
print("-------------------")
print(preds1)


gs1=joblib.load("model_gs.pkl")
preds11 = gs1.predict(Xcvec_df_a1)
print("-------------------")
print(preds11) '''