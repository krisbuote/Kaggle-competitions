import pandas as pd
from sklearn import preprocessing

'''
Predict if a passenger survived or died on the Titanic!
Training data includes PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked


Decisions: Remove Name because Sex and Age are relevant, name isn't.
Lot's of cabin entries are NaN, too many missing to be useful
Ticket Number is messy and irrelevant
ID # also irrelvant to prediction, but necessary for identification.
'''

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

print(df_train.head()) # Check out the data

# Create a function to see how many missing entries are in each feature
def check_null(df_train, df_test):
    print("Training Data:")
    print(pd.isnull(df_train).sum())
    print("\nTest Data:" )
    print(pd.isnull(df_test).sum())

check_null(df_train, df_test)


# names or ticket numbers, not useful. Too many missing Cabin numbers.
df_train = df_train.drop(columns=['Name', 'Ticket', 'Cabin'])
df_test = df_test.drop(columns=['Name', 'Ticket', 'Cabin'])

# Data is now: ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
print(list(df_train))

# Data is missing Age entries in both, 2 Embarked entries in Training, and 1 Fare entry in Test. Need to fill in:
df_train["Age"].fillna(value=df_train["Age"].median(), inplace = True)
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
df_train['Embarked'].fillna('S', inplace=True)
df_test['Fare'].fillna(df_train['Fare'].mean(), inplace=True)

check_null(df_train, df_test) # Check for null again -> No NaN Values !

'''We need to transform categorical data (Sex = Male, Female)  into numerical: (Sex = 0, 2)'''
# Find the locations where the rows contain 'male', and the column is 'Sex'. male -> 0
df_train.loc[df_train['Sex'] == 'male', 'Sex'] = 0
df_train.loc[df_train['Sex'] == 'female', 'Sex'] = 2

df_test.loc[df_test['Sex'] == 'male', 'Sex'] = 0
df_test.loc[df_test['Sex'] == 'female', 'Sex'] = 2

# Embark locations S, Q, C -> 0, 1 ,2
df_train.loc[df_train['Embarked'] == 'S', 'Embarked'] = 0
df_train.loc[df_train['Embarked'] == 'Q', 'Embarked'] = 1
df_train.loc[df_train['Embarked'] == 'C', 'Embarked'] = 2

df_test.loc[df_test['Embarked'] == 'S', 'Embarked'] = 0
df_test.loc[df_test['Embarked'] == 'Q', 'Embarked'] = 1
df_test.loc[df_test['Embarked'] == 'C', 'Embarked'] = 2

print(df_train.head()) # Check out the Embarked and Sex columns

''' Scale the ages to be on range 0,2'''
age_scaler = preprocessing.MinMaxScaler(feature_range=(0,2))

# Get arrays
age_train = df_train['Age'].values.reshape(-1,1)
age_test = df_test['Age'].values.reshape(-1,1)

# Scale
age_train_scaled = age_scaler.fit_transform(age_train)
age_test_scaled = age_scaler.transform(age_test)

# Update dataframes
df_train['Age'] = age_train_scaled
df_test['Age'] = age_test_scaled

''' Scale the fares to be on range 0, 2'''
fare_scaler = preprocessing.MinMaxScaler(feature_range=(0,2))

fare_train, fare_test = df_train['Fare'].values.reshape(-1,1), df_test['Fare'].values.reshape(-1,1)

fare_train_scaled = fare_scaler.fit_transform(fare_train)
fare_test_scaled = fare_scaler.transform(fare_test)

df_train['Fare'] = fare_train_scaled
df_test['Fare'] = fare_test_scaled

print(df_train.head())
print(df_test.head())

### Save the processed data to send to neural net ###
df_train.to_csv('./train_preprocessed.csv', index=False)
df_test.to_csv('./test_preprocessed.csv', index=False)



