from keras.models import load_model
import pandas as pd
import numpy as np

# Load Test Data
test_df = pd.read_csv('./test_preprocessed.csv')
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
PassengerId = test_df['PassengerId'].values.reshape(len(test_df), 1)
PassengerId = np.squeeze(PassengerId)

X_test = test_df[features].values

# Load the Model
model = load_model('./model/titanic_model_dense6_dropout.h5')

# Make predictions
predictions = np.squeeze(model.predict(X_test))
prediction_ints = [int(round(prediction)) for prediction in predictions]
print(prediction_ints)

# Create dataframe with PassengerId and predictions (0, 1)
d = {'PassengerId':PassengerId, 'Survived':prediction_ints}
submission_df = pd.DataFrame(data=d)

submission_df.to_csv('./titanic_submission2.csv', index=False)