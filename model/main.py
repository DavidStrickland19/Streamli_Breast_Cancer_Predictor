import pandas as pd
import sklearn  
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle5 as pickle

#the data we pass here is our new clean data from down below
def create_model(data):

    #Here we will divide the data into predictors and predicted
    X = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis']

    #Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 42 )



    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    

    #test the model
    #accuracy_score takes two paramters
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ',accuracy_score(y_test, y_pred) )
    print('Classification report: \n', classification_report(y_test, y_pred))

    return model, scaler




    # Next we need to scale the data, or make it uniform, because the oroginal data has large differences in value
    # We will use standard scalar from sklearn to do this



def get_clean_data():
    data = pd.read_csv('data/data.csv')
    
    #we will drop column unamed 32 because it is all NAN's. Also need to drop the id column
    data = data.drop(['Unnamed: 32', 'id'], axis = 1)

    #We also need to drop the diagnosis column
    #We want Malicious (M) cells to be represented with "1" and Benine (B) will now be represented with 0

    #To do this, we will use a map function with key-value pairs. M  key, value  = 1
    data['diagnosis']  =  data['diagnosis'].map({'M' : 1, 'B' : 0})



    return data


def main():
    data = get_clean_data()
    
    model, scaler = create_model(data)

    #wb , b stands for binary, f stads for file
    with open('model/model.pkl', 'wb' ) as f:
        pickle.dump(model, f)

        #We want to save our model within "Model, so we add this in the path"
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    #Next we will create and train our model

if __name__ == '__main__' :
    main() 