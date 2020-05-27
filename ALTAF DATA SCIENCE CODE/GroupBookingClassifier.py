"""
Apply various classfiers.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

class Classify:
    def runOne(self, X_train, y_train, X_test, y_test, modelName):
        if modelName == 'LR':
            classifier = LogisticRegression(random_state = 42)
        if modelName == 'NB':
            classifier = GaussianNB()
        if modelName == 'DT':
            classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 42) # can be gini.
        if modelName == 'RF':
            classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
        if modelName == 'SVM':
            classifier = SVC(kernel = 'linear', random_state = 42)
        if modelName == 'XGB':
            #classifier = XGBClassifier(random_state = 42)
            classifier = XGBClassifier(random_state = 42, max_depth = 7, min_child_weight = 1)
        if modelName == 'KNN':
            classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        if modelName == 'ANN':
            self.runANN(X_train, y_train, X_test, y_test, modelName)
        #train the model. 
        classifier.fit(X_train, y_train)  
        # predictions on train.
        print('Accuracy on training set -----------')
        train_scores = self.crossValidate(classifier, X_train, y_train)
        test_scores = None
        #print('Accuracy on validation set -----------')
        #test_scores = self.crossValidate(classifier, X_test, y_test)
        # predictions on test.
        y_pred = classifier.predict(X_test)        
        return list((classifier, y_pred, train_scores, test_scores)) 
    
    def crossValidate(self, model, X_test, y_test):   
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_test, y_test, cv=10)
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return scores
    
    def logisticRegression(self, X_train, y_train, X_test, y_test):
        # training.
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)        
        # predictions.
        y_pred = classifier.predict(X_test)
        # validation
        scores = self.crossValidate(classifier, X_test, y_test)
        return list((y_pred, scores)) 
       
    def runANN(self, X_train, y_train, X_test, y_test):                
        #import keras
        from keras.models import Sequential
        from keras.layers import Dense
        
        classifier = Sequential() # initialize.
        classifier.add(Dense(activation="relu", input_dim=X_train.shape[1], units=6, kernel_initializer="uniform")) # first layer
        classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform")) # hidden layer.
        classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = "uniform")) # output layer.
        
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # compile.
        
        # Fit ANN to training set.
        classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
        
        # predict.
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)        
        return classifier, y_pred
        
#        # confusion matrix.
#        from sklearn.metrics import confusion_matrix
#        cm = confusion_matrix(y_test, y_pred)
#        (cm[0][0] + cm[1][1]) / 2000
   
    
    
            
            
            
        
       
