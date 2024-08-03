# knn

```

    # Train the a K-NN classifier
    #===========================Fill This===========================#
    oKnnClassifier = KNeighborsClassifier(n_neighbors = K, metric = metricChoice) #<! Creating the object
    oKnnClassifier = oKnnClassifier.fit(mX, vY) #<! Training on the data
    #===============================================================#
    
    # Predict
    #===========================Fill This===========================#
    vYY = oKnnClassifier.predict(mX) #<! Prediction
    #===============================================================#

    # Score (Accuracy)
    #===========================Fill This===========================#
    scoreAcc = oKnnClassifier.score(mX,vY) #<! Score
    #===============================================================#


```