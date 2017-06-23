from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt, arange, transpose

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt('Data/train.csv', delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt('Data/test.csv', delimiter=',', dtype='f8')[1:]

    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=4)
    rf.fit(train, target)
    predicted_probs = [x[1] for x in rf.predict_proba(test)]

    savetxt('Data/submission.csv', 
            transpose([arange(1, len(predicted_probs) + 1), predicted_probs]), 
            header='MoleculeId,PredictedProbability', 
            comments='',
            fmt='%i,%f')

if __name__=="__main__":
    main()

