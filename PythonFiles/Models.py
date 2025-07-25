import numpy as np
import heapq
from math import sqrt
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt
import statistics


# In[3]:


class LogisticRegression:
    def __init__(self):
        self.learning_rate = 0.01
        self.num_iterations = 1000
        self.weights = None
        self.bias = None
        self.iter = 0

    def set_learning_rate(self, val):
        self.learning_rate = val

    def sigmoid(self, z):
        # Sigmoid function to convert values to probabilities between 0 and 1
        return 1 / (1 + np.exp(-z)) #sigmoid(z) = 1 / ( 1 + e( - z ) )

    def fit(self, data, labels): #training the logistic regression model
        num_samples, num_features = data.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        converge=0.0001
        converged = False
        cost1 = 1
        count = 0
        self.iter = 0
        
        while not converged and count<self.num_iterations:
        # Gradient descent
        #for i in range(self.num_iterations):
            #Hypothesis Function
            linear_model = np.dot(data, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Compute gradients
            #∂J/∂w = (1/m) * Σ[(h(x) - y) * x] , ∂J/∂b = (1/m) * Σ(h(x) - y)

            dw = (1/num_samples) * np.dot((predictions - labels),data)
            db = (1/num_samples) * np.sum(predictions - labels)

            # Update the parameters in the opposite direction of the gradient
            #w := w - α * ∂J/∂w  ,  b := b - α * ∂J/∂b
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            cost = 0   
            self.iter += 1
            probabilities = self.sigmoid(linear_model)
            cost = -1/num_samples * (np.dot(1 - labels, np.log(1 - probabilities + converge)) + np.dot(labels, np.log(probabilities + converge)))
            
            if abs(cost1-cost)<=converge:
                converged = True
            cost1=cost
            count+=1
            
        return 
        
    def k_fold (self, data, labels, k):
        accuracies = []
        iterations = []
        index_length = len(data)//k
        counter = 0
        
        for i in range(k):
            if i == k-1:
                data_testing_set = data[counter:]
                data_training_set = data[:counter]
                label_testing_set = labels[counter:]
                label_training_set = labels[:counter]
                
            else:
                data_testing_set = data[counter:index_length+counter]
                label_testing_set = labels[counter:index_length+counter]
                if counter == 0:
                    data_training_set = data[index_length+1:]
                    label_training_set = labels[index_length+1:]
                else:
                    data_training_set = np.concatenate((data[0:counter] , data[index_length+counter+1:]))
                    label_training_set = np.concatenate((labels[0:counter] , labels[index_length+counter+1:]))
                    
            counter+=index_length
            self.fit(np.array(data_training_set),np.array(label_training_set))
            labels_pred = self.predict(np.array(data_testing_set))
            accuracy = self.evaluate_acc(np.array(label_testing_set),np.array(labels_pred))
            accuracies.append(accuracy)
            iterations.append(self.iter)
        
        return accuracies, iterations
            
    def predict(self, data):
        #Hypothesis Function
        linear_model = np.dot(data, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return [1 if p >= 0.5 else 0 for p in predictions]

    def evaluate_acc(self, label_true, label_pred):
        correct = np.sum(label_true == label_pred)
        total = len(label_true)
        return correct / total


# In[4]:


# 1) a new data point is input that we need to classify
# 2) check the classification of the k nearest elements
# 3) assunming we have 2 unique classifications (a,b). we take the classification of the dominant group
# 4) if a tie exists take the class with the shortest distance from 

#to calculate distance we can use the Euclidean distance formula sqrt(sum i to N (x1_i — x2_i)²)


class kNN:
    def __init__(self, k, dist_metric="euclidean"):     
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        initialize kNN model
        ===================================================================================
        PARAMETERS:
        ===================================================================================
        * self (kNN): 
        ----------------------------------------
        kNN model to define k values, training data, and distance metric
        ----------------------------------------
        * k (int):
        ----------------------------------------
        integer representing number of neighbours to compare to
        ----------------------------------------
        * dist_metric (string):
        ----------------------------------------
        string representing distance metric formula to follow
        ===================================================================================
        '''
        self.k = k #num of neighbours
        self.dist_metric = dist_metric #equation to calculate distance with
        self.train_data = None #initialize using fit method
        self.train_labels = None
        
    def fit(self, data, labels):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        set train_data and train_labelsby loading in Train data used to compare new data
        ===================================================================================
        PARAMETERS:
        ===================================================================================
        * data[] (NumPy Array): 
        ----------------------------------------
        list of data with labels seperated
        ----------------------------------------
        * labels (NumPy Array):
        ----------------------------------------
        list of labels with data removed
        ===================================================================================
        '''
        self.train_data = data
        self.train_labels = labels

    def predict(self, new_data):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        given new data, compare its items to the k closest elements of training data based 
        on a set distance metric and predict the datas classification.
        ===================================================================================
        PARAMETERS:
        ===================================================================================
        * self (kNN): 
        ----------------------------------------
        kNN model with predefined k values, training data, and distance metric
        ----------------------------------------
        * new_data (NumPy Array):
        ----------------------------------------
        Array of new data to predict classifications for
        ===================================================================================
        RETURNS:
        ===================================================================================
        * predictions (List):
        ----------------------------------------
        list of labels for each item in new_data
        ===================================================================================
        '''
        predictions = []#return array of predicted classifications, for each row in new_data
        for new_row in new_data:
            # calculate distances between new data and training data                   
            k_neighbours = self.__neighbours(new_row) #determine the k nearest neighbours using preffered distance metric
            classifications = []#for the given neighbors check their label
            distances = [] #for tiebreak if need be
            for result in k_neighbours:
                #print(f"new = {new_row}: train = {result[2]}")
                i = result[1]#results formatted [row, index of row], so take the index to find the associated label
                dist = result[0]
                classifications.append(self.train_labels[i])#add label at index i
                distances.append(dist)

            
            #check for ties in classifications here
            # thinking is use multimode to check for classification ties, to know then we need to check lowest distance 
            classifications_mode = statistics.multimode(classifications)
            #print(classifications_mode)
            #print(distances)

            if len(classifications_mode) > 1: #i.e we have a tie
                #go over distances, and get the lowest average and append that to predictions
                #print(classifications_mode)
                #make an array of size len(classifications) to store distances
                distances_mode = [0.0] * len(classifications_mode)
                #print(distances_mode, len(distances_mode), len(classifications_mode))
                #if label from classifications is in modes, add the distance to the appropriate index in the distances array
                classification_index = 0
                for classification in classifications:
                    if classification in classifications_mode: #if the classification is part of the multimode
                        #get the proper index for the distance array for that 
                        distance_index = classifications_mode.index(classification)
                        #print(distance_index)
                        #print(distances_mode)
                        distances_mode[distance_index] += distances[classification_index] #increase the correct distance by the distance of the classification

                    classification_index += 1

                #print(distances)
                #print(distances_mode)
                #choose the minimum distance label and append to predictions
                min_distance = min(distances_mode)

                min_distance_index = distances_mode.index(min_distance)
                #print(min_distance_index)
                min_classification = classifications_mode[min_distance_index]

                #print("The label with the minimum distance to neighbors is:", min_classification, "with distance:", min_distance)

                predictions.append(min_classification)

            else:
                predictions.append(str(max(classifications, key=classifications.count))) #from collections import Counter
        
        return predictions
    
    def evaluate_acc(self,predictions, test_labels, display=True):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        Compare model predictions to actual values and print success rate
        ===================================================================================
        '''
        total = len(predictions)
        hits = 0

        for i in range(total):
            if predictions[i] == test_labels[i]:
                hits += 1
#             print(f"guess:{predictions[i]}|answer:{test_labels[i]}")
        percentage = round(100*hits/total,2)
        if display:
            print(f"Success Rate: %{percentage}")

        return percentage
    
    def setK(self,k):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        Compare model predictions to actual values and print success rate
        ===================================================================================
        '''
        self.k = k
    
    def __calc_distance(self,newRow, trainRow):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        Private Function used in self.__neighbours(). Given a row from new data, calculate 
        the distance based on a set metric from a row in Train data
        ===================================================================================
        PARAMETERS:
        ===================================================================================
        * self (kNN): 
        ----------------------------------------
        kNN model with predefined k values, training data, and distance metric
        ----------------------------------------
        * newRow[] (List of data points (float/int)):
        ----------------------------------------
        data row to compare distance from train row data 
        ----------------------------------------
        * trainRow[] (List of data points (float/int)):
        ----------------------------------------
        data row to compare distance with test data
        ===================================================================================
        RETURNS:
        ===================================================================================
        * distance (float):
        ----------------------------------------
        float distance between to rows of data
        ===================================================================================
        '''
        distance = 0
        if self.dist_metric == "euclidean":
            for i in range(len(newRow)):
                squared = pow(newRow[i] - trainRow[i],2)
                distance += squared
            distance = sqrt(distance)
        return(distance)

    def __neighbours(self, new_row):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        private function used in self.predict(). Given a row from new data, return k number
        of neigbours based on distance
        ===================================================================================
        PARAMETERS:
        ===================================================================================
        * self (kNN): 
        ----------------------------------------
        kNN model with predefined k values, training data, and distance metric
        ----------------------------------------
        * newRow[] (List of data points (float/int)):
        ----------------------------------------
        data row to compare distance from train row data 
        ----------------------------------------
        * trainRow[] (List of data points (float/int)):
        ----------------------------------------
        data row to compare distance with test data
        ===================================================================================
        RETURNS:
        ===================================================================================
        * k_neighbours (List):
        ----------------------------------------
        list of k closest neighbours based on distance metric
        ===================================================================================
        '''
        distances = []#heap array
        #for every row of data
        for index in range(len(self.train_data)):#use index to return that value later
            train_row = self.train_data[index]#current row of train data
            dist = self.__calc_distance(new_row, train_row)#calculate distance between new row and train data row
            heapq.heappush(distances, [-dist, index, list(train_row)])#make negative value temporarily to assure we have smallest values 
            if len(distances) > self.k:#past k values remove largest from heap
                heapq.heappop(distances)
        
        k_neighbours = [[-dist, index, train_row] for dist, index, train_row in sorted(distances)]#make positive values, only 5 smallest remain

        return k_neighbours
    
    def __foldSplit(self, data, k_folds):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        split data into folds used for kFoldCross function 
        ===================================================================================
        PARAMETERS:
        ===================================================================================
        * self (kNN): [not used]
        ----------------------------------------
        kNN model with predefined k values, training data, and distance metric
        ----------------------------------------
        * data (numpy array):
        ----------------------------------------
        data to split into training and test data based on k value
        ----------------------------------------
        * k_folds (int):
        ----------------------------------------
        number of folds to split data into       
        ===================================================================================
        '''  
        data_size = len(data)
        fold_size = data_size//k_folds
        data_split = []
        
        for i in range(k_folds):
#             print(i)
            start_i = i * fold_size
            end_i = (i + 1) * fold_size if i < k_folds - 1 else len(data)

            # Extract the test data for this fold
            test_data = data[start_i:end_i]
#             print(test_data)

            # Extract the training data for this fold
            train_data = np.concatenate([data[:start_i], data[end_i:]], axis=0)
            data_split.append((train_data, test_data))        
        return data_split
    
    def kFoldCross(self, data, k_folds, display=True):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        A script to run an evaluation of the same data with different training values based 
        on k Fold cross validatio 
        ===================================================================================
        PARAMETERS:
        ===================================================================================
        * self (kNN): [not used]
        ----------------------------------------
        kNN model with predefined k values, training data, and distance metric
        ----------------------------------------
        * data (numpy array):
        ----------------------------------------
        data to split into training and test data based on k value
        ----------------------------------------
        * k_folds (int):
        ----------------------------------------
        number of folds to split data into       
        ===================================================================================
        '''    
        
        dataSplit = self.__foldSplit(data,k_folds)
        size = len(dataSplit)
        accAvg = 0
        
        for i in range(size):
            if display:
                print("=================================")
                print(f"Training with Fold {i+1}")
                print("---------------------------------")

            train_data_list,test_data_list = dataSplit[i]
            train_data, train_labels = self.__seperateLabels(train_data_list)
            test_data, test_labels = self.__seperateLabels(test_data_list)
            
            self.fit(train_data, train_labels)
            predictions = self.predict(test_data)
            accAvg += self.evaluate_acc(predictions, test_labels, display)
        accAvg/=k_folds
        if display:
            print(f"Average Accuracy: %{accAvg}")
        return accAvg
            

        
    def __seperateLabels(self, data):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        seperate data into values and classifications (data,labels)
        ===================================================================================
        PARAMETERS:
        ===================================================================================
        * self (kNN): [not used]
        ----------------------------------------
        kNN model with predefined k values, training data, and distance metric
        ----------------------------------------
        * data (NumPy Array):
        ----------------------------------------
        data imported for assignment
        ===================================================================================
        RETURNS:
        ===================================================================================
        * values , labels
        ----------------------------------------
        a tuple of 2 numpy arrays, one of data and one of labels
        ===================================================================================
        '''
        values = data[:,:-1].astype(float)#data only
        labels = data[:,-1]#classifications only
        
        return (values,labels)
        
    def testTrainSplit(self, data, testSplit=0.7):
        '''
        ===================================================================================
        DESCRIPTION: 
        ===================================================================================
        Used externally to split data into test and train data. probably remove later
        ===================================================================================
        PARAMETERS:
        ===================================================================================
        * self (kNN): [not used]
        ----------------------------------------
        kNN model with predefined k values, training data, and distance metric
        ----------------------------------------
        * data (NumPy Array):
        ----------------------------------------
        data imported for assignment
        ----------------------------------------
        * testSplit (float):
        ----------------------------------------
        ratio of data to be used for testing, default 70/30 split
        ===================================================================================
        RETURNS:
        ===================================================================================
        * dataSplit (tuple(List1,List2)):
        ----------------------------------------
        a tuple of 2 lists where list 1 contains training data and training labels,
        similarily list 2 contains test data and tes labels
        ===================================================================================
        '''
        split = int(len(data) * testSplit )
        #Split train data (70% standard)
        train_data, train_labels = self.__seperateLabels(data[:split]) 
        test_data, test_labels = self.__seperateLabels(data[split:])
        
        dataSplit = ([train_data, train_labels], [test_data, test_labels])
        
        return(dataSplit)

def main() -> None:
    """
    Train / evaluate LogisticRegression or kNN on one of the three datasets.

    Examples
    --------
    python Models.py --model logistic  --dataset adult
    python Models.py --model knn       --dataset mushroom --k 7
    """
    # ────────────────────────────────────────────────────────────────
    import argparse, contextlib, os, sys, numpy as np

    # Make Functions.py visible
    script_dir = os.path.dirname(os.path.abspath(__file__))      # …/PythonFiles
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from Functions import readFile, readFileKNN

    # 1 ─ CLI arguments
    p = argparse.ArgumentParser(description="Run LR or kNN on a UCI dataset")
    p.add_argument("--model",   choices=["logistic", "knn"], required=True)
    p.add_argument("--dataset",
              choices=["adult", "rice", "mushroom", "iris", "ionosphere"],
             default="adult")
    p.add_argument("--folds",   type=int,   default=5)
    p.add_argument("--lr",      type=float, default=0.01)   # LR only
    p.add_argument("--iters",   type=int,   default=1000)   # LR only
    p.add_argument("--k",       type=int,   default=5)      # kNN only
    args = p.parse_args()

    # 2 ─ locate the short file-name
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
    file_map = {
    "adult":      "adult.data",
    "rice":       "Rice_Cammeo_Osmancik.arff.txt",
    "mushroom":   "agaricus-lepiota.data",
    "iris":       "iris.data",
    "ionosphere": "ionosphere.data",
 }
    base_name = file_map[args.dataset]

    # helper: temporary chdir
    @contextlib.contextmanager
    def cd(path):
        prev = os.getcwd()
        os.chdir(path)
        try:    yield
        finally: os.chdir(prev)

    # 3 ─ load the data
    with cd(data_dir):
        if args.model == "logistic":
            X, y = readFile(base_name)                 # → numeric already
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=int)
        else:                                          # kNN path
            combined = readFileKNN(base_name)          # may contain strings

    # 4 ─ if we are in the kNN branch, encode features to numbers
    if args.model == "knn":

        def encode_features(mat: np.ndarray) -> np.ndarray:
            """
            Replace every non-numeric column in `mat` (all rows, except
            the last column = label) with 0,1,2,… ordinal tokens.
            Labels stay untouched.
            Returns an object-dtype array suitable for __seperateLabels().
            """
            mat = mat.astype(object)                   # keep heterogeneity
            n_cols = mat.shape[1] - 1                  # skip label col

            for col in range(n_cols):
                try:
                    mat[:, col].astype(float)          # already numeric?
                except ValueError:
                    uniq, inv = np.unique(mat[:, col], return_inverse=True)
                    mat[:, col] = inv                  # 0…N encoding
            return mat

        combined = encode_features(combined)

    # 5 ─ run the chosen model
    if args.model == "logistic":
        lr_model = LogisticRegression()
        lr_model.set_learning_rate(args.lr)
        lr_model.num_iterations = args.iters

        accs, iters = lr_model.k_fold(X, y, args.folds)
        print("Fold accuracies:", np.round(accs, 4))
        print("Avg accuracy   :", round(np.mean(accs), 4))
        print("Iterations/fit :", iters)
    else:
        knn_model = kNN(k=args.k)
        avg_acc   = knn_model.kFoldCross(combined, args.folds, display=True)
        print("Avg accuracy   :", round(avg_acc, 4))


if __name__ == "__main__":
    main()
