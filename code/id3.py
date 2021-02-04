# Author: Sami Alperen Akgun
# Email: sami.alperen.akgun@gmail.com

############ Libraries ############
import os
import numpy as np 

############ Decision Tree Classes ############
class DT(object):
    def __init__(self, FeatureNum=-1,Children=[],LeafValue=None):
        self.FeatureNum = FeatureNum
        self.Children = Children
        self.LeafValue = LeafValue

class DT_cont(object):
    def __init__(self, FeatureNum=-1,Left=[],Right=[],ThresholdNum=None,LeafValue=None):
        self.FeatureNum = FeatureNum
        self.ThresholdNum = ThresholdNum
        self.Left = Left
        self.Right = Right
        self.LeafValue = LeafValue

############ Functions ############
def read_data_string(data_path, filename,feature_number):
    """
        This function reads the data from data_path/filename
        WARNING: This function assumes that features of data 
                is separated by commas in the file
        Input: data_path --> The full directory path of data
                filename --> name of the file (With extension)
                feature_number --> Total feature number in the data
        Output: X --> numpy car array that contains feature values
                size(X) --> sample size x feature number
                Y --> numpy array that contains labels
                size(Y) --> sample size x 1 
    """

    # Warning: Numpy indices start from zero
    X = np.genfromtxt(data_path + "/" + filename,delimiter=",",dtype="str")[:,0:feature_number]

    # Last column of datafile contains output labels
    Y = np.genfromtxt(data_path + "/" + filename,delimiter=",",dtype="str")[:,feature_number]
    Y = Y.reshape(Y.shape[0],1)

    return X,Y

def read_data(data_path, filename,feature_number):
    """
        This function reads the data from data_path/filename
        WARNING: This function assumes that features of data 
                is separated by commas in the file
        Input: data_path --> The full directory path of data
                filename --> name of the file (With extension)
                feature_number --> Total feature number in the data
        Output: X --> numpy array that contains feature values
                size(X) --> sample size x feature number
                Y --> numpy array that contains labels
                size(Y) --> sample size x 1 
    """

    # Warning: Numpy indices start from zero
    X = np.genfromtxt(data_path + "/" + filename,delimiter=",")[:,1:feature_number+1]

    # Last column of datafile contains output labels
    Y = np.genfromtxt(data_path + "/" + filename,delimiter=",")[:,0:1]
    Y = Y.reshape(Y.shape[0],1)

    return X,Y

def calculate_prob(input_data,feature_values):
    """
        This function calculates probability of each attribute 
        for a given input data
        Input: input_data --> ndimensional array or vector 
                
        For example: if size(input_data) = mxn 
        Output size --> total_feature_value_number xn --> probability of each attribute
        ,i.e. columns are composed of probability values

        For ex: size is 3xn since each attribute might have three distinct
        values for tic-tac-toe --> "x", "o" or "b" 
        output row 1 --> Prob of "x"
        output row 2 --> Prob of "o"
        output row 3 --> Prob of "b"
    """
    m = input_data.shape[0] #total sample size
    n = input_data.shape[1] #total feature(attribute) number
   
    total_feature_value_number = len(feature_values)
    probs = np.zeros((total_feature_value_number,n))
    #if all values are the same --> Assuming we have two labels only!!!
    if check_all_values_same(input_data,feature_values):
        for i in range(total_feature_value_number):
            if input_data[0] == feature_values[i]:
                probs[i,0] = 1.0
    else:   
        for i in range(n):
            current_attribute_data = input_data[:,i]
            total_prob = 0
            for j in range(total_feature_value_number):
                # Calculate how many times we have specific value for attribute
                total = current_attribute_data[current_attribute_data == feature_values[j]].shape[0]
                probs[j,i] = total/m
                total_prob = total_prob + probs[j,i]
            #print("Total prob: ", total_prob)

            # Total probability must be equal to 1.0
            epsilon = 0.001
            if (total_prob - 1.0) > epsilon:
                print("Total prob. is not equal to 1.0!")
    
    return probs


def entrophy(data_input,feature_values):
    """
        This function calculates entrophy for a given ndim np array
        It uses calculate_prob function to calculate probabilities first,
        then it uses these probabilities to calculate entrophy. 
        Input: data_input --> mxn , where m is total sample size
                                    and n is total feature number
        Output --> 1xn --> Entrophy for each attribute/feature
    """
    probabilities = calculate_prob(data_input,feature_values)
    # This is the number that contains how many different values 
    # a feature can take 
    # For example: for tic-tac-toe, each feature can be "x","o","b"
    # So total_feature_out_number --> 3
    total_feature_out_number = probabilities.shape[0]
    n = probabilities.shape[1]
    H = np.zeros((1,n))

    for i in range(n):
        sum_ent = 0
        for j in range(total_feature_out_number):
            if probabilities[j,i] == 0:
                sum_ent = 0
            else:
                sum_ent += probabilities[j,i] * np.log2(probabilities[j,i])
        H[:,i] = -1 * sum_ent

    return H

def divide_data(input_X,input_Y,Attribute,features_X,threshold=None):
    """
        This function divides input_X and input_Y arrays into subsets
        for a given attribute
        Inputs: input_X --> Input array contains values of features
                input_Y --> Input array contains values of output labels
                Attribute --> attribute used to divide the data, i.e. column
                number of input_X --> starts from zero!
                features_X --> contains feature values, such as "x", "o", "b"
                features_Y --> contains output labels such as "positive", "negative"
                threshold --> optional threshold parameter that can be used with the data
                              that have continious attribute values
                For example: wine.data dataset has 13 features that has continious values
                like 13.0, 12.3, 1.35 etc.
        Output: List of divided sets X and Y
                For example: for tic-tac-toe, output contains 3 separate np arrays X
                and 3 separate np arrays Y
                divided_X[0] --> divided output for "x"
                divided_X[1] --> divided output for "o"
                divided_X[2] --> divided output for "b"
                divided_Y[0] --> corresponding labels for "x"
    """
    selected_column = input_X[:,Attribute]
    divided_X = []
    divided_Y = []
    if threshold == None: # --> means given data has discrete feature values like tic-tac
        total_X_feature_number = len(features_X)
        
        for i in range(total_X_feature_number):
            divided_X.append(input_X[selected_column == features_X[i]])
            divided_Y.append(input_Y[selected_column == features_X[i]])
    else: # --> means given data has continious feature values like wine dataset
        # 2 feature values in total --> values less than threshold and higher than threshold
        total_X_feature_number = 2 
        
        divided_X.append(input_X[selected_column <= threshold])
        divided_X.append(input_X[selected_column > threshold])
        divided_Y.append(input_Y[selected_column <= threshold])
        divided_Y.append(input_Y[selected_column > threshold])

    #print("Divided X")
    #print(divided_X)
    #print("Divided Y")
    #print(divided_Y)

    return divided_X, divided_Y

def return_all_thresholds(input_X,input_Y,Attribute):
    """
        This function returns all potential optimum threshold points
        using given input_X and input_Y data for a given Attribute
        For example, let's say Attribute = 0 
        input_X = [[12,13,14],  corresponding input_Y = [1,
                   [13,16,17],                           1,
                   [16,16,17],                           2,
                   [17,16,17],                           3,
                   [19,16,17],                           3,
                   [21,16,17]]                           3]
        Output will be --> [(13+16)/2, (16+17)/2] = [14.5, 16.5] 
        since optimal points can not be between values of same class!
        Before checking adjacent values of given attribute and corresponding
        class label, it sorts the input_X and input_Y.
    """
    selected_column = input_X[:,Attribute]
    sorted_selected_column_indices = np.argsort(selected_column)
    sorted_X = selected_column[sorted_selected_column_indices]
    sorted_Y = input_Y[sorted_selected_column_indices]
    
    #print("Unsorted X")
    #print(selected_column)
    #print("unsorted Y")
    #print(input_Y)
    #print("Sorted X")
    #print(sorted_X)
    #print("Sorted Y")
    #print(sorted_Y)

    thresholds = []
    total_sample_number = len(sorted_X)
    for i in range(total_sample_number-1):
        # If two adjacent class labels are not the same --> possible threshold
        if sorted_Y[i] != sorted_Y[i+1]:
            # let's say sorted_X[i] = 12 and and sorted_X[i+1] = 14
            # calculated threshold --> (12+14)/2 = 13 ( mean of two values) 
            calculated_threshold = (sorted_X[i]+sorted_X[i+1])/2  #mean
            thresholds.append(calculated_threshold)

    
    #print("Threshold:")
    #print(thresholds)
    return thresholds


def information_gain(input_set_Y, features_Y, entrophy_Y):
    """
        This function calculates information gain I(X,Y)
        Inputs: entrophy_Y --> entrophy(Y) = H(Y)
                input_set_X --> input array that contains features and their values
                input_set_Y --> input_array that contains class labels
                features_Y --> list contains class label values
        Output: Information Gain I --> scalar
                I(X,Y) = H(Y) - H(Y|X)
        For example: input_set_Y here is actually contains Y|X. 
                        In tic-tac-toe, input_set_X is the divided set for
                        let's say attribute 1 and value "x" and input_y
                        is the output labels for the data that has attribute 1 = "x"
    """
   
    if len(input_set_Y) == 0:
        I = 0
    else:
        I = entrophy_Y - entrophy(input_set_Y,features_Y)
  
    return I

def gain_ratio(input_set_Y, features_Y, I):
    """
        This function calculates gain ratio:
        Gain ratio = I(X,Y) / Split information
        where split information = H(Y|X)
        Inputs: I --> Information gain I(X,Y)
                input_set_Y --> input_array that contains class labels
                features_Y --> list contains class label values
        Output: Gain ratio (GR) --> scalar
                I(X,Y) = H(Y) - H(Y|X)
        For example: input_set_Y here is actually contains Y|X. 
                        In tic-tac-toe, input_set_X is the divided set for
                        let's say attribute 1 and value "x" and input_y
                        is the output labels for the data that has attribute 1 = "x"
    """
   
    if len(input_set_Y) == 0:
        GR = 0
    else:
        GR = I / entrophy(input_set_Y,features_Y)

    return GR

def most_common_value(input_data,features):
    """
        This function checks the input_data array and returns
        the most common value in the array
        For example, if input_data = Y for tic-tac-toe case,
        this function will either return "positive" or "negative"
        Input --> features --> it is a list consists of feature values 
                               like "positive", "negative" for tic-tac-toe
        Output --> The most common value --> can be anything such as string, float etc.
    """

    total_feature_value_number = len(features)

    most_common_feature_size = 0
    for i in range(total_feature_value_number):
        feature_size = input_data[input_data==features[i]].shape[0]
        if feature_size > most_common_feature_size:
            most_common_feature_size = feature_size
            most_common_feature = features[i]

    return most_common_feature

def check_all_values_same(input_data,feature_values):
    """
        This function checks whether all the labels in a given
        input array is the same or not
        Input --> feature values contains all possible labels
                  Input array --> numpy array 
        Output --> Boolean: True --> if all elements are the same
                   output_value --> the value in input_data
        For example, if input is [positive,positive,positive]
        and feature_values = [positive,negative]
                    Boolen: True --> and output_value = "positive"
        If input doesn't have all the same values --> output_value --> 0
    """
    total_value_number = len(feature_values)
    total_sample_size = input_data.shape[0] 
    sizes = [] # this contains length of all labels with the same class
    for i in range(total_value_number):
        sizes.append(input_data[input_data == feature_values[i]].shape[0])
    #print("input_data")
    #print(input_data)
    #print("sizeS:")
    #print(sizes)
    # If sizes contains total_sample_size, this means that all labels of input_data
    # are the same
    # For example, they are all "positive" or all "negative" in tic-tac-toe case
    #print("Input_data y inside check all values blalbalbla")
    #print(input_data)
    if total_sample_size in sizes:
        #print("All labels of input data are the same!")
        output_boolean = True
    else:
        output_boolean = False
    
    #if output_boolean == True:
    #    output_value = input_data[0]
    #else:
    #    output_value = 0

    return output_boolean#,output_value


def BuildingDT(input_X,input_Y,feature_X_values,feature_Y_values,Attribute,mc_label):
    """
        This function creates an ID3 decision tree
        Inputs: input_X --> np array contains data with features/attributes
                input_Y --> np array contains output labels 
                feature_X_values --> list composed of possible values of features
                feature_Y_values --> list composed of possible values of out labels
                Attribute --> selected attribute/feature number(s) (column index)
                              It is a list composed of column indices
                mc_label --> most common label in the original data Y
        Output: Decision Tree DT (class)
    """
    # If there is no output label
    if input_Y.shape[0] == 0:
        return DT(LeafValue=mc_label)
    
    children = [] #children nodes from the current selected node 
    mcv = most_common_value(input_Y,feature_Y_values) # most common value of given input Y
    
    # If all labels of input_Y is the same or Attribute is empty
    all_values_same_boolean = check_all_values_same(input_Y,feature_Y_values)   
    if all_values_same_boolean or Attribute == []:
        return DT(LeafValue=mcv)
    else:
        best_IG = 0
        best_set_X = []
        best_set_Y = []
        best_Attribute = 0
        current_entrophy = entrophy(input_Y,feature_Y_values)
        for index in Attribute:
            # index is the column index --> decides which attribute to use
            divided_X, divided_Y = divide_data(input_X,input_Y,index,feature_X_values)
            
            for j in range(len(divided_Y)): #iterate through each possible feature value
                # For tic-tac-toe --> divided_Y contains subsets for "x", "o" and "b"
                current_IG = information_gain(divided_Y[j],feature_Y_values,current_entrophy)
                if current_IG >= best_IG:
                    best_IG = current_IG
                    best_Attribute = index
                    best_feature_ind = j
                    

        # Now best attribute is selected --> divide data using that attribute
        # best_Attribute --> gives the column number of best attribute
        # best_feature_ind --> gives which feature value has the most IG, i.e. "x", "o" or "b" in tic-tac
        best_set_X, best_set_Y = divide_data(input_X,input_Y,best_Attribute,feature_X_values)
     
        # Remove the current attribute from the attribute list to
        # create remaning attributes --> no need to use same attribute over and
        # over again! --> If it is needed, it is gonna show up in the recursion anyway
        remaning_Attributes = []
        for index in Attribute:
            if index != best_Attribute:
                remaning_Attributes.append(index)

        
        for i in range(len(best_set_X)):
            children.append( BuildingDT(best_set_X[i],best_set_Y[i],feature_X_values,feature_Y_values,  
                                                remaning_Attributes,mcv) )

        return DT(FeatureNum=best_Attribute,Children=children)
        

def BuildingDT_GR(input_X,input_Y,feature_X_values,feature_Y_values,Attribute,mc_label):
    """
        This function creates an ID3 decision tree based on maximizing gain ratio (GR)
        Inputs: input_X --> np array contains data with features/attributes
                input_Y --> np array contains output labels 
                feature_X_values --> list composed of possible values of features
                feature_Y_values --> list composed of possible values of out labels
                Attribute --> selected attribute/feature number(s) (column index)
                              It is a list composed of column indices
                mc_label --> most common label in the original data Y
        Output: Decision Tree DT (class)
    """
    # If there is no output label
    if input_Y.shape[0] == 0:
        return DT(LeafValue=mc_label)
    
    children = [] #children nodes from the current selected node 
    mcv = most_common_value(input_Y,feature_Y_values) # most common value of given input Y
    
    # If all labels of input_Y is the same or Attribute is empty
    all_values_same_boolean = check_all_values_same(input_Y,feature_Y_values)   
    if all_values_same_boolean or Attribute == []:
        return DT(LeafValue=mcv)
    else:
        best_GR = 0 #best gain ratio
        best_set_X = []
        best_set_Y = []
        best_Attribute = 0
        current_entrophy = entrophy(input_Y,feature_Y_values)
        for index in Attribute:
            # index is the column index --> decides which attribute to use
            divided_X, divided_Y = divide_data(input_X,input_Y,index,feature_X_values)
            
            for j in range(len(divided_Y)): #iterate through each possible feature value
                # For tic-tac-toe --> divided_Y contains subsets for "x", "o" and "b"
                current_IG = information_gain(divided_Y[j],feature_Y_values,current_entrophy)
                current_GR = gain_ratio(divided_Y[j],feature_Y_values,current_IG)
                if current_GR >= best_GR:
                    best_GR = current_GR
                    best_Attribute = index
                    best_feature_ind = j
                    

        # Now best attribute is selected --> divide data using that attribute
        # best_Attribute --> gives the column number of best attribute
        # best_feature_ind --> gives which feature value has the most IG, i.e. "x", "o" or "b" in tic-tac
        best_set_X, best_set_Y = divide_data(input_X,input_Y,best_Attribute,feature_X_values)
     
        # Remove the current attribute from the attribute list to
        # create remaning attributes --> no need to use same attribute over and
        # over again! --> If it is needed, it is gonna show up in the recursion anyway
        remaning_Attributes = []
        for index in Attribute:
            if index != best_Attribute:
                remaning_Attributes.append(index)

        
        for i in range(len(best_set_X)):
            children.append( BuildingDT(best_set_X[i],best_set_Y[i],feature_X_values,feature_Y_values,  
                                                remaning_Attributes,mcv) )

        return DT(FeatureNum=best_Attribute,Children=children)

def BuildingDT_cont(input_X,input_Y,feature_X_values,feature_Y_values,Attribute,mc_label):
    """
        This function creates an ID3 decision tree
        Inputs: input_X --> np array contains data with features/attributes
                input_Y --> np array contains output labels 
                feature_X_values --> list composed of possible values of features
                feature_Y_values --> list composed of possible values of out labels
                Attribute --> selected attribute/feature number(s) (column index)
                              It is a list composed of column indices
                mc_label --> most common label in the original data Y
        Output: Decision Tree DT (class)
    """
    # If there is no output label
    if input_Y.shape[0] == 0:
        return DT_cont(LeafValue=mc_label)
    
    left = [] # contains values less or equal than threshold  
    right = [] # higher than threshold
    mcv = most_common_value(input_Y,feature_Y_values) # most common value of given input Y
    
    # If all labels of input_Y is the same or Attribute is empty
    all_values_same_boolean = check_all_values_same(input_Y,feature_Y_values)   
    if all_values_same_boolean or Attribute == []:
        return DT_cont(LeafValue=mcv)
    else:
        best_IG = 0
        best_set_X = []
        best_set_Y = []
        best_Attribute = 0
        best_threshold = 0.0
        current_entrophy = entrophy(input_Y,feature_Y_values)
        for index in Attribute:
            # index is the column index --> decides which attribute to use
            all_thresholds = return_all_thresholds(input_X,input_Y,index)
            for threshold in all_thresholds:
                # Get divided data for each threshold for a given attribute = index
                divided_X, divided_Y = divide_data(input_X,input_Y,index,feature_X_values,threshold)

                for j in range(len(divided_Y)): #iterate through each possible feature value
                    # For wine.data --> divided_Y contains subsets for lower then threshold
                    # and higher than the threshold
                    current_IG = information_gain(divided_Y[j],feature_Y_values,current_entrophy)
                    if current_IG >= best_IG:
                        best_IG = current_IG
                        best_Attribute = index
                        best_feature_ind = j
                        best_threshold = threshold
                    

        # Now best attribute and threshold are selected --> divide data using that attribute and thresh
        # best_Attribute --> gives the column number of best attribute
        # best threshold --> gives the best threshold value which gives max IG
        # best_feature_ind --> gives which feature value has the most IG, i.e. "x", "o" or "b" in tic-tac
        best_set_X, best_set_Y = divide_data(input_X,input_Y,best_Attribute,feature_X_values,best_threshold)
     
        # Remove the current attribute from the attribute list to
        # create remaning attributes --> no need to use same attribute over and
        # over again! --> If it is needed, it is gonna show up in the recursion anyway
        remaning_Attributes = []
        for index in Attribute:
            if index != best_Attribute:
                remaning_Attributes.append(index)

        # Left contains values less or equal than threshold
        # Right contains values more than threshold
        # For more detail, check divide data function
        
        return DT_cont(Left=BuildingDT_cont(best_set_X[0],best_set_Y[0],feature_X_values,feature_Y_values,
                        remaning_Attributes,mcv),
                       Right=BuildingDT_cont(best_set_X[1],best_set_Y[1],feature_X_values,feature_Y_values, 
                        remaning_Attributes,mcv),
                       FeatureNum=best_Attribute,ThresholdNum=best_threshold 
                       )
                    
def BuildingDT_cont_GR(input_X,input_Y,feature_X_values,feature_Y_values,Attribute,mc_label):
    """
        This function creates an ID3 decision tree based on maximizing gain ratio (GR)
        Inputs: input_X --> np array contains data with features/attributes
                input_Y --> np array contains output labels 
                feature_X_values --> list composed of possible values of features
                feature_Y_values --> list composed of possible values of out labels
                Attribute --> selected attribute/feature number(s) (column index)
                              It is a list composed of column indices
                mc_label --> most common label in the original data Y
        Output: Decision Tree DT (class)
    """
    # If there is no output label
    if input_Y.shape[0] == 0:
        return DT_cont(LeafValue=mc_label)
    
    left = [] # contains values less or equal than threshold  
    right = [] # higher than threshold
    mcv = most_common_value(input_Y,feature_Y_values) # most common value of given input Y
    
    # If all labels of input_Y is the same or Attribute is empty
    all_values_same_boolean = check_all_values_same(input_Y,feature_Y_values)   
    if all_values_same_boolean or Attribute == []:
        return DT_cont(LeafValue=mcv)
    else:
        best_GR = 0
        best_set_X = []
        best_set_Y = []
        best_Attribute = 0
        best_threshold = 0.0
        current_entrophy = entrophy(input_Y,feature_Y_values)
        for index in Attribute:
            # index is the column index --> decides which attribute to use
            all_thresholds = return_all_thresholds(input_X,input_Y,index)
            for threshold in all_thresholds:
                # Get divided data for each threshold for a given attribute = index
                divided_X, divided_Y = divide_data(input_X,input_Y,index,feature_X_values,threshold)

                for j in range(len(divided_Y)): #iterate through each possible feature value
                    # For wine.data --> divided_Y contains subsets for lower then threshold
                    # and higher than the threshold
                    current_IG = information_gain(divided_Y[j],feature_Y_values,current_entrophy)
                    current_GR = gain_ratio(divided_Y[j],feature_Y_values,current_IG)
                    if current_GR >= best_GR:
                        best_GR = current_GR
                        best_Attribute = index
                        best_feature_ind = j
                        best_threshold = threshold
                    

        # Now best attribute and threshold are selected --> divide data using that attribute and thresh
        # best_Attribute --> gives the column number of best attribute
        # best threshold --> gives the best threshold value which gives max IG
        # best_feature_ind --> gives which feature value has the most IG, i.e. "x", "o" or "b" in tic-tac
        best_set_X, best_set_Y = divide_data(input_X,input_Y,best_Attribute,feature_X_values,best_threshold)
     
        # Remove the current attribute from the attribute list to
        # create remaning attributes --> no need to use same attribute over and
        # over again! --> If it is needed, it is gonna show up in the recursion anyway
        remaning_Attributes = []
        for index in Attribute:
            if index != best_Attribute:
                remaning_Attributes.append(index)

        # Left contains values less or equal than threshold
        # Right contains values more than threshold
        # For more detail, check divide data function
        #left.append( BuildingDT_cont(best_set_X[0],best_set_Y[0],feature_X_values,feature_Y_values,  
                                                #remaning_Attributes,mcv) )
        #right.append( BuildingDT_cont(best_set_X[1],best_set_Y[1],feature_X_values,feature_Y_values,  
                                                #remaning_Attributes,mcv) )
        return DT_cont(Left=BuildingDT_cont(best_set_X[0],best_set_Y[0],feature_X_values,feature_Y_values,
                        remaning_Attributes,mcv),
                       Right=BuildingDT_cont(best_set_X[1],best_set_Y[1],feature_X_values,feature_Y_values, 
                        remaning_Attributes,mcv),
                       FeatureNum=best_Attribute,ThresholdNum=best_threshold 
                       )
                    

            
def predict(in_tree,in_point,features):
    """
        This function predicts the output label for a given in_point
        using in_tree (decision tree), which was built using training data before
        Input: in_tree --> Input decision tree (at the beginning it is just root node)
               in_point --> Input point which contains all the features to predict output
               features --> feature values --> tic-tac-toe case: "x", "o", "b"
        For example, for tic-tac-toe --> in_point = ["x","o","x","o", ....] --> size 1x9
        Output: predicted class label (output) for given in_point
        For tic-tac-toe --> it is either "positive" or "negative"
    """

    if in_tree.LeafValue: #If given tree node is a leaf
        return in_tree.LeafValue

    feature_index = in_tree.FeatureNum # This is the feature(attribute) index number
    
    total_feature_value_number = len(features)
    for i in range(total_feature_value_number):
        if in_point[feature_index] == features[i]:
            return predict(in_tree.Children[i],in_point,features)  

def predict_cont(in_tree,in_point):
    """
        This function predicts the output label for a given in_point with continious
        feature values using in_tree (decision tree), which was built using training data before
        Input: in_tree --> Input decision tree (at the beginning it is just root node)
               in_point --> Input point which contains all the features to predict output
        Output: predicted class label (output) for given in_point
    """
    if in_tree.LeafValue: #If given tree node is a leaf
        return in_tree.LeafValue

    feature_index = in_tree.FeatureNum # This is the feature(attribute) index number
    
    total_feature_value_number = len(in_point)
    for i in range(total_feature_value_number):
        if in_point[feature_index] <= in_tree.ThresholdNum:
            return predict_cont(in_tree.Left,in_point)
        else:
            return predict_cont(in_tree.Right,in_point)
 
def ten_fold_cv(X_data,Y_data,features_X,features_Y,UseGR=False):
    """
        This function applies 10 fold Cross validation on the given input data
        Inputs: X_data --> input data contains features and their values
                Y_data --> input data contains output labels
                features_X --> feature values like "x","o","b" in tic-tac-toe
                features_Y --> output labels like "positive", "negative" in tic-tac
        Inputs: UseGR --> Optional boolean input
        If it is selected True, then trees will be created based on maximizing gain ratio
        instead of information gatio
        Output: true_error --> scalar
    """

    # Shuffle data at the beginning 
    x1_size, x2_size = X_data.shape
    y1_size, y2_size = Y_data.shape

    combined_data = np.concatenate((X_data,Y_data),axis=1)
    np.random.shuffle(combined_data) #this function shuffles

    X_shuffled = combined_data[:,0:x2_size]
    Y_shuffled = combined_data[:,x2_size:]
    
    # Divide data into 10 parts
    m = x1_size #total sample size
    remainder = 10 - m%10 #last piece will have this much less elements
    regular_length = (m+remainder) // 10
    # For example, when m=958 --> remainder is 2
    # Piece1 length --> 96 , Piece2 length --> 96 ... Piece 10 length --> 96-2=94
    
    best_accuracy = 0 
    accuracies = [] # this list contains accuracy for each validation set

    ##### First 9 pieces --> last piece might have different length than regular_length
    for i in range(9):
        # initial and end point index
        init = i*regular_length
        end = (i+1) * regular_length
        
        # selected test set
        test_X = X_shuffled[init:end,:] #contains features related to groundtruth
        test_Y = Y_shuffled[init:end,:] #contains true labels (groundtruth)
        
        # selected training set --> rest of data after selecting test set
        x_part1 = X_shuffled[0:init,:]
        x_part2 = X_shuffled[end:,:]
        y_part1 = Y_shuffled[0:init,:]
        y_part2 = Y_shuffled[end:,:]
        training_X = np.concatenate((x_part1,x_part2))
        training_Y = np.concatenate((y_part1,y_part2))

        # train the tree using training set
        if UseGR == False:
            current_tree = BuildingDT(training_X,training_Y,features_X,features_Y,
                    [0,1,2,3,4,5,6,7,8],most_common_value(training_Y,features_Y))
        elif UseGR == True:
            current_tree = BuildingDT_GR(training_X,training_Y,features_X,features_Y,
                    [0,1,2,3,4,5,6,7,8],most_common_value(training_Y,features_Y))

        predicted_labels = np.empty((test_Y.shape),dtype=object)
        for i in range(test_X.shape[0]):
            test_list = list(test_X[i,:])
            predicted_labels[i] = predict(current_tree,test_list,features_X) 
        
        # This variable holds the total number of correct predictions
        correct_prediction_number = len(test_Y[test_Y == predicted_labels])
        current_accuracy = correct_prediction_number/regular_length
     
        accuracies.append(current_accuracy) #hold all accuracies in a list

        if current_accuracy >= best_accuracy:
            best_tree = current_tree
            best_test_X = test_X
            best_test_Y = test_Y
            best_accuracy = current_accuracy

    ##### Last piece --> length could be less than regular_length
    # selected test set
    test_X_last_piece = X_shuffled[end:,:] #contains features related to groundtruth
    test_Y_last_piece = Y_shuffled[end:,:] #contains true labels (groundtruth)
    
    # selected training set --> rest of data after selecting test set
    training_X_last_piece = X_shuffled[0:end,:]
    training_Y_last_piece = Y_shuffled[0:end,:]
    
    # train the last tree using last training set piece
    if UseGR == False: 
        last_tree = BuildingDT(training_X_last_piece,training_Y_last_piece,features_X,features_Y,
                    [0,1,2,3,4,5,6,7,8],most_common_value(training_Y_last_piece,features_Y))
    elif UseGR == True:
        last_tree = BuildingDT_GR(training_X_last_piece,training_Y_last_piece,features_X,features_Y,
                    [0,1,2,3,4,5,6,7,8],most_common_value(training_Y_last_piece,features_Y))    
    
    last_predicted_labels = np.empty((test_Y_last_piece.shape),dtype=object)
    for i in range(test_X_last_piece.shape[0]):
        #print(list(test_X[i,:]))
        #print(type(test_X[i,:]))
        test_list = list(test_X_last_piece[i,:])
        last_predicted_labels[i] = predict(last_tree,test_list,features_X) 
    
    last_length = test_X.shape[0]


    last_prediction_number = len(test_Y_last_piece[test_Y_last_piece == last_predicted_labels])
    last_accuracy = last_prediction_number/regular_length
    
    accuracies.append(last_accuracy)

    if last_accuracy >= best_accuracy:
        best_accuracy = last_accuracy
        best_tree = last_tree
        best_test_X = test_X_last_piece
        best_test_Y = test_Y_last_piece    

    return accuracies, best_accuracy, best_tree, best_test_X, best_test_Y

def ten_fold_cv_cont(X_data,Y_data,features_Y,UseGR=False):
    """
        This function applies 10 fold Cross validation on the given input data
        with continious feature values
        Inputs: X_data --> input data contains features and their values
                Y_data --> input data contains output labels
                features_Y --> output labels like "1", "2", "3" in wine.data
                UseGR --> Optional boolean input
        If it is selected True, then trees will be created based on maximizing gain ratio
        instead of information gatio
        Output: true_error --> scalar
    """

    # Shuffle data at the beginning 
    x1_size, x2_size = X_data.shape
    y1_size, y2_size = Y_data.shape

    combined_data = np.concatenate((X_data,Y_data),axis=1)
    np.random.shuffle(combined_data) #this function shuffles

    X_shuffled = combined_data[:,0:x2_size]
    Y_shuffled = combined_data[:,x2_size:]
    
    # Divide data into 10 parts
    m = x1_size #total sample size
    remainder = 10 - m%10 #last piece will have this much less elements
    regular_length = (m+remainder) // 10
    # For example, when m=958 --> remainder is 2
    # Piece1 length --> 96 , Piece2 length --> 96 ... Piece 10 length --> 96-2=94
    
    best_accuracy = 0 
    accuracies = [] # this list contains accuracy for each validation set

    ##### First 9 pieces --> last piece might have different length than regular_length
    for i in range(9):
        # initial and end point index
        init = i*regular_length
        end = (i+1) * regular_length
        
        # selected test set
        test_X = X_shuffled[init:end,:] #contains features related to groundtruth
        test_Y = Y_shuffled[init:end,:] #contains true labels (groundtruth)
        
        # selected training set --> rest of data after selecting test set
        x_part1 = X_shuffled[0:init,:]
        x_part2 = X_shuffled[end:,:]
        y_part1 = Y_shuffled[0:init,:]
        y_part2 = Y_shuffled[end:,:]
        training_X = np.concatenate((x_part1,x_part2))
        training_Y = np.concatenate((y_part1,y_part2))

        # train the tree using training set
        if UseGR == False:
            current_tree = BuildingDT_cont(training_X,training_Y,[],features_Y,
                    [0,1,2,3,4,5,6,7,8,9,10,11,12],most_common_value(training_Y,features_Y))
        elif UseGR == True:
            current_tree = BuildingDT_cont_GR(training_X,training_Y,[],features_Y,
                    [0,1,2,3,4,5,6,7,8,9,10,11,12],most_common_value(training_Y,features_Y))

        predicted_labels = np.empty((test_Y.shape),dtype=object)
        for i in range(test_X.shape[0]):
            test_list = list(test_X[i,:])
            predicted_labels[i] = predict_cont(current_tree,test_list) 
        
        # This variable holds the total number of correct predictions
        correct_prediction_number = len(test_Y[test_Y == predicted_labels])
        current_accuracy = correct_prediction_number/regular_length
     
        accuracies.append(current_accuracy) #hold all accuracies in a list

        if current_accuracy >= best_accuracy:
            best_tree = current_tree
            best_test_X = test_X
            best_test_Y = test_Y
            best_accuracy = current_accuracy

    ##### Last piece --> length could be less than regular_length
    # selected test set
    test_X_last_piece = X_shuffled[end:,:] #contains features related to groundtruth
    test_Y_last_piece = Y_shuffled[end:,:] #contains true labels (groundtruth)
    
    # selected training set --> rest of data after selecting test set
    training_X_last_piece = X_shuffled[0:end,:]
    training_Y_last_piece = Y_shuffled[0:end,:]
    
    # train the last tree using last training set piece
    if UseGR == False: 
        last_tree = BuildingDT_cont(training_X_last_piece,training_Y_last_piece,[],features_Y,
                    [0,1,2,3,4,5,6,7,8,9,10,11,12],most_common_value(training_Y_last_piece,features_Y))
    elif UseGR == True:
        last_tree = BuildingDT_cont_GR(training_X_last_piece,training_Y_last_piece,[],features_Y,
                    [0,1,2,3,4,5,6,7,8,9,10,11,12],most_common_value(training_Y_last_piece,features_Y))

    last_predicted_labels = np.empty((test_Y_last_piece.shape),dtype=object)
    for i in range(test_X_last_piece.shape[0]):
        #print(list(test_X[i,:]))
        #print(type(test_X[i,:]))
        test_list = list(test_X_last_piece[i,:])
        last_predicted_labels[i] = predict_cont(last_tree,test_list) 
    
    last_length = test_X.shape[0]

    last_prediction_number = len(test_Y_last_piece[test_Y_last_piece == last_predicted_labels])
    last_accuracy = last_prediction_number/regular_length
    
    accuracies.append(last_accuracy)

    if last_accuracy >= best_accuracy:
        best_accuracy = last_accuracy
        best_tree = last_tree
        best_test_X = test_X_last_piece
        best_test_Y = test_Y_last_piece    

    return accuracies, best_accuracy, best_tree, best_test_X, best_test_Y

def calculate_conf_matrix(tree,test_X,test_Y,features_X,features_Y):
    """
        This function calculates confusion matrix using a given tree.
        It uses given tree with given test_X to create prediction and
        compares them with the ground truth labels in test_Y
        features_X --> contains possible feature values --> "x","b" in tic-tac
    """

    predictions = np.empty((test_Y.shape),dtype=object)
    for i in range(test_X.shape[0]):
        test_list = list(test_X[i,:])
        predictions[i] = predict(tree,test_list,features_X) 

    total_true_positives = len(test_Y[test_Y == features_Y[0]])
    total_true_negatives = len(test_Y[test_Y == features_Y[1]])
    
    true_positives_indices = (test_Y == features_Y[0])
    true_negative_indices = (test_Y == features_Y[1])
    candidate_pred_positives = predictions[true_positives_indices]
    candidate_pred_negatives = predictions[true_negative_indices]

    # When the true label is positive, prediction is positive
    correct_predic_positives = len(candidate_pred_positives[candidate_pred_positives == features_Y[0]])
    # When the true label is positive, prediction is negative
    wrong_predic_negatives = len(candidate_pred_positives[candidate_pred_positives == features_Y[1]])
    # When the true label is negative, prediction is negative
    correct_predic_negatives = len(candidate_pred_negatives[candidate_pred_negatives == features_Y[1]])
    # When the true label is negative, prediction is positive
    wrong_predic_positives = len(candidate_pred_negatives[candidate_pred_negatives == features_Y[0]])

    #print("Total number of test samples: ", len(test_Y))
    #print("Total true positives: ", total_true_positives)
    #print("Total true negatives: ", total_true_negatives)
    
    print("###########Confusion matrix##############")
    print("Correct predict positives: ", correct_predic_positives)
    print("Wrong predict negatives: ", wrong_predic_negatives)
    print("Wrong predict positives: ", wrong_predic_positives)
    print("Correct predict negatives: ", correct_predic_negatives)

    
def calculate_conf_matrix_cont(tree,test_X,test_Y,features_Y):
    """
        This function calculates confusion matrix using a given tree 
        for a given data with continious attribute values
        It uses given tree with given test_X to create prediction and
        compares them with the ground truth labels in test_Y
        For wine.data --> features_Y --> "1", "2", "3"
    """

    predictions = np.empty((test_Y.shape),dtype=object)
    for i in range(test_X.shape[0]):
        test_list = list(test_X[i,:])
        predictions[i] = predict_cont(tree,test_list) 

    total_true_ones = len(test_Y[test_Y == features_Y[0]])
    total_true_twos = len(test_Y[test_Y == features_Y[1]])
    total_true_threes = len(test_Y[test_Y == features_Y[2]])

    # Assuming features_Y --> [1,2,3] (Order is important!)
    true_ones_indices = (test_Y == features_Y[0])
    true_twos_indices = (test_Y == features_Y[1])
    true_threes_indices = (test_Y == features_Y[2])

    candidate_pred_ones = predictions[true_ones_indices]
    candidate_pred_twos = predictions[true_twos_indices]
    candidate_pred_threes = predictions[true_threes_indices]

    # When the true label is one and prediction is one
    correct_predic_ones = len(candidate_pred_ones[candidate_pred_ones == features_Y[0]])
    # When the true label is two and prediction is two
    correct_predic_twos = len(candidate_pred_twos[candidate_pred_twos == features_Y[1]])
    # When the true label is three and prediction is three
    correct_predic_threes = len(candidate_pred_threes[candidate_pred_threes == features_Y[2]])

    # When the true label is one, but prediction is two
    wrong_true1_predic2 = len(candidate_pred_ones[candidate_pred_ones == features_Y[1]])
    # When the true label is one, but prediction is three
    wrong_true1_predic3 = len(candidate_pred_ones[candidate_pred_ones == features_Y[2]])

    # When the true label is two, but prediction is one
    wrong_true2_predic1 = len(candidate_pred_twos[candidate_pred_twos == features_Y[0]])
    # When the true label is two, but prediction is three
    wrong_true2_predic3 = len(candidate_pred_twos[candidate_pred_twos == features_Y[2]])

    # When the true label is three, but prediction is one
    wrong_true3_predic1 = len(candidate_pred_threes[candidate_pred_threes == features_Y[0]])
    # When the true label is three, but prediction is two
    wrong_true3_predic2 = len(candidate_pred_threes[candidate_pred_threes == features_Y[1]])

    #print("True label:")
    #print(test_Y)
    #print("Predictions")
    #print(predictions)
    
    print("###########Confusion matrix##############")
    print("True 1 predict 1: ", correct_predic_ones)
    print("True 1 predict 2: ", wrong_true1_predic2)
    print("True 1 predict 3: ", wrong_true1_predic3)
    print("True 2 predict 1: ", wrong_true2_predic1)
    print("True 2 predict 2: ", correct_predic_twos)
    print("True 2 predict 3: ", wrong_true2_predic3)
    print("True 3 predict 1: ", wrong_true3_predic1)
    print("True 3 predict 2: ", wrong_true3_predic2)
    print("True 3 predict 3: ", correct_predic_threes)


def ten_times_10_fold(X_in,Y_in,X_features,Y_features,UseGR=False):
    """
        This function calls 10-fold cross validation function 10 times
        and returns best tree, best accuracy, best training set
        Inputs: UseGR --> Optional boolean input
        If it is selected True, then trees will be created based on maximizing gain ratio
        instead of information gatio
    """
    
    ten_time_accurs_mean = []
    ten_time_accurs_var = []
    ten_time_best_accr = 0

    for i in range(10):
        one_time_accurs,one_time_best_accr,one_time_best_tree,one_time_best_X,one_time_best_Y=( 
                     ten_fold_cv(X_in,Y_in,X_features,Y_features,UseGR))

        ten_time_accurs_mean.append(np.mean(one_time_accurs))
        ten_time_accurs_var.append(np.var(one_time_accurs))
        if one_time_best_accr >= ten_time_best_accr:
            ten_time_best_tree = one_time_best_tree
            ten_time_best_test_X = one_time_best_X
            ten_time_best_test_Y = one_time_best_Y
            ten_time_best_accr = one_time_best_accr

    print("Mean of the accuracy: ", np.mean(ten_time_accurs_mean))
    print("Var of the accuracy: ", np.var(ten_time_accurs_var))
    print("Best accuracy: ", ten_time_best_accr)

    return ten_time_best_accr, ten_time_best_tree, ten_time_best_test_X, ten_time_best_test_Y

def ten_times_10_fold_cont(X_in,Y_in,Y_features,UseGR=False):
    """
        This function calls 10-fold cross validation function 10 times
        and returns best tree, best accuracy, best training set
        Inputs: UseGR --> Optional boolean input
        If it is selected True, then trees will be created based on maximizing gain ratio
        instead of information gatio
    """
    
    ten_time_accurs_mean = []
    ten_time_accurs_var = []
    ten_time_best_accr = 0

    for i in range(10):
        one_time_accurs,one_time_best_accr,one_time_best_tree,one_time_best_X,one_time_best_Y=( 
                     ten_fold_cv_cont(X_in,Y_in,Y_features,UseGR))

        ten_time_accurs_mean.append(np.mean(one_time_accurs))
        ten_time_accurs_var.append(np.var(one_time_accurs))
        if one_time_best_accr >= ten_time_best_accr:
            ten_time_best_tree = one_time_best_tree
            ten_time_best_test_X = one_time_best_X
            ten_time_best_test_Y = one_time_best_Y
            ten_time_best_accr = one_time_best_accr

    print("Mean of the accuracy: ", np.mean(ten_time_accurs_mean))
    print("Var of the accuracy: ", np.var(ten_time_accurs_var))
    print("Best accuracy: ", ten_time_best_accr)

    return ten_time_best_accr, ten_time_best_tree, ten_time_best_test_X, ten_time_best_test_Y



def main():
    """
        This is the main function of this script
    """

    ############ Read Data ############
    # If you run the code from pattern_recognition_assignment1 path, uncomment below
    data_dir = os.getcwd() + '/data' 
    # If you run the code from code directory, uncomment below
    #data_path = os.getcwd() +  ".." / "data"/

    # Tic-tac-toe data has 9 features and 2 output labels --> discrete attributes
    X_tictac, Y_tictac = read_data_string(data_dir,"tic-tac-toe.data",9)
    features_X_tictac = ["x","o","b"]
    features_Y_tictac = ["positive","negative"]
    
    # Wine data has 13 features and 3 output labels --> continuous attributes
    # All features of wine X is float --> no need to have features
    # WARNING: column 1 is composed of output labels in wine.data
    X_wine, Y_wine = read_data(data_dir,"wine.data",13)
    features_Y_wine = [1,2,3]

    
    ############ Test tic-tac-toe data ############
    print('###################################')
    print("######### Test tic-tac-toe data ###########")

    # Apply 10 times 10-fold cross validation
    print("Based on Information Gain")
    best_accr_tic, best_tree_tic, best_test_X_tic, best_test_Y_tic = ten_times_10_fold(X_tictac,Y_tictac,features_X_tictac,features_Y_tictac,False)
    # Calculate confusion matrix using best tree     
    calculate_conf_matrix(best_tree_tic,best_test_X_tic,best_test_Y_tic,features_X_tictac,features_Y_tictac)
    
    print("Based on Gain Ratio")
    best_accr_tic2, best_tree_tic2, best_test_X_tic2, best_test_Y_tic2 = ten_times_10_fold(X_tictac,Y_tictac,features_X_tictac,features_Y_tictac,True)
    # Calculate confusion matrix using best tree     
    calculate_conf_matrix(best_tree_tic2,best_test_X_tic2,best_test_Y_tic2,features_X_tictac,features_Y_tictac)
    
    ############ Wine data ############
    print('###################################')
    print("######### Test wine data ###########")

    # Apply 10 times 10-fold cross validation
    print("Based on Information Gain")
    best_accr_wine, best_tree_wine, best_test_X_wine, best_test_Y_wine = ten_times_10_fold_cont(X_wine,Y_wine,features_Y_wine,False)
    # Calculate confusion matrix using best tree     
    calculate_conf_matrix_cont(best_tree_wine,best_test_X_wine,best_test_Y_wine,features_Y_wine)
    
    print("Based on Gain Ratio")
    best_accr_wine2, best_tree_wine2, best_test_X_wine2, best_test_Y_wine2 = ten_times_10_fold_cont(X_wine,Y_wine,features_Y_wine,True)
    # Calculate confusion matrix using best tree     
    calculate_conf_matrix_cont(best_tree_wine2,best_test_X_wine2,best_test_Y_wine2,features_Y_wine)
    
    


if __name__ == "__main__": main()
















