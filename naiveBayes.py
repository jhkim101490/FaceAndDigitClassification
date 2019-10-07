# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import time


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False  # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        start = time.time()
        self.features = list(set([f for datum in trainingData for f in list(datum.keys())]))

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
        end = time.time()
        print(end - start)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.
        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.
        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        "*** YOUR CODE HERE ***"
        '''
            Prior Probability
        '''
        self.prior_probability = util.Counter()
        total_cnt_label = util.Counter()
        total_cnt_label.incrementAll(self.legalLabels, 0) # initialize for count how many each numbers have
        for label in trainingLabels:
            total_cnt_label.incrementAll([label], 1)
        total_cnt_label.normalize()
        self.prior_probability = total_cnt_label

        '''
            Conditional Probability
        '''
        self.conditional = util.Counter()
        non_conditional_feature_label = util.Counter()
        conditional_feature_label = util.Counter()
        for idx,current in enumerate(trainingData):
            real_label = trainingLabels[idx]
            for feature, value in current.items():
                if value >= 1:
                    conditional_feature_label.incrementAll([(feature, real_label)], 1)

                elif value == 0:
                    non_conditional_feature_label.incrementAll([(feature, real_label)], 1)

        total_features_labels = util.Counter()  # {}
        total_features_labels = conditional_feature_label.__add__(non_conditional_feature_label)

        '''
            Smoothing & conditional probability
        '''

        for label in self.legalLabels:
            for feature in self.features:
                conditional_feature_label.incrementAll([(feature, label)], self.k)
                non_conditional_feature_label.incrementAll([(feature, label)], self.k)

        for i, value in conditional_feature_label.items():
            self.conditional[i] = float(value) / float(total_features_labels[i])

        self.classify(trainingData)



    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"
        # Prior = probability of label (number of times label occurs in our trainingLabels out of all the other label values)
        # Datum = (feature,value)
        # loop through all of the legal labels, initialize logJoint[label] to log(prior)
        test = []
        test2 = []
        for label in self.legalLabels:
            logJoint[label] = math.log(self.prior_probability[label])
            # loop through all the features and their values in datum
            # Adjusted probability is the sum of log(prior) and the conditional probability of features given the label.
            for feature, value in datum.items():
                # If value for feature is 1, the event 'occurs' and we add the conditional probability,
                # Otherwise if value is 0, the event does not occur and we add the probability of the event not occuring (1- conditional probability)
                probability = self.conditional[feature, label] if value >= 1 else ((1 - self.conditional[feature, label])
                    if (1 - self.conditional[feature, label]) > 0 else 1) # it will negative value if 1-greater than 1, then it will be error occured,so just set log(1) which is 1
                logJoint[label] += math.log(probability)
            '''
            Debugging to solve logJoint[label] += math.log(probability) ValueError: math domain error
                if value > 0:
                    test.append(self.conditional[feature,label])
                else:
                    if self.conditional[feature,label] > 1: =>> here self.conditional has many greater than 1, so if we do not check then it will be error
                        test2.append(self.conditional[feature,label])
            '''

        # Return the adjusted probability for label l -> (posterior)
        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)
        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
