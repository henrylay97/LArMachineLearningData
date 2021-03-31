#!/usr/bin/env python
# PandoraBDT.py

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn import preprocessing
from datetime import datetime

import numpy as np
import scipy.stats as sci
import sys
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
import math

from PandoraMVA import *

def DrawVariablesDF(df, params, logY=False):
    for column in df:
        if column == 'Labels':
            continue    
        fig, ax = plt.subplots()
        df.pivot(columns='Labels')[column].plot.hist(bins=50, alpha=0.5, color=params['signalCols'], edgecolor='k', density=True, ax=ax)
        ax.legend(params['labelNames']);
        ax.set_xlabel(column.replace("_", " "))

        plt.tight_layout()
        plt.savefig('Feature_' + column + '.png')
        plt.savefig('Feature_' + column + '.pdf')
        plt.show()
        plt.close()

def DrawVariablesDF(df, params, topdir, save=True):
    for column in df:
        if column == 'Labels':
            continue    
        fig, ax = plt.subplots()
        df.pivot(columns='Labels')[column].plot.hist(bins=50, alpha=0.5, color=params['signalCols'], edgecolor='k', density=True, ax=ax)
        ax.legend(params['labelNames']);
        ax.set_xlabel(column.replace("_", " "))

        plt.tight_layout()
        if save:
            plt.savefig(topdir + 'Feature_' + column + '.png')
            plt.savefig(topdir + 'Feature_' + column + '.pdf')
        plt.show()
        plt.close()

        
def DrawVariables(X, Y, labels, logY=True, class_names=['Background', 'Signal']):
    plot_colors = ['r', 'b']
    signal_definition = [0, 1]

    num_rows, num_cols = X.shape
    for feature in range(0, num_cols):
        # Set the plot range for 2% - 98% to eliminate outliers
        plot_range = (np.quantile(X[:, feature], 0.02),
                      np.quantile(X[:, feature], 0.98))

        for i, label, color in zip(signal_definition, class_names, plot_colors):
            entries, bins, patches = plt.hist(X[:, feature][Y == i],
                                              bins=50,
                                              range=plot_range,
                                              facecolor=color,
                                              label=label,
                                              alpha=.5,
                                              edgecolor='k',
                                              density=True)
            if (logY):
                plt.yscale('log')

        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, y1, y2 * 1.1))
        plt.legend()
        plt.ylabel('Entries')
        plt.xlabel(labels[feature].replace("_", " "))

        plt.tight_layout()
        plt.savefig('Feature_' + labels[feature] + '.png')
        plt.savefig('Feature_' + labels[feature] + '.pdf')
        plt.show()
        plt.close()

# --------------------------------------------------------------------------------------------------


def PlotGridSearch(df, label):
    plt.figure(figsize=(10, 10))
    plt.title(label)

    ax = sns.heatmap(df, cmap='coolwarm', annot=True, square=True, fmt='.2g')

    # ax.invert_yaxis()
    # ax.set_ylim(-0., len(df.columns)-0.5)
    plt.savefig(label.replace(" ", "_") + ".png", bbox_inches='tight')
    plt.savefig(label.replace(" ", "_") + ".pdf", bbox_inches='tight')
    plt.show()
    plt.close()


# --------------------------------------------------------------------------------------------------


def Correlation(df, label):
    plt.figure(figsize=(10, 10))
    plt.title(label)

    ax = sns.heatmap(df.corr(), cmap='coolwarm', vmax=1.0, vmin=-1.0,
                     annot=True, square=True, fmt='.2g')

    ax.invert_yaxis()
    # ax.set_ylim(-0., len(df.columns)-0.5)
    plt.savefig(label.replace(" ", "_") + ".png", bbox_inches='tight')
    plt.savefig(label.replace(" ", "_") + ".pdf", bbox_inches='tight')
    plt.show()
    plt.close()

# --------------------------------------------------------------------------------------------------


def Correlation(df, label, topdir, save=False):
    plt.figure(figsize=(10, 10))
    plt.title(label)

    ax = sns.heatmap(df.corr(), cmap='coolwarm', vmax=1.0, vmin=-1.0,
                     annot=True, square=True, fmt='.2g')

    ax.invert_yaxis()

    if save:
        plt.savefig(topdir + '/' + label.replace(" ", "_") + ".png", bbox_inches='tight')
        plt.savefig(topdir + '/' + label.replace(" ", "_") + ".pdf", bbox_inches='tight')
    plt.show()
    plt.close()


# --------------------------------------------------------------------------------------------------


def TrainAdaBoostClassifer(X_train, Y_train, n_estimatorsValue=3, max_depthValue=2, learning_rateValue=1.0,
                           algorithmValue='SAMME', random_stateValue=42):
    # Load the BDT object
    bdtModel = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depthValue),
                                  n_estimators=n_estimatorsValue, learning_rate=learning_rateValue,
                                  algorithm=algorithmValue, random_state=random_stateValue)

    # Train the model
    startTime = time.time()
    bdtModel.fit(X_train, Y_train)
    endTime = time.time()

    return bdtModel, endTime - startTime

# --------------------------------------------------------------------------------------------------


def TrainGradientBoostingClassifer(X_train, Y_train, n_estimatorsValue=3, max_depthValue=2, learning_rateValue=0.1, subsampleValue=1.0, lossFunction='deviance', random_stateValue=42):
    # Load the BDT object
    bdtModel = GradientBoostingClassifier(max_depth=max_depthValue,
                                          n_estimators=n_estimatorsValue, learning_rate=learning_rateValue,
                                          loss=lossFunction, random_state=random_stateValue)

    # Train the model
    startTime = time.time()
    bdtModel.fit(X_train, Y_train)
    endTime = time.time()

    return bdtModel, endTime - startTime

# --------------------------------------------------------------------------------------------------


def WriteXmlFile(filePath, adaBoostClassifer, BDTName):
    datetimeString = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    with open(filePath, "w+") as modelFile:
        indentation = 0
        indentation = OpenXmlTag(
            modelFile,    'AdaBoostDecisionTree', indentation)
        WriteXmlFeature(modelFile, BDTName, 'Name', indentation)
        WriteXmlFeature(modelFile, datetimeString, 'Timestamp', indentation)

        for idx, estimator in enumerate(adaBoostClassifer.estimators_):
            boostWeight = adaBoostClassifer.estimator_weights_[idx]
            WriteDecisionTree(estimator, modelFile,
                              indentation, idx, boostWeight)

        CloseXmlTag(modelFile,
                    'AdaBoostDecisionTree', indentation)

# --------------------------------------------------------------------------------------------------


def Recurse(node, parentnode, depth, position, indentation, decisionTree, modelFile):
    indentation = OpenXmlTag(modelFile, 'Node', indentation)
    WriteXmlFeature(modelFile, node, 'NodeId', indentation)
    WriteXmlFeature(modelFile, parentnode, 'ParentNodeId', indentation)

    if decisionTree.feature[node] != _tree.TREE_UNDEFINED:
        name = decisionTree.feature[node]  # (int)(node) #feature_name[node]
        threshold = decisionTree.threshold[node]
        WriteXmlFeature(modelFile, name, 'VariableId', indentation)
        WriteXmlFeature(modelFile, threshold, 'Threshold', indentation)
        WriteXmlFeature(
            modelFile, decisionTree.children_left[node], 'LeftChildNodeId', indentation)
        WriteXmlFeature(
            modelFile, decisionTree.children_right[node], 'RightChildNodeId', indentation)
        indentation = CloseXmlTag(modelFile, 'Node', indentation)
        indentation = indentation + 4
        Recurse(decisionTree.children_left[node], node, depth +
                1, 'Left', indentation, decisionTree, modelFile)
        Recurse(decisionTree.children_right[node], node, depth +
                1, 'Right', indentation, decisionTree, modelFile)
        indentation = indentation - 4
    else:
        result = decisionTree.value[node]
        if (result.tolist()[0][1] > result.tolist()[0][0]):
            WriteXmlFeature(modelFile, 'true', 'Outcome', indentation)
        else:
            WriteXmlFeature(modelFile, 'false', 'Outcome', indentation)

        indentation = CloseXmlTag(modelFile, 'Node', indentation)

# --------------------------------------------------------------------------------------------------


def WriteDecisionTree(estimator, modelFile, indentation, treeIdx, boostWeight):
    decisionTree = estimator.tree_
    indentation = OpenXmlTag(modelFile, 'DecisionTree', indentation)

    WriteXmlFeature(modelFile, treeIdx, 'TreeIndex', indentation)
    WriteXmlFeature(modelFile, boostWeight, 'TreeWeight', indentation)
    Recurse(0, -1, 1, 'Start', indentation, decisionTree, modelFile)

    indentation = CloseXmlTag(modelFile, 'DecisionTree', indentation)

# --------------------------------------------------------------------------------------------------


def SerializeToPkl(fileName, model):
    with open(fileName, 'wb') as f:
        pickle.dump(model, f)

# --------------------------------------------------------------------------------------------------


def LoadFromPkl(fileName):
    with open(fileName, 'rb') as f:
        model = pickle.load(f)

        return model

# --------------------------------------------------------------------------------------------------


def FindOptimalSignificanceCut(bdtModel, X_train, Y_train, parameters):
    # Find optimal cut based on significance
    train_results = bdtModel.decision_function(X_train)
    fig, ax = plt.subplots()

    sigEff = 0
    bkgRej = 0
    sigEntries = []
    bkgEntries = []

    for i, n, g in zip(parameters['SignalDefinition'], parameters['ClassNames'], parameters['PlotColors']):
        entries, bins, patches = ax.hist(train_results[Y_train == i],
                                         bins=parameters['nBins'],
                                         range=(-1, 1),
                                         facecolor=g,
                                         label='Class %s' % n,
                                         alpha=.5,
                                         edgecolor='k')
        if i == 1:
            sigEntries = entries
        elif i == 0:
            bkgEntries = entries

    nSigEntries = sum(sigEntries)
    nBkgEntries = sum(bkgEntries)
    optimalSignif = 0
    optimalSigEff = 0
    optimalBkgRej = 0
    optimalBinCut = 0
    optimalScoreCut = 0

    for binCut in range(0, parameters['nBins']):
        nSigPassing = sum(sigEntries[binCut:])
        nBkgFailing = sum(bkgEntries[:binCut])
        signif = nSigPassing * nBkgFailing

        if (signif > optimalSignif):
            sigEff = 100 * nSigPassing / nSigEntries
            bkgRej = 100 * nBkgFailing / nBkgEntries
            optimalSignif = signif
            optimalBinCut = binCut
            optimalScoreCut = bins[optimalBinCut]
            optimalSigEff = sigEff
            optimalBkgRej = bkgRej

    print('Optimal signif : ' + str(optimalSignif))
    print('Optimal sigEff : ' + str(optimalSigEff))
    print('Optimal bkgRej : ' + str(optimalBkgRej))
    print('Optimal binCut : ' + str(optimalBinCut))
    print('Optimal scoreCut : ' + str(round(optimalScoreCut,2)))
    parameters['OptimalBinCut'] = optimalBinCut
    parameters['OptimalScoreCut'] = optimalScoreCut

    #plt.show()
    plt.close()

# --------------------------------------------------------------------------------------------------


def PlotBdtScores(bdtModel, X_test, Y_test, X_train, Y_train, title, parameters, topDir, save=False):
    # Testing BDT Using Remainder of Training Sample
    test_results = bdtModel.decision_function(X_test)
    train_results = bdtModel.decision_function(X_train)

    test_results_signal = test_results[Y_test == 1]
    train_results_signal = train_results[Y_train == 1]
    test_results_background = test_results[Y_test == 0]
    train_results_background = train_results[Y_train == 0]

    fig, ax = plt.subplots()

    ax.set_title('Overtraining Test: ' + title)

    sigEff = 0
    bkgRej = 0

    for i, n, g in zip(parameters['SignalDefinition'], parameters['ClassNames'], parameters['PlotColors']):
        entries, bins, patches = ax.hist(train_results[Y_train == i],
                                         bins=parameters['nBins'],
                                         range=(-1, 1),
                                         facecolor=g,
                                         label='%s' % n,
                                         alpha=.5,
                                         density=True,
                                         edgecolor='k')

        counts, bin_edges = np.histogram(test_results[Y_test == i],
                                         range=(-1, 1), bins=parameters['nBins'], density=True)

        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
        ax.errorbar(bin_centres, counts, fmt='o', color=g)

        if i == 1:
            nEntries = sum(counts)
            nEntriesPassing = sum(counts[parameters['OptimalBinCut']:])
            sigEff = nEntriesPassing/nEntries
        elif i == 0:
            nEntries = sum(counts)
            nEntriesFailing = sum(counts[:parameters['OptimalBinCut']])
            bkgRej = nEntriesFailing/nEntries

    signalKSTest, ksSig = sci.ks_2samp(
        test_results_signal, train_results_signal)
    backgroundKSTest, ksBck = sci.ks_2samp(
        test_results_background, train_results_background)

    score = bdtModel.score(X_test,Y_test)

    plt.text(0.88, 0.5, "Sig Eff: {:.2%}\nBkg Rej: {:.2%}\nScore Cut: {:.2}\n\nSig KS: {:.2}\nBack KS: {:.2}\nSig P: {:.2}\nBck P: {:.2}\n\nScore: {:.4} "
             .format(sigEff, bkgRej, parameters['OptimalScoreCut'], signalKSTest, backgroundKSTest, ksSig, ksBck, score),
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)

    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2 * 1.1))
    plt.legend(loc='upper right')
    plt.ylabel('Samples')
    plt.xlabel('Score')
    plt.tight_layout()

    if save:
        plt.savefig(topDir + '/' + title.replace(" ", "_") + '_NTrees_' +
                    str(parameters['nTrees']) + '_TreeDepth_' + str(parameters['TreeDepth']) + '.pdf')
        plt.savefig(topDir + '/' + title.replace(" ", "_") + '_NTrees_' +
                    str(parameters['nTrees']) + '_TreeDepth_' + str(parameters['TreeDepth']) + '.png')

    plt.show()
    plt.close()

    print("KS Signal:     "+str(signalKSTest)+" with P value: "+str(ksSig))
    print("KS BackGround: "+str(backgroundKSTest)+" with P value: "+str(ksBck))
