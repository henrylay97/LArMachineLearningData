#!/usr/bin/env python
# example.py

from PandoraBDT import *

if __name__=="__main__":

  # Settings ------------------------------------------------------------------------------------

    trainingFile      = '/home/tyley/PandoraBDT/thesis/vertex/region/VertexRegionMix.txt'

    bdtName           = 'VertexBDTRegion'
    treeDepth         = 3
    nTrees            = 100
    trainTestSplit    = 0.5

    plotFeatures      = False # Draws distributions of signal and background class features, then exits
    serializeToPkl    = False
    serializeToXml    = True
    loadFromPkl       = False
    makeScorePlots    = True # Makes plots of BDT score for training and testing samples
    xmlFileName       = bdtName + '_NTrees_' + str(nTrees) + '_TreeDepth_' + str(treeDepth) + '.xml'
    pklFileName       = bdtName + '_NTrees_' + str(nTrees) + '_TreeDepth_' + str(treeDepth) + '.pkl'

    #----------------------------------------------------------------------------------------------

    if plotFeatures:
      # Load the training data
      OverwriteStdout('Loading training set data for plotting from file ' + trainingFile + '\n')
      trainSet, nFeatures, nExamples = LoadData(trainingFile, ',')
      X_org, Y_org = SplitTrainingSet(trainSet, nFeatures)

      # Plot Variables then Exit
      DrawVariables(X_org, Y_org)
      Correlation(X_org, Y_org)
      sys.exit()

    if loadFromPkl:
      OverwriteStdout('Loading model from file ' + pklFileName + '\n')
      bdtModel = LoadFromPkl(pklFileName)

    else:
      # Load the training data
      OverwriteStdout('Loading training set data from file ' + trainingFile + '\n')
      trainSet, nFeatures, nExamples = LoadData(trainingFile, ',')
      X_org, Y_org = SplitTrainingSet(trainSet, nFeatures)

      # Train the BDT
      X, Y = Randomize(X_org, Y_org, True)
      X_train, Y_train, X_test, Y_test = Sample(X, Y, trainTestSplit)

      num = 4

      nTreesArray = np.logspace(0,num-1,num)
      maxDepthArray = np.arange(1,num+1)

      scores = np.empty((num,num))

      i = 0
      for nTreesValues in nTreesArray:
        j = 0
        for maxDepthValue in maxDepthArray:
          nTreesValue = int(nTreesValues)
          OverwriteStdout('Training AdaBoostClassifer...')
          bdtModel, trainingTime = TrainAdaBoostClassifer(X_train, Y_train, n_estimatorsValue=nTreesValue, max_depthValue=maxDepthValue, algorithmValue="SAMME")

          OverwriteStdout(('Trained AdaBoostClassifer with ' + str(nFeatures) + ' features and ' +
            str(nExamples) + ' examples (%d seconds, %i TreeDepth, %i nTrees)\n' % (trainingTime,
              maxDepthValue, nTreesValue)))

            # Validate the model
          modelScore = ValidateModel(bdtModel, X_test, Y_test)
          OverwriteStdout('Model score: %.2f%%\n' % (modelScore * 100))

          xmlFileName = bdtName + '_NTrees_' + str(nTreesValues) + '_TreeDepth_' + str(maxDepthValue) + '.xml'
          if serializeToXml:
            OverwriteStdout('Writing model to xml file ' + xmlFileName + '\n')
            datetimeString = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            WriteXmlFile(xmlFileName, bdtModel, bdtName)

          if makeScorePlots:
            parameters = {
                'ClassNames':['True Vertex','Incorrect Vertex'],
                'SignalDefinition': [1, 0],
                'PlotColors': ['b', 'r'],
                'nBins': 100,
                'PlotStep': 1.0,
                'OptimalBinCut': 0,
                'OptimalScoreCut': 0.0,
                'nTrees': nTreesValues,
                'TreeDepth': maxDepthValue
                }

            FindOptimalSignificanceCut(bdtModel, X_train, Y_train, parameters)
            PlotBdtScores(bdtModel, X_test, Y_test, X_train, Y_train, 'Region', parameters)

          scores[i][j] = modelScore

          j+=1
        i+=1

      OverwriteStdout("Finished Training, making plots...")

      # %% Do plotting
      fig = plt.figure(figsize=(6, 6))
      ax = fig.add_subplot(111)
      im = ax.imshow(scores, interpolation='nearest', cmap=plt.cm.bwr, origin='lower')

      plt.ylabel('NTrees')
      plt.xlabel('TreeDepth')
      fig.colorbar(im)
      plt.yticks(np.arange(len(nTreesArray)), nTreesArray)
      plt.xticks(np.arange(len(maxDepthArray)), maxDepthArray)
      plt.title('Validation accuracy')

      # Add the text
      for  y in range(num):
        for  x in range(num):
          label = '%.3f' %scores[y, x]
          ax.text(x, y, label, color='black', ha='center', va='center')

      plt.savefig("RegionGrid.pdf")
