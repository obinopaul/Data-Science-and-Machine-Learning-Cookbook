import pandas as pd
from MLD import MLD
from NearestNeighborSearch import NearestNeighborSearch
from sklearn.model_selection import train_test_split
import time
import numpy as np
from neuralNetwork import neuralNetwork
from Evaluate import Evaluate
from MLD_AMG import MLD_AMG
from testNeuralNetwork import testNetwork
from averageResults import averageResults
import tensorflow as tf
import keras
import pathlib # Create director if it does not exist (requires python 3.5+)
import argparse
import multiprocessing as mp

# For recording results
from secret import api_
import neptune
from memory_profiler import profile

from neuralNetwork import recall_m, precision_m, specificity_m, gmean_m
from neptune_tensorboard import enable_tensorboard_logging

def Multilevel(run, data, dataName, max_ite=1, prop=0.8, neural_earlyStop = None, multilevel=1, n_neighbors=10,
               Q=0.4, r=2, eta=1,Upperlim=500, Imb_size=300, Model_Selec=1, numBorderPoints=10, coarsening="AMG",
               loss="cross", alpha=0.5, gamma=4,Level_size=1, refineMethod="border", hb_epochs=10, hyperband_ite=1,
               epochs=100, batch_size=32, Dropout=0.3, batchnorm=0.2, factor=3, patienceLevel=2, weights=False, label=""):

    """
    The main function. Takes data and trains either a multilevel or traditional neural network
    Inputs:
        data: The dataset
        dataName: The name of dataset being used.
        max_ite: Number of iterations to average results over
        prop: Proportion of data to split into test/train  
        n_neighbors: Number of neighbors to use for coarsening
        Upperlim: Maximum size of data in each class for Training
        Imb_size: If the size of each class is less than Imb_size, then perform coarsening
        Model_Selec: Perform hyperparameter optimization (1=Yes, 0=No)
        numBorderPoints: Number of border points to use in refinement
        coarse: Indicator for whether or not at coarsest level
        loss: Loss function to use, "focal" or "cross"
        Level_size: # of levels. For keeping track of depth of coarsening
        refineMethod: Which refinement method to use, "border" or "flip"
        epochs: Number of epochs to train neural network for
        patience_level: Patience for refinement - if no improvement after n levels, stop refining and return best results
        weights: Whether or not to weight the loss function
        label: Label to give results file. 
    """

    data = pd.read_csv(data)
    """
    # Debug code:
    from sklearn.datasets import make_classification
    features, target = make_classification(n_samples=100,
                                           n_features=3,
                                           n_informative=3,
                                           n_redundant=0,
                                           n_classes=2,
                                           #weights=[0.9, 0.1],
                                           random_state=42)

    data = pd.DataFrame(features)
    data["Label"] = target
    """

    Results = []
    totalTime = []
    Best = {}

    # Model Training Options
    options = {"n_neighbors": n_neighbors, "Upperlim": Upperlim, "Model_Selec": Model_Selec, "Imb_size": Imb_size,
               "loss": loss, "alpha": alpha, "gamma": gamma, "numBorderPoints": numBorderPoints, "Coarsening": coarsening,
               "Q": Q, "r": r, "eta":eta, "refineMethod": refineMethod, "hb_epochs":hb_epochs,"hyperband_ite":hyperband_ite,
               "epochs": epochs, "batch_size": batch_size, "Dropout": Dropout,
               "BatchNorm": batchnorm,  "factor": factor, "patienceLevel": patienceLevel, "weights": weights,
               "dataName": dataName, "max_ite": max_ite, "multilevel": multilevel, "neural_earlyStop": neural_earlyStop}

    num_workers = 8
    pool = mp.Pool(num_workers)

    for ite in range(1, max_ite+1):
        start = time.time()

        # To make sure not re-running the same neural network
        tf.keras.backend.clear_session()

        # Create train/test data
        traindata, testdata = train_test_split(data, stratify=data["Label"], train_size=prop, random_state=42)

        # Create train/validation data using the train dataset
        traindata, valdata = train_test_split(traindata, stratify=traindata["Label"], train_size=prop, random_state=42)

        # Validation is kept separate for determining when to stop refinement
        val_lbl = valdata["Label"]
        valdata = valdata.drop(["Label"], axis=1)

        # Testing data is only used at the end of the algorithm to estimate final performance
        test_lbl = testdata["Label"]
        testdata = testdata.drop(["Label"], axis=1)


        if multilevel == 1:

            Ptraindata = traindata[traindata["Label"] == 1]
            Ntraindata = traindata[traindata["Label"] == 0]

            Ptraindata.reset_index(drop=True, inplace=True)
            Ntraindata.reset_index(drop=True, inplace=True)

            Ptrainlbl = Ptraindata["Label"]
            Ntrainlbl = Ntraindata["Label"]

            Ptraindata = Ptraindata.drop(["Label"], axis=1)
            Ntraindata = Ntraindata.drop(["Label"], axis=1)

            # Training data used separately
            train_lbl = traindata["Label"]
            traindata = traindata.drop(["Label"], axis=1)

            nNeighbors, ndistances = NearestNeighborSearch(Ntraindata, n_neighbors)
            pNeighbors, pdistances = NearestNeighborSearch(Ptraindata, n_neighbors)

            # Put everything into a dictionary to keep all relevant info together
            negativeData = {"Data": Ntraindata, "Labels": Ntrainlbl, "KNeighbors": nNeighbors, "Weights": ndistances, "WA": None}
            positiveData = {"Data": Ptraindata, "Labels": Ptrainlbl, "KNeighbors": pNeighbors, "Weights": pdistances, "WA": None}

            level = 0

            if options["Coarsening"] == "AMG":
                model, posBorderData, negBorderData, max_Depth, options, Best, flag, Level_results, level =\
                    MLD_AMG(run, traindata, train_lbl, valdata, val_lbl, level, negativeData, positiveData,
                        Best=Best, options=options, pool=pool)
            else:
                model, posBorderData, negBorderData, max_Depth, options, Best, flag, Level_results, level = \
                    MLD(run, traindata, train_lbl, valdata, val_lbl, level, negativeData, positiveData,
                            Best=Best, options=options, pool=pool)

            history = model.history
            formatFilename = "models/%s/best/"
            filename = formatFilename % (options["dataName"])

            bestModel = keras.models.load_model(filename, compile=False,  custom_objects={"recall_m": recall_m,
                                                                          "precision_m": precision_m,
                                                                          "specificity_m": specificity_m,
                                                                          "gmean_m": gmean_m})


            res = Evaluate(bestModel, testdata, test_lbl)
            Results.append(res)

            del bestModel, posBorderData, negBorderData, flag, Level_results, level

        else:

            train_lbl = traindata["Label"]
            traindata = traindata.drop(["Label"], axis=1)

            model = neuralNetwork(run, traindata, train_lbl, valdata, val_lbl, options)

            history = model.history

            res = Evaluate(model, testdata, test_lbl)
            Results.append(res)
            Best['difference'] = 0
            max_Depth = 0

        end = time.time()
        totalTime.append(end-start)

    pool.close()
    pool.join()

    """
    for epoch in range(len(history.epoch)):
        run["train/auc"].append(history.history['auc'][epoch])
        run["train/loss"].append(history.history['loss'][epoch])
        run["train/recall"].append(history.history['recall'][epoch])
        run["train/accuracy"].append(history.history['accuracy'][epoch])
        run["train/precision"].append(history.history['precision'][epoch])

        run["val/auc"].append(history.history['val_auc'][epoch])
        run["val/loss"].append(history.history['val_loss'][epoch])
        run["val/recall"].append(history.history['val_recall'][epoch])
        run["val/accuracy"].append(history.history['val_accuracy'][epoch])
        run["val/precision"].append(history.history['val_precision'][epoch])
    """

    allResults = pd.DataFrame(Results)

    allResults["Time"] = totalTime

    formatDirect = "Results/%s/" % options["dataName"]
    pathlib.Path(formatDirect).mkdir(parents=True, exist_ok=True)
    formatFilename = "Multilevel_%depochs%dRefine%sBorderPoints%dUpperLim%dLoss%s%smaxIte%d_allResults.csv"
    filename = formatFilename % (multilevel, epochs, refineMethod, numBorderPoints, Upperlim, loss, label, max_ite)
    filename = formatDirect + filename
    allResults.to_csv(filename)

    run["allResults"].upload(filename)

    Results = averageResults(Results)
    aveRefine = np.mean(Best['difference'])
    aveCoarsenDepth = np.mean(max_Depth)
    averageTime = np.mean(totalTime)     
    # Save results into an excel file

    resultsTable = pd.DataFrame({
        'GMean': [Results["GMean"]],
        'AUC': [Results["AUC"]],
        'Precision': [Results["Precision"]],
        'Acc': [Results["Acc"]],
        'Recall': [Results["Recall"]],
        'Spec': [Results["Spec"]],
        'F1': [Results["F1"]],
        'Time (sec)': [averageTime],
        'Refine Level': [aveRefine],
        'Average Max Depth': [aveCoarsenDepth],
    }, columns=['GMean','AUC', 'Precision', 'Acc', 'Recall', 'Spec','F1', 'Time (sec)', 'Refine Level', 'Average Max Depth'])

    formatDirect = "Results/%s/" % options["dataName"]
    pathlib.Path(formatDirect).mkdir(parents=True, exist_ok=True)

    formatFilename = "Multilevel_%depochs%dRefine%sBorderPoints%dNeighbors%dLoss%s%smaxIte%d.xlsx"
    filename = formatFilename % (multilevel, epochs, refineMethod, numBorderPoints, n_neighbors, loss, label, max_ite)
    filename = formatDirect + filename

    resultsTable.to_excel(filename, sheet_name='Sheet1', index=False)

    run["AUC"] = resultsTable["AUC"]
    run["Precision"] = resultsTable["Precision"]
    run["GMean"] = resultsTable["GMean"]
    run["Acc"] = resultsTable["Acc"]
    run["Recall"] = resultsTable["Recall"]
    run["Spec"] = resultsTable["Spec"]
    run["Time (sec)"] = resultsTable["Time (sec)"]
    run["Refine Level"] = resultsTable["Refine Level"]
    run["Average Max Depth"] = resultsTable["Average Max Depth"]

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='segment', fromfile_prefix_chars='@')


    # Data
    parser.add_argument('--data', type=str, default='../Data/Hypothyroid.csv', help='Data filepath')
    parser.add_argument('--dataName', type=str, default='Hypothyroid', help='Name of data being tested')
    parser.add_argument('--max_ite', type=int, default=1, help='Number of iterations to average over')
    parser.add_argument('--label', type=str, default=None, help='Label to add to results')

    # Multilevel Parameters
    parser.add_argument('--Imb_size', type=int, default=300, help='Imbalance Size')
    parser.add_argument('--Coarsening', type=str, default="AMG", help='Coarsening method')

    parser.add_argument('--multilevel', type=int, default=1, help='Whether to run multilevel version of model or not')
    parser.add_argument('--n_neighbors', type=int, default=10, help='Number of nearest neighbors to consider')
    parser.add_argument('--Upperlim', type=int, default=500, help='Maximum number of points to consider at coarsest '
                                                                  'level')
    parser.add_argument('--numBorderPoints', type=int, default=10, help='Number of border points to consider in refine')
    parser.add_argument('--refineMethod', type=str, default="flip", help='Refinement method. "Border" or "flip"')
    parser.add_argument('--patienceLevel', type=int, default=3, help='Max refinement without improvement before stopping')

    # AMG Coarsening parameters
    parser.add_argument('--Q', type=float, default=0.4, help='Fraction of points to consider as secondary seeds')
    parser.add_argument('--r', type=int, default=1, help='Interpolation complexity')
    parser.add_argument('--eta', type=int, default=1, help='How much above average volume to be in seed set')

    # Neural Network
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--loss', type=str, default='cross', help='Loss function. Can be "cross" or "focal"')
    parser.add_argument('--alpha', type=float, default=0.5, help='Focal loss alpha value')
    parser.add_argument('--gamma', type=float, default=4, help='Focal loss gamma value')
    parser.add_argument('--Dropout', type=float, default=0.3, help='Dropout Rate')
    parser.add_argument('--batchnorm', type=float, default=0.2, help='Batch Normalization Rate')
    parser.add_argument('--Model_Selec', type=int, default=1, help='Perform Hyperparameter Tuning?')
    parser.add_argument('--factor', type=int, default=3, help='Hyperband reduction factor')
    parser.add_argument('--neural_earlyStop', type=int, default=None, help='Early stopping for neural networks')
    parser.add_argument('--prop', type=float, default=0.8, help='Proportion of data to use for training/val')

    # Hyperparameter Tuner
    parser.add_argument('--hb_epochs', type=int, default=None, help='Max epochs to train in hb')
    parser.add_argument('--hyperband_ite', type=int, default=1, help='Number of times to run the hb algorithm')

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')

    return parser


if __name__ == "__main__":

    run = neptune.init_run(
        project="rachellb/Multilevel",
        api_token=api_,
        tags=["Post-fix", "Refine on All"])

    enable_tensorboard_logging(run)

    parser = create_parser()
    args = parser.parse_args()

    params = {'Name': args.dataName,
              'Multilevel': args.multilevel,
              'n_neighbors': args.n_neighbors,
              'Epochs': args.epochs,
              'Upperlim': args.Upperlim,
              'BorderPoints': args.numBorderPoints,
              'Loss': args.loss,
              'Batch_size': args.batch_size,
              "Dropout": args.Dropout,
              "BatchNorm": args.batchnorm,
              "PatienceLevel": args.patienceLevel,
              "refineMethod": args.refineMethod,
              "Coarsening": args.Coarsening,
              "max_ite": args.max_ite,
              "Tuning": args.Model_Selec,
              "prop": args.prop,
              "hb_epochs": args.hb_epochs,
              "hyperband_ite": args.hyperband_ite,
              "Q": args.Q,
              "r": args.r,
              "eta":args.eta}

    Multilevel(run, data=args.data, dataName=args.dataName, prop=args.prop, neural_earlyStop=args.neural_earlyStop,
               multilevel=args.multilevel, n_neighbors=args.n_neighbors, Q=args.Q, r=args.r, eta=args.eta,
               epochs=args.epochs, Upperlim=args.Upperlim, max_ite=args.max_ite, Imb_size=args.Imb_size,
               Model_Selec=args.Model_Selec, numBorderPoints=args.numBorderPoints, loss=args.loss, coarsening=args.Coarsening,
               alpha=args.alpha, gamma=args.gamma,hb_epochs=args.hb_epochs, hyperband_ite=args.hyperband_ite, refineMethod=args.refineMethod, label=args.label,  batch_size=args.batch_size, Dropout=args.Dropout,
               batchnorm=args.batchnorm, factor=args.factor, patienceLevel=args.patienceLevel)

    run["parameters"] = params

    run.stop()
