package put.cs.idss.ml.weka;

        import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class CompareClassifiers {

    public CompareClassifiers() {
        // TODO Auto-generated constructor stub
    }

    public static double roundDouble(double x, int n) {
        String s = "#.";
        for (int i = 0; i < n; i++) {
            s += "#";
        }
        DecimalFormat twoDForm = new DecimalFormat(s);
        return Double.parseDouble(twoDForm.format(x));
    }

    public static void main(String[] args) throws Exception {
        //classifiers that we want to compare
        ArrayList<Classifier> classifiers = new ArrayList<>();
        classifiers.add(new weka.classifiers.bayes.NaiveBayes());
        classifiers.add(new weka.classifiers.functions.Logistic());

        String dataset = "badges2"; // badges2 / credit-a-mod / credit-a
        double partOfDataset = 1.0; // part of randomized train set (0 .. 1)
        long seed = 1;

        BufferedReader readerTrain = new BufferedReader(new FileReader("../../data/"+dataset+"-train.arff"));
        Instances trainSetTmp = new Instances(readerTrain);
        int newTrainSetSize = (int)((double)trainSetTmp.numInstances() * partOfDataset);
        trainSetTmp.randomize(new Random(seed));
        Instances trainSet = new Instances(trainSetTmp, 0, newTrainSetSize);
        readerTrain.close();

        BufferedReader readerTest = new BufferedReader(new FileReader("../../data/"+dataset+"-test.arff"));
        Instances testSet = new Instances(readerTest);
        testSet.randomize(new Random(seed));
        readerTest.close();

        if (trainSet.classIndex() == -1) trainSet.setClassIndex(trainSet.numAttributes() - 1);
        if (testSet.classIndex() == -1) testSet.setClassIndex(testSet.numAttributes() - 1);

        System.out.println("Data loaded and randomized:");
        System.out.println(" - train set size: " + trainSet.numInstances());
        System.out.println(" - test set size:  " + testSet.numInstances());

        HashMap<String,Long> trainingTimes = new HashMap<>();
        HashMap<String,Long> testingTimes = new HashMap<>();

        for (Classifier classifier : classifiers) {
            String classifierName = classifier.getClass().getSimpleName();
            System.out.println("Training " + classifierName + "...");
            long trainingTimeStart = System.currentTimeMillis();
            classifier.buildClassifier(trainSet);
            long trainingTime = System.currentTimeMillis() - trainingTimeStart;
            System.out.println(" - training time: " + trainingTime);
            trainingTimes.put(classifierName, trainingTime);
        }

        HashMap<String,Double> loss01 = new HashMap<>();
        HashMap<String,Double> squaredError = new HashMap<>();

        double sum = testSet.numInstances();
        for (Classifier classifier : classifiers) {
            String classifierName = classifier.getClass().getSimpleName();
            System.out.println("Testing " + classifierName + "...");
            long testingTimeStart = System.currentTimeMillis();
            for (int i = 0; i < testSet.numInstances(); i++) {
                Instance instance = testSet.instance(i);
                int truth = (int) instance.classValue();

                double[] distribution = classifier.distributionForInstance(instance);
                int prediction = distribution[1] >= distribution[0] ? 1 : 0;

                double _loss01 = truth == prediction ? 0 : 1;
                double _squaredError = Math.pow(1.0 - distribution[truth], 2);

                if(loss01.containsKey(classifierName)) {
                    _loss01 += loss01.get(classifierName);
                    _squaredError += squaredError.get(classifierName);
                }
                loss01.put(classifierName, _loss01);
                squaredError.put(classifierName, _squaredError);
            }
            long testingTime = System.currentTimeMillis() - testingTimeStart;
            testingTimes.put(classifierName, testingTime);
        }

        System.out.println("\nRESULTS:\n");
        for (Classifier classifier : classifiers) {
            String classifierName = classifier.getClass().getSimpleName();
            long trainingTime = trainingTimes.get(classifierName);
            long testingTime = testingTimes.get(classifierName);
            double _loss01 = loss01.get(classifierName) / sum;
            double _squaredError = squaredError.get(classifierName) / sum;
            System.out.println(classifierName + " :");
            System.out.println(" - training time:  " + trainingTime);
            System.out.println(" - testing time:   " + testingTime);
            System.out.println(" - 0/1 loss:       " + roundDouble(_loss01, 4));
            System.out.println(" - squared-error:  " + roundDouble(_squaredError, 4));
            System.out.println();
        }
    }
}
