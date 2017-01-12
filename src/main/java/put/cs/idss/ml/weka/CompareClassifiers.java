package put.cs.idss.ml.weka;

import put.cs.idss.ml.weka.Comparator.ClassifiersComparator;
import put.cs.idss.ml.weka.Errors.LearningErrorFunction;
import put.cs.idss.ml.weka.Errors.SquaredErrorFunction;
import put.cs.idss.ml.weka.Errors.ZeroOneLossError;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Author: NOT ME
 */
public class CompareClassifiers {

    public CompareClassifiers() {
        randomGenerator = new Random();
    }


    public static double roundDouble(double number, int decimalPlaces) {
        if (decimalPlaces < 0) throw new IllegalArgumentException();

        BigDecimal bigDecimal = new BigDecimal(number);
        bigDecimal = bigDecimal.setScale(decimalPlaces, RoundingMode.HALF_UP);
        return bigDecimal.doubleValue();
    }

    private static String dataPath = "data/";
    private static String trainDataAppendix = "-train.arff";
    private static String testDataAppendix = "-test.arff";

    private static Instances loadData(String path) throws IOException {
        return Experiment.LoadData(path);
    }

    private static Instances getDataPart(double instancesPercent, Instances instances){
        int partSize = (int)((double)instances.numInstances() * instancesPercent);
        return new Instances(instances, 0, partSize);
    }

    private static Random randomGenerator = new Random();

    private static Instances randomize(Instances instances){
        instances.randomize(randomGenerator);
        return instances;
    }

    private static String combinePath(String datasetName, boolean isTrain){
        if(isTrain)
            return dataPath+datasetName+trainDataAppendix;
        else
            return dataPath+datasetName+testDataAppendix;
    }

    private static HashMap<String, String> reports = new HashMap<>();

    private static void compareForDifferentSize(double[] datasetPercents, String[] datasetNames, List<Classifier> classifiers) throws Exception {


        for (String datasetName : datasetNames){

            for (Classifier classifier : classifiers){
                String name = getClassifierName(classifier);

                if(!reports.containsKey(name))
                    reports.put(name,"\nClassifier: "+name+"\nsamples, 0/1 loss, squared\n");

                AppendToReport(datasetName, name);
                AppendToReport("Test na danych testowych", name);
            }

            datasetPrecentLoop(datasetPercents, classifiers, datasetName, true);

            for (Classifier classifier : classifiers){
                String name = getClassifierName(classifier);

                AppendToReport(datasetName, name);
                AppendToReport("Test na danych treningowych", name);
            }

            datasetPrecentLoop(datasetPercents, classifiers, datasetName, false);
        }

        for (Map.Entry<String, String> report:reports.entrySet()){
            File file = new File("reports/"+report.getKey()+".txt");
            file.createNewFile();
            try (BufferedWriter writer = Files.newBufferedWriter(file.toPath())) {
                    writer.write(report.getValue());
            }
        }
    }

    private static void AppendToReport(String appended, String name) {
        String report = reports.get(name);
        report += "\n"+appended+"\n";
        reports.put(name, report);
    }

    private static void datasetPrecentLoop(double[] datasetPercents, List<Classifier> classifiers, String datasetName, boolean testOnTest) throws Exception {
        for (double datasetPercent : datasetPercents){
            Instances trainInstances = getDataPart(datasetPercent, randomize(loadData(combinePath(datasetName, true))));
            Instances testInstances = randomize(loadData(combinePath(datasetName, false)));
            Experiment.ConfigureClassIndex(trainInstances);
            Experiment.ConfigureClassIndex(testInstances);


                ClassifiersComparator comparator = new ClassifiersComparator(classifiers, trainInstances, testOnTest ? testInstances : trainInstances);

                List<LearningErrorFunction> errorFunctions = new ArrayList<>();
                LearningErrorFunction zeroOneLoss =new ZeroOneLossError();
                LearningErrorFunction squared = new SquaredErrorFunction();
                errorFunctions.add(zeroOneLoss);
                errorFunctions.add(squared);

                comparator.compare(errorFunctions);


                for (Classifier classifier : classifiers){
                    String name = getClassifierName(classifier);
                    String report = reports.get(name);
                    report+= format(datasetPercent,2)+
                            " , "+format(zeroOneLoss.getError(name),4)+
                            " , "+format(squared.getError(name),4)+
                            "\n";
                    reports.put(name, report);
                }
        }
    }

    private static String format(double number, int decimalPlaces){
        return String.format("%-8."+decimalPlaces+"f",roundDouble(number, decimalPlaces));
    }

    private static String getClassifierName(Classifier classifier) {
        return classifier.getClass().getSimpleName();
    }

    public static void main(String[] args) throws Exception {
        ArrayList<Classifier> classifiers = createClassifiersToCompare();


        compareForDifferentSize(IntStream.range(1,100).mapToDouble(value -> (double)value/100d).toArray(), new String[]{"credit-a-mod", "credit-a", "spambase", "badges2"}, classifiers);

    }

    private static ArrayList<Classifier> createClassifiersToCompare() {
        ArrayList<Classifier> classifiers = new ArrayList<>();
        classifiers.add(new weka.classifiers.bayes.NaiveBayes());
        classifiers.add(new weka.classifiers.functions.Logistic());
        return classifiers;
    }
}
