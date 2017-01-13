package put.cs.idss.ml.weka.Comparator;

import put.cs.idss.ml.weka.Errors.LearningErrorFunction;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.HashMap;
import java.util.List;

/**
 * Created by Rafal on 12.01.2017.
 */
public class ClassifiersComparator {

    private List<Classifier> classifiers;
    private Instances trainInstances;
    private Instances testInstances;
    private SimpleTimer timer = new SimpleTimer();

    private HashMap<String, Long> trainingTimes = new HashMap<>();
    private HashMap<String, Long> testingTimes = new HashMap<>();

    private String getClassifierName(Classifier classifier){
        return classifier.getClass().getSimpleName();
    }

    public ClassifiersComparator(List<Classifier> classifiers, Instances trainInstances, Instances testInstances){

        this.classifiers = classifiers;
        this.trainInstances = trainInstances;
        this.testInstances = testInstances;
    }

    public void trainClassifiers() throws Exception {
        for(Classifier classifier : classifiers){
            timer.start();
            classifier.buildClassifier(trainInstances);
            logTrainingTime(classifier);

        }
    }
    private void logTestingTime(Classifier classifier){
        timer.stop();
        testingTimes.put(getClassifierName(classifier), timer.elapsed);
        timer.reset();
    }

    private void logTrainingTime(Classifier classifier){
        trainingTimes.put(getClassifierName(classifier), timer.elapsed);
        timer.reset();
    }


    public void compare(List<LearningErrorFunction> errorFunctions) throws Exception {
        trainClassifiers();

        for (Classifier classifier : classifiers){
            timer.start();
            for (LearningErrorFunction errorFunction : errorFunctions){
                errorFunction.calculate(classifier, testInstances, trainInstances);
            }
            logTestingTime(classifier);
        }
    }

}
