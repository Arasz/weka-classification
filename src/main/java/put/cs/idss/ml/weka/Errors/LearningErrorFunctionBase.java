package put.cs.idss.ml.weka.Errors;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;

/**
 * Created by Rafal on 12.01.2017.
 */
public abstract class LearningErrorFunctionBase implements LearningErrorFunction
{
    protected HashMap<String, Double> error = new HashMap<>();

    protected double correctValueProbability = 1.0;

    @Override
    public double getError(String classifierName) {
        return error.get(classifierName);
    }

    public abstract double calculateError(double realValue, double predictedValue);

    protected abstract double getCorrectValue(Classifier classifier, Instance testInstance);
    protected abstract double getPredictedValue(Classifier classifier, Instance testInstance) throws Exception;

    public void calculate(Classifier classifier, Instances testInstances, Instances trainInstances) throws Exception {
        int testDataLength = testInstances.numInstances();

        double localError = 0;

        for (int instanceIndex = 0 ; instanceIndex < testDataLength ; instanceIndex++){

            Instance testInstance = testInstances.instance(instanceIndex);

            double predictedValue = getPredictedValue(classifier, testInstance);
            double correctValue = getCorrectValue(classifier, testInstance);


            localError += calculateError(correctValue, predictedValue);
        }

        error.put(classifier.getClass().getSimpleName(), localError/(double) testDataLength);
    }
}
