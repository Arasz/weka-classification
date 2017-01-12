package put.cs.idss.ml.weka.Errors;

import weka.classifiers.Classifier;
import weka.core.Instance;

/**
 * Created by Rafal on 12.01.2017.
 */
public class SquaredErrorFunction extends LearningErrorFunctionBase {

    @Override
    public String getErrorName() {
        return "Squared error";
    }

    @Override
    protected double getCorrectValue(Classifier classifier, Instance testInstance) {
        return correctValueProbability;
    }
    @Override
    protected double getPredictedValue(Classifier classifier, Instance testInstance) throws Exception {
        double[] distribution = classifier.distributionForInstance(testInstance);
        return distribution[(int) testInstance.classValue()];
    }

    @Override
    public double calculateError(double realValue, double predictedValue)
    {
        return Math.pow(realValue - predictedValue, 2);
    }

}
