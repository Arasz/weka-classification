package put.cs.idss.ml.weka.Errors;

import weka.classifiers.Classifier;
import weka.core.Instance;

/**
 * Created by Rafal on 12.01.2017.
 */
public class ZeroOneLossError extends LearningErrorFunctionBase {
    @Override
    public String getErrorName() {
        return "0/1 loss";
    }

    protected double getCorrectValue(Classifier classifier, Instance testInstance){
        return testInstance.classValue();
    }

    @Override
    protected double getPredictedValue(Classifier classifier, Instance testInstance) throws Exception {
        double[] distribution = classifier.distributionForInstance(testInstance);
        return  distribution[1] >= distribution[0] ? 1 : 0;
    }

    @Override
    public double calculateError(double realValue, double predictedValue) {
        int predicted = (int) Math.round(predictedValue);
        int real = (int) Math.round(realValue);
        return real == predicted ? 0 : 1;
    }
}
