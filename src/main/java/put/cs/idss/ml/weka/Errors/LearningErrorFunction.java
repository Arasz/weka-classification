package put.cs.idss.ml.weka.Errors;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Created by Rafal on 12.01.2017.
 */
public interface LearningErrorFunction {
    String getErrorName();

    double getError(String classifierName);

    void calculate(Classifier classifier, Instances testInstances, Instances trainInstances) throws Exception;
}
