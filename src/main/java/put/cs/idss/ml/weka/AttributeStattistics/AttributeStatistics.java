package put.cs.idss.ml.weka.AttributeStattistics;

/**
 * Created by Rafal on 12.01.2017.
 */
public interface AttributeStatistics {

    void calculate();

    double[] getStatistics();
}
