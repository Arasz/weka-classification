package put.cs.idss.ml.weka;

import weka.core.Attribute;
import weka.core.Instances;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by Rafal on 12.01.2017.
 */
public class NumericAttributeStatistics implements AttributeStatistics {

    private Attribute attribute;
    private Instances instances;
    private double[] statistics;
    private double classValue;

    private static final int  distributionParameters= 2;

    private static final int  meanParameter= 0;

    private static final int  stdParameter= 1;

    public NumericAttributeStatistics( int classValue, Attribute attribute, Instances instances){
        this.classValue = classValue;
        this.attribute = attribute;
        this.instances = instances;
    }

    @Override
    public void calculate() {
        double[] attributeValues = instances.attributeToDoubleArray(attribute.index());
        double[] classValues = instances.attributeToDoubleArray(instances.classIndex());

        statistics = new double[distributionParameters];

        int[] indices = getIndicesForClassValue(classValues);

        statistics[meanParameter] = calculateMean(attributeValues, indices);

        statistics[stdParameter] = calculateStd(attributeValues, indices, statistics[meanParameter]);

    }



    private int[] getIndicesForClassValue(double[] classValues){
        return IntStream.range(0, classValues.length)
                .filter(index -> classValues[index] == classValue)
                .toArray();
    }

    private double calculateMean(double[] attributeValues, int[] indices) {
        return Arrays.stream(indices)
                .mapToDouble(index -> attributeValues[index])
                .average()
                .getAsDouble();

    }
    private double calculateStd(double[] attributeValues, int[] indices, double mean) {
        return Math.sqrt(
                Arrays.stream(indices)
                .mapToDouble(index -> Math.pow(attributeValues[index] - mean, 2))
                .sum()/((double) indices.length-1));
    }


    @Override
    public double[] getStatistics() {
        if(statistics == null)
            calculate();
        return statistics;
    }
}
