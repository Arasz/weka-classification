package put.cs.idss.ml.weka.AttributeStattistics;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by Rafal on 12.01.2017.
 */
public class ClassStatistics implements AttributeStatistics {
    private Attribute attribute;
    private Instances instances;

    private double[] statistics;

    public ClassStatistics( Attribute attribute, Instances instances){
        this.attribute = attribute;
        this.instances = instances;
    }

    public void calculate(){
        AttributeStats attributeStats = instances.attributeStats(attribute.index());
        int allValuesCount = attributeStats.totalCount;
        int distinctValuesCount = attributeStats.distinctCount;

        double[] attributeValues = instances.attributeToDoubleArray(attribute.index());

        double[] distinctValues = Arrays.stream(attributeValues)
                .distinct()
                .sorted()
                .toArray();

        statistics = new double[distinctValuesCount];

        for (int i = 0; i< distinctValuesCount ; i++){
            int finalI = i;
            statistics[i] = (double) IntStream.range(0, attributeValues.length)
                    .filter(index -> attributeValues[index] == distinctValues[finalI])
                    .count() / allValuesCount;
        }
    }

    public double[] getStatistics()
    {
        return statistics;
    }
}
