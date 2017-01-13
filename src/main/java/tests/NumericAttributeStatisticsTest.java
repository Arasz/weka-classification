package tests;

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import put.cs.idss.ml.weka.AttributeStattistics.AttributeStatistics;
import put.cs.idss.ml.weka.Experiment;
import put.cs.idss.ml.weka.AttributeStattistics.NumericAttributeStatistics;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.Enumeration;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Created by Rafal on 12.01.2017.
 */
public class NumericAttributeStatisticsTest {

    private static Attribute attribute;
    private static String testDataSetPath = "data/test-numeric.arff";
    private static Instances instances;


    private static Attribute findNumericAttribute() throws Exception {
        Enumeration attributes = instances.enumerateAttributes();
        while (attributes.hasMoreElements())
        {
            Attribute attribute = (Attribute) attributes.nextElement();
            if(attribute.isNumeric())
                return attribute;
        }
        throw new Exception("Numeric attribute can not be found.");
    }

    @BeforeAll
    public static void setUp() throws Exception {
        instances = Experiment.LoadData(testDataSetPath);
        Experiment.ConfigureClassIndex(instances);
        attribute = findNumericAttribute();
    }

    @Test
    public void CalculateNumericStatistics_CheckMeanAndStdValue_ShouldBeCorrectlyCalculated(){
        int expectedLength = 2;
        double meanValue = 0.35796703852618439;
        double stdValue = 0.262253062899326;
        int classValue = 0;

        AttributeStatistics attributeStatistic = new NumericAttributeStatistics(classValue, attribute, instances);

        attributeStatistic.calculate();
        double[] statistics = attributeStatistic.getStatistics();

        assertThat(statistics.length)
                .isEqualTo(expectedLength);

        assertThat(statistics[0])
                .isCloseTo(meanValue, Offset.offset(0.01));

        assertThat(statistics[1])
                .isCloseTo(stdValue, Offset.offset(0.01));


    }

}
