package tests;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import put.cs.idss.ml.weka.Experiment;
import put.cs.idss.ml.weka.AttributeStattistics.NominalAttributeStatistic;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.Enumeration;

import static org.assertj.core.api.Assertions.*;


public class NominalAttributeStatisticTest{

    private static Attribute attribute;
    private static String testDataSetPath = "data/grypa-train.arff";
    private static Instances instances;


    private static Attribute findNominalAttribute() throws Exception {
        Enumeration attributes = instances.enumerateAttributes();
        while (attributes.hasMoreElements())
        {
            Attribute attribute = (Attribute) attributes.nextElement();
            if(attribute.isNominal())
                return attribute;
        }
        throw new Exception("Nominal attribute can not be found.");
    }

    @BeforeAll
    public static void setUp() throws Exception {
        instances = Experiment.LoadData(testDataSetPath);
        Experiment.ConfigureClassIndex(instances);
        attribute = findNominalAttribute();
    }

    @Test
    public void CalculateStatistics_CheckCalculatedStatistics_ShouldCorrectlyCalculateStatistics(){
        int expectedLength = 2;
        double[] expectedStats = new double[]{3d/5d, 2d/5d};
        int classValue = 1;

        NominalAttributeStatistic attributeStatistic = new NominalAttributeStatistic(classValue, attribute, instances);

        attributeStatistic.calculate();
        double[] statistics = attributeStatistic.getStatistics();

        assertThat(statistics.length)
                .isEqualTo(2);
        assertThat(statistics)
                .containsExactlyInAnyOrder(expectedStats);


    }

}