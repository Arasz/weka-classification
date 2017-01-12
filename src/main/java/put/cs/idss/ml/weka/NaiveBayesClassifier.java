package put.cs.idss.ml.weka;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class NaiveBayesClassifier extends Classifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 7550409893545527343L;

	/** number of classes */
	protected int numClasses;

	/** counts, means, standard deviations, priors..... */
	//protected double[.....

	public NaiveBayesClassifier() {
		// TODO Auto-generated constructor stub
	}

	private AttributeStatistics attributeStatistics[][];
	private double[] classesStatistics;



	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		numClasses = data.numClasses();
		
		// remove instances with missing class
		data.deleteWithMissingClass();

		initializeDataStatistics(data);

		calculateDataStatistics(data);

	}

    private void initializeDataStatistics(Instances data)
    {
        int numClasses = data.numClasses();
        int numAttributes = data.numAttributes();

        attributeStatistics = new AttributeStatistics[numClasses][numAttributes];
        classesStatistics = new double[numClasses];
    }

	private void calculateDataStatistics(Instances data) {
		calculateAttributesStatistics(data); // P(x|y)
		calculateClassesStatistics(data); // P(y)
	}

	private void calculateClassesStatistics(Instances data) {

		AttributeStatistics classesStats = new ClassStatistics(data.classAttribute(), data);
		classesStats.calculate();
		classesStatistics = classesStats.getStatistics();
	}

	private void calculateAttributesStatistics(Instances data) {
		for(int attributeIndex = 0; attributeIndex < data.numAttributes() - 1; attributeIndex++) {

			Attribute attribute = data.attribute(attributeIndex);

			for(int classIndex = 0; classIndex < data.numClasses(); classIndex++) {
				if(attribute.isNominal())
				{
                    attributeStatistics[classIndex][attributeIndex] = new NominalAttributeStatistic(classIndex, attribute, data);
                } else {
                    attributeStatistics[classIndex][attributeIndex] = new NumericAttributeStatistics(classIndex, attribute, data);
				}
			}
		}
	}

	private double calculateNormalizationFactorForInstance(Instance instance)
	{
		//P(X)
		double normalizationFactor = 0;
		for (int classValue = 0 ; classValue < numClasses ; classValue++){
			normalizationFactor+= calculatePrioriProbabilityForInstance(instance, classValue)*classesStatistics[classValue];
		}
		return normalizationFactor;
	}

	private double calculatePrioriProbabilityForInstance(Instance instance, int classValue){
		double probability = 1;
		for (int attributeIndex = 0; attributeIndex< instance.numAttributes()-1; attributeIndex++){
			double attributeValue = instance.toDoubleArray()[attributeIndex];

			if(instance.attribute(attributeIndex).isNominal())
				probability *= getNominalAttributeProbability(classValue, attributeIndex, (int) attributeValue);
			else {
				probability *= getNumericAttributeProbability(classValue, attributeIndex, attributeValue);
			}
		}
		return probability;
	}




	private double getNumericAttributeProbability(int classValue, int attributeIndex, double attributeValue) {
		double[] stats = attributeStatistics[classValue][attributeIndex].getStatistics();
		double mean = stats[0];
		double std = stats[1];
		return normalDistributionProbabilityDensityFunction(attributeValue, mean, std);
	}

	private double normalDistributionProbabilityDensityFunction(double value, double mean, double std)
	{
		double factor = 1/(std * Math.sqrt(Math.PI));
		double expArgument = (-Math.pow((value-mean), 2))/(2*Math.pow(std,2));
		return factor * Math.exp(expArgument);
	}

	private double getNominalAttributeProbability(int classValue, int attributeIndex, int attributeValue) {
		return attributeStatistics[classValue][attributeIndex].getStatistics()[attributeValue];
	}


	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] distribution = new double[numClasses];
		double normalizationFactor = calculateNormalizationFactorForInstance(instance);

		for (int classValue = 0; classValue< numClasses ; classValue++) {

			double aPriori = calculatePrioriProbabilityForInstance(instance, classValue);
			double classProbability = classesStatistics[classValue];
			distribution[classValue] = (aPriori*classProbability)/normalizationFactor;
		}
		// Remember to normalize probabilities!
		
		return distribution;    
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double classValue = 0.0;
		double max = Double.MIN_VALUE;
		double[] dist = distributionForInstance(instance);
		
		for(int i = 0; i < dist.length; i++) {
			if(dist[i] > max) {
				classValue = i;
				max = dist[i];
			}
		}
		
		return classValue;
	}

}
