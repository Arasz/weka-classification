package put.cs.idss.ml.weka;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

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

	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		numClasses = data.numClasses();
		
		// remove instances with missing class
		data.deleteWithMissingClass();

		/* 1. Initialize arrays of counts for nominal attributes, 
		 * means and std.devs. for numeric attributes,
		 * and a priori probabilities of the classes. */
		for(int i = 0; i < data.numAttributes() - 1; i++) {
			Attribute attribute = data.attribute(i);
			for(int j = 0; j < data.numClasses(); j++) {
				if(attribute.isNominal()) {
					//counts[j][i] = ...
				} else {
					//counts[j][i] = ...
				}
			}
		}
		
		// 2. compute counts and sums.
		for(int i = 0; i < data.numInstances(); i++) {
			Instance instance = data.instance(i);
			int classValue = (int) instance.classValue();
			for(int j = 0; j < data.numAttributes() - 1; j++) {
				if(data.attribute(j).isNominal()) {
					// ...
				} else {
					// ...
				}
			}
			// ...
		}
		
		// 3. Compute means.
		
		// 4. Compute standard deviations.
		
		// 5. normalize counts and a priori probabilities
	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] distribution = new double[numClasses];
		
		// Your code :)
		
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
