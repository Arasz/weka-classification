package put.cs.idss.ml.weka;

import weka.attributeSelection.ClassifierSubsetEval;
import weka.attributeSelection.LinearForwardSelection;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.util.Random;

public class CrossValidationExperiment {
	
	
	public static void compareCrossValidationScenarios(Classifier classifier, Instances dataSet,
			int folds, int numTopAttributes, Random random) throws Exception {
		
		if (dataSet.classIndex() == -1) dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		Instances dataSet2 = new Instances(dataSet);
		dataSet2.randomize(random);
		
		ClassifierSubsetEval cse = new  ClassifierSubsetEval(); 
		cse.setClassifier(Classifier.makeCopy(classifier));
		cse.setUseTraining(true);
		
		WrapperSubsetEval wse = new WrapperSubsetEval();
		wse.setClassifier(Classifier.makeCopy(classifier));
		wse.setFolds(folds);
		
		LinearForwardSelection lfs = new LinearForwardSelection();
		lfs.setNumUsedAttributes(numTopAttributes);
		
		AttributeSelection as = new AttributeSelection();
		as.setInputFormat(dataSet);
		as.setEvaluator(cse);
		as.setSearch(lfs);
		
		Instances filteredInstances = Filter.useFilter(dataSet, as);
		
		//Scenario 1
		weka.classifiers.Evaluation eval = new Evaluation(filteredInstances);
		String[] options = {};
		eval.crossValidateModel(Classifier.makeCopy(classifier), filteredInstances, folds, random, options);
		System.out.println("Scenario 1:\n"+eval.toSummaryString()+"\n--------------------\n");
		System.out.println(eval.toMatrixString());
		
		//----------
		
		AttributeSelectedClassifier asc = new AttributeSelectedClassifier();
		WrapperSubsetEval wse2 = new WrapperSubsetEval();
		wse2.setClassifier(Classifier.makeCopy(classifier));
		
		LinearForwardSelection lfs2 = new LinearForwardSelection();
		lfs2.setNumUsedAttributes(numTopAttributes);
		
		asc.setSearch(lfs2);
		asc.setEvaluator(cse);
		asc.setClassifier(Classifier.makeCopy(classifier));
		
		//Scenario 2
		weka.classifiers.Evaluation eval2 = new Evaluation(dataSet2);
		eval2.crossValidateModel(Classifier.makeCopy(asc), dataSet2, folds, random, options);
		System.out.println("Scenario 2:\n"+eval2.toSummaryString());
		System.out.println(eval2.toMatrixString());
	}

	static public Instances generateRandomBinaryModel(String name, int numInstances, int numAttributes, Random random) {
		
		FastVector attributes = new FastVector();
		for(int i = 0; i < numAttributes - 1; i++) {
			Attribute attr = new Attribute("feature_"+(i+1));
			attributes.addElement(attr);
		}
		
		FastVector classValues = new FastVector();
		classValues.addElement("0");
		classValues.addElement("1");
		Attribute label = new Attribute("class", classValues);
		attributes.addElement(label);
		
		Instances instances = new Instances(name, attributes, 0);
		
		double[][] dataset = new double[numInstances][];
		
		for(int i = 0; i < numInstances; i++) {
			double[] x = new double[numAttributes - 1];
			for(int j = 0; j < numAttributes - 1; j++) {
				x[j] = 2 * random.nextDouble() - 1;
			}
			dataset[i] = x;
		}
		
		
		for(int i = 0; i < numInstances; i++) {
			Instance inst = new Instance(numAttributes);
			inst.setDataset(instances);
			
			double[] x = dataset[i];
			
			for(int j = 0; j < numAttributes - 1; j++) {
				inst.setValue(j, x[j]);
			}
			
			String classValue = "0";
			
			if(random.nextDouble() > 0.5)
				classValue = "1";
								
			inst.setValue(numAttributes - 1, classValue);
			instances.add(inst);
		}
		
		return instances;
	}

	
	public static void main(String[] args) throws Exception {
		
		Logistic classifier = new Logistic();
		classifier.setRidge(0.00001);
		
		Random random = new Random();
		
		Instances dataSet = generateRandomBinaryModel("dataset", 100, 100, random);
		
		compareCrossValidationScenarios(classifier, dataSet, 10, 2, random);
	}

}
