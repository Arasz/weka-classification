package put.cs.idss.ml.weka;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class Experiment {


	public static Instances LoadData(String path) throws IOException {
		BufferedReader bufferedReader = new BufferedReader(new FileReader(path));
		Instances dataset = new Instances(bufferedReader);
		bufferedReader.close();

		return dataset;
	}

	public static void ConfigureClassIndex(Instances data){
		if(data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
	}

	public static void runExperiment(Classifier classifier, String trainSetPath,
			String testSetPath) throws Exception {

		Instances trainSet = LoadData(trainSetPath);

		Instances testSet = LoadData(testSetPath);
		
		ConfigureClassIndex(trainSet);
		ConfigureClassIndex(testSet);
		
		System.out.println("Data loaded.");
		
		classifier.buildClassifier(trainSet);
		System.out.println("Classifier has been learned.");
		
		System.out.println("                        \th\ty\tdist");
		
		double sum = 0;
		double correct = 0;
		for(int i = 0; i < testSet.numInstances(); i++) {
			Instance instance = testSet.instance(i);
			int truth = (int) instance.classValue();
			int prediction = (int) classifier.classifyInstance(instance);
			System.out.println("Prediction for instance " + i + "\t" + prediction + "\t" + truth + "\t" + Arrays.toString(classifier.distributionForInstance(instance)));
			sum++;
			if(truth == prediction) correct++;
		}
		System.out.println("\nAccuracy: " + (correct/sum));
	}

	public static void main(String[] args) throws Exception {
		NaiveBayesClassifier hc = new NaiveBayesClassifier();
		runExperiment(hc, "data/spambase-train.arff", "data/spambase-test.arff");
	}

}
