package put.cs.idss.ml.weka;

import java.util.Random;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class DecisionBoundary {
	
	protected static double NORM_CONST = Math.sqrt(2 * Math.PI);

	protected Random fRandom = new Random();

	public DecisionBoundary() {
		// TODO Auto-generated constructor stub
	}
	
	public Instances generateRandomBinaryModel(String name, int numInstances, double[] w, double w0) {
		FastVector attributes = new FastVector();
		for(int i = 0; i < 2; i++) {
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
			double[] x = new double[2];
			for(int j = 0; j < 2; j++) {
				x[j] = 2*fRandom.nextDouble()-1;
			}
			dataset[i] = x;
		}
		
		
		for(int i = 0; i < numInstances; i++) {
			Instance inst = new Instance(3);
			inst.setDataset(instances);
			
			double[] x = dataset[i];
			
			double f = w0;
			for(int j = 0; j < 2; j++) {
				inst.setValue(j, x[j]);
				f +=  w[j]*x[j]; 
			}
			
			String classValue = "0";
			
			 
			
			if(f > 0)
				classValue = "1";
								
			inst.setValue(2, classValue);
			instances.add(inst);
		}
		
		return instances;
	}

	public static void main(String[] args) throws Exception {
		
		final double w0 = 0.0;
		
		final double[] w = {1,1};
		
		
		javax.swing.SwingUtilities.invokeLater(new Runnable() {
			
			public void run() {
				XYSeriesCollection seriesCollection = new XYSeriesCollection();
				
				XYSeries series1 = new XYSeries("0");
				XYSeries series2 = new XYSeries("1");
				
				DecisionBoundary sm = new DecisionBoundary();
				Instances inst = sm.generateRandomBinaryModel("dataset", 10000, w, w0);
				
				inst.setClassIndex(2);
				
				NaiveBayes nb = new NaiveBayes();
				
				try {
					nb.buildClassifier(inst);
				
					for(int i = 0; i < inst.numInstances(); i++) {
						Instance instance = inst.instance(i);
						double[] a = instance.toDoubleArray();
						double y = nb.classifyInstance(instance);
						
						if(y == 0) {
							series1.add(a[0], a[1]);
						} else {
							series2.add(a[0], a[1]);
						}
					}
				} catch(Exception e ){System.out.println(e);};
				
				seriesCollection.addSeries(series1);
				seriesCollection.addSeries(series2);
				
				XYDataset dataset = seriesCollection;
				
				JFreeChart chart = ChartFactory.createScatterPlot(
						"Naive Bayes Decision Boundary ", "x1", "x2", dataset);

				ChartFrame frame = new ChartFrame("Naive Bayes Decision Boundary", chart);
				frame.pack();
				frame.setVisible(true);
			}
		});
	}

}
