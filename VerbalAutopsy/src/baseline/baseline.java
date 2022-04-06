package baseline;

import java.io.FileWriter;

import com.googlecode.jfilechooserbookmarks.core.Utils;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class baseline {
	
	static long startTime = System.nanoTime();

	public static void main(String[] args) throws Exception {
		DataSource dsTrain = new DataSource(args[0]);
		Instances train = dsTrain.getDataSet();
		train.setClassIndex(train.numAttributes()-1);
		DataSource dsTest = new DataSource(args[1]);
		Instances test = dsTest.getDataSet();
		test.setClassIndex(test.numAttributes()-1);
		
		System.out.println(train.get(5).classAttribute().toString());
		
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(train);
		
		
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(nb, test);
		
		long endTime   = System.nanoTime();
        long totalTime = endTime - startTime;
		int minoritarioa = weka.core.Utils.minIndex(train.attributeStats(train.classIndex()).nominalCounts);
		
		FileWriter fw = new FileWriter(args[2]);
		fw.write("Naive bayes-en exekuzio denbora: " + totalTime);
		fw.write("\n");
		fw.write("Klase minoritarioaren f-measure = " + eval.fMeasure(minoritarioa));
		fw.close();

	}

}
