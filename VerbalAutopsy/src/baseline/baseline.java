package baseline;

import java.io.FileWriter;

import com.googlecode.jfilechooserbookmarks.core.Utils;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
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
		
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(train);
		
		
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(nb, test);
		
		long endTime   = System.nanoTime();
        long totalTime = endTime - startTime;
        int minclassIndex = 0; // adib lehenengoa
		int minClassFreq =  train.numInstances(); // balio altuenaerkin hasieratu
			
		
		for(int i=0; i<train.attribute(train.classIndex()).numValues(); i++){			
			int classFreq = train.attributeStats(train.classIndex()).nominalCounts[i];			
			if (classFreq != 0 && classFreq < minClassFreq) {
				minclassIndex = i;
				minClassFreq = classFreq;
			}			
		}
		System.out.println(minclassIndex + "      " + minClassFreq);
	//	int minoritarioa = weka.core.Utils.minIndex(train.attributeStats(train.classIndex()).nominalCounts);
		
		minclassIndex = 37;
		
		FileWriter fw = new FileWriter(args[2]);
		fw.write("Naive bayes-en exekuzio denbora: " + totalTime/1000000 + " milisegundu.");
		fw.write("\n");
		fw.write("Klase minoritarioaren f-measure = " + eval.fMeasure(minclassIndex));
		fw.write("\n");
		fw.write("Ondo klasifikatutako instantzia ehunekoa: " + eval.pctCorrect());
		fw.write("\n");
		fw.write(eval.toMatrixString());
		fw.close();
		
		FileWriter fw2 = new FileWriter(args[3]);
		fw2.write("Naive Bayes-en iragarpenak:");
		int instantzia = 1;
		for (Prediction p :eval.predictions()) {
            fw2.write("\n" + instantzia + " Actual value: " + test.get(instantzia-1).classValue() + ", predicted: " + p.predicted());
            instantzia++;
        }
		fw2.close();

	}

}
