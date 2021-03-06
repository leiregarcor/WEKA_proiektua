package baseline;

import java.io.FileWriter;
import java.util.Random;

import com.googlecode.jfilechooserbookmarks.core.Utils;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Klase honetan Naive Bayes entrenatuko da Aurre klasean sortutako train datu-sortaren bitartez
 * eta Aurre klasean sortutako dev datu-sortarekin testeatuko da parametro ekorketarik
 * egin gabe lor daitekeen f-measure, pctcorrect eta predikzioak lortzearren, kalitatearen behe borne bat izateko
 * @author aitor
 * @author andoni
 * @author leire
 */
public class baseline {
	
	static long startTime = System.nanoTime();

	/**
	 * @param args exekutagarria deitzean terminalean sartutako balioak 
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		
		/*Argumentuak:
		 0- trainBOWFSS
		 1- DevBOWFSS
		 2- evaluation.txt
		 3- predictions.txt
		 3- trainDev
		*/		
		
		DataSource dsTrain = new DataSource(args[0]);
		Instances train = dsTrain.getDataSet();
		train.setClassIndex(train.numAttributes()-1);
		DataSource dsTest = new DataSource(args[1]);
		Instances test = dsTest.getDataSet();
		test.setClassIndex(test.numAttributes()-1);
		DataSource dsTrainDev = new DataSource(args[4]);
		Instances trainDev = dsTrainDev.getDataSet();
		trainDev.setClassIndex(trainDev.numAttributes()-1);
		
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(train);
		
		System.out.println("Ebaluatzaileak eraikitzen");
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(nb, test);
		
		NaiveBayes nb2 = new NaiveBayes();
		nb2.buildClassifier(trainDev);
		
		long endTime   = System.nanoTime();
        long totalTime = endTime - startTime;
        
		Evaluation eval2 = new Evaluation(trainDev);
		eval2.crossValidateModel(nb2, trainDev, 10, new Random(1));
		
		Evaluation eval3 = new Evaluation(trainDev);
		eval3.evaluateModel(nb2, trainDev);
		
		
        int minclassIndex = 0; // adib lehenengoa
		int minClassFreq =  train.numInstances(); // balio altuenaerkin hasieratu
			
		
		for(int i=0; i<train.attribute(train.classIndex()).numValues(); i++){			
			int classFreq = train.attributeStats(train.classIndex()).nominalCounts[i];			
			if (classFreq != 0 && classFreq < minClassFreq) {
				minclassIndex = i;
				minClassFreq = classFreq;
			}			
		}
		//	int minoritarioa = weka.core.Utils.minIndex(train.attributeStats(train.classIndex()).nominalCounts);
		
		minclassIndex = 37;
		
		System.out.println("Ebaluazioak egiten");
		
		FileWriter fw = new FileWriter(args[2]);
		fw.write("Naive bayes-en exekuzio denbora: " + totalTime/1000000000 + " segundu.");
		fw.write("\n");
		System.out.println("Hold-out");
		fw.write("Klase minoritarioaren f-measure hold-out= " + eval.fMeasure(minclassIndex));
		fw.write("\n");
		fw.write("Ondo klasifikatutako instantzia ehunekoa hold-out: " + eval.pctCorrect());
		fw.write("\n");
		System.out.println("10 fold cross-validation");
		
		
		fw.write("Klase minoritarioaren f-measure 10 fold cv= " + eval2.fMeasure(minclassIndex));
		fw.write("\n");
		fw.write("Ondo klasifikatutako instantzia ehunekoa 10 fold cv: " + eval2.pctCorrect());
		fw.write("\n");
		fw.write("Klase minoritarioaren f-measure ez-zintzoa= " + eval3.fMeasure(minclassIndex));
		fw.write("\n");
		fw.write("Ondo klasifikatutako instantzia ehunekoa ez-zintzoa: " + eval3.pctCorrect());
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
		System.exit(0);
	}

}
