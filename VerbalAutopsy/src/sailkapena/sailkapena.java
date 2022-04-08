package sailkapena;

import java.io.File;
import java.io.FileWriter;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.RenameAttribute;
import weka.filters.unsupervised.attribute.Reorder;

/**
 * Klase honek GetModel klasean sortutako .model-a kargatzea, emandako bigarren .csv datu sortari preprocess aplikatzea
 * eta datu-sorta horrekin .model-a ebaluatzea eta iragarpenak gordetzea da.
 * @author aitor
 * @author andoni
 * @author leire
 */
public class sailkapena {

	/**
	 * @param args exekutagarria deitzean terminalean sartutako balioak
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		/*Argumentuak:
		 0- .model
		 1- testBlind.csv
		 2- predictions.txt
		 3- HiztegiaFSS
		 4- testBlind.arff
		 5- testBlindGarbia.csv
		*/
		
		aurreprozesamendua.Aurre.bihurketa(args[1], args[4], args[5]);
		
		DataSource ds = new DataSource(args[4]);
		Instances testBlind = ds.getDataSet();
		testBlind.setClassIndex(testBlind.numAttributes()-1);
		
		NominalToString nts = new NominalToString();
		nts.setAttributeIndexes("6");
		nts.setInputFormat(testBlind);
		testBlind = Filter.useFilter(testBlind, nts);
		
		RenameAttribute ra = new RenameAttribute();
		ra.setAttributeIndices("2");
		ra.setReplace("moduleAttr");
		ra.setInputFormat(testBlind);
		testBlind = Filter.useFilter(testBlind, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("3");
		ra.setReplace("ageAttr");
		ra.setInputFormat(testBlind);
		testBlind = Filter.useFilter(testBlind, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("4");
		ra.setReplace("siteAttr");
		ra.setInputFormat(testBlind);
		testBlind = Filter.useFilter(testBlind, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("5");
		ra.setReplace("sexAttr");
		ra.setInputFormat(testBlind);
		testBlind = Filter.useFilter(testBlind, ra);
		
		
		FixedDictionaryStringToWordVector fds = new FixedDictionaryStringToWordVector();
		File hiztegiBerria = new File(args[3]);
		fds.setDictionaryFile(hiztegiBerria);
		fds.setInputFormat(testBlind);
		

		testBlind = Filter.useFilter(testBlind, fds);
		System.out.println("train_dev-ren atributu kop: " + testBlind.numAttributes());
		
		
		Reorder order = new Reorder();
		order.setAttributeIndices("3,2,7,5,first,8-14,4,15-last,6");
        order.setInputFormat(testBlind);
        testBlind = Filter.useFilter(testBlind, order);		
		
        
		SMO model = (SMO) SerializationHelper.read(args[0]);
		
		Evaluation eval = new Evaluation(testBlind);
		eval.evaluateModel(model, testBlind);
		
		FileWriter fw = new FileWriter(args[2]);
		fw.write("SMO-ren iragarpenak: ");
		int instantzia = 1;
		for (Prediction p:eval.predictions()) {
			fw.write("\n"+instantzia+" Actual value: "+p.actual()+"; predicted value: "+p.predicted()); 
			instantzia++;
		}
		
		fw.close();
		System.out.println("Iragarpenak bukatu dira.");
	}

}
