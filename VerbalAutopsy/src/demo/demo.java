package demo;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.RenameAttribute;
import weka.filters.unsupervised.attribute.Reorder;

public class demo {

	public static void main(String[] args) throws Exception {
		/*Argumentuak:
		 0- .model
		 1- demoBlind.csv
		 2- predictions.txt
		 3- HiztegiaFSS
		*/
		bihurketa(args[1], "/home/aitor/Escritorio/Demo/Laguntzaileak/demoBlind.arff", "/home/aitor/Escritorio/Demo/Laguntzaileak/demoBlindGarbia.csv");
		
		DataSource ds = new DataSource("/home/aitor/Escritorio/Demo/Laguntzaileak/demoBlind.arff");
		Instances demoBlind = ds.getDataSet();
		demoBlind.setClassIndex(demoBlind.numAttributes()-1);
		
		DataSource ds1 = new DataSource("/home/aitor/Escritorio/Demo/Laguntzaileak/trainBOWFSS.arff");
		Instances trainDev = ds1.getDataSet();
		trainDev.setClassIndex(trainDev.numAttributes()-1);
		
		NominalToString nts = new NominalToString();
		nts.setAttributeIndexes("6");
		nts.setInputFormat(demoBlind);
		demoBlind = Filter.useFilter(demoBlind, nts);
		
		RenameAttribute ra = new RenameAttribute();
		ra.setAttributeIndices("2");
		ra.setReplace("moduleAttr");
		ra.setInputFormat(demoBlind);
		demoBlind = Filter.useFilter(demoBlind, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("3");
		ra.setReplace("ageAttr");
		ra.setInputFormat(demoBlind);
		demoBlind = Filter.useFilter(demoBlind, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("5");
		ra.setReplace("siteAttr");
		ra.setInputFormat(demoBlind);
		demoBlind = Filter.useFilter(demoBlind, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("4");
		ra.setReplace("sexAttr");
		ra.setInputFormat(demoBlind);
		demoBlind = Filter.useFilter(demoBlind, ra);
		
		System.out.println("demo_dev-ren atributu kop: " + demoBlind.numAttributes());
		
		
		FixedDictionaryStringToWordVector fds = new FixedDictionaryStringToWordVector();
		File hiztegiBerria = new File(args[3]);
		fds.setDictionaryFile(hiztegiBerria);
		fds.setInputFormat(demoBlind);
		

		demoBlind = Filter.useFilter(demoBlind, fds);
		System.out.println("demo_dev-ren atributu kop BOW ostean: " + demoBlind.numAttributes());
		
		
		Reorder order = new Reorder();
		order.setAttributeIndices("3,2,7,5,first,8-14,4,15-last,6");
        order.setInputFormat(demoBlind);
        demoBlind = Filter.useFilter(demoBlind, order);
        
        SMO model = (SMO) SerializationHelper.read(args[0]);
		
		Evaluation eval = new Evaluation(demoBlind);
		eval.evaluateModel(model, demoBlind);
		
		FileWriter fw = new FileWriter(args[2]);
		fw.write("SMO-ren iragarpenak: ");
		int instantzia = 1;
		for (Prediction p:eval.predictions()) {
			System.out.println("##########################################");
			System.out.println(instantzia+" Predicted value: "+p.predicted()+ ", hau da, " + trainDev.attribute(trainDev.numAttributes()-1).value((int)p.predicted()));
			System.out.println("##########################################");
			fw.write("\n"+instantzia+" Predicted value: "+p.predicted()+ ", hau da, " + trainDev.attribute(trainDev.numAttributes()-1).value((int)p.predicted())); 
			instantzia++;
		}
		
		fw.close();
		System.out.println("Iragarpenak bukatu dira.");
		
	}
	
	
	
	public static void bihurketa(String csv_a ,String j_path, String h_path) throws Exception {
		//Parametroak:
		// --> csv_a: .csv fitxategiaren path-a
		// --> j_path: csv_a-ren path berbera, baina .arff bukaerarekin
		// --> h_path: bihurturiko .arff-a gorden nahi den path-a
		Path csvPath = Path.of(csv_a);
		String csvEdukia = Files.readString(csvPath);
		String csvEdukiaOna = csvEdukia.replace("'", " ");
		
		File csvFile = new File(h_path);		
		FileWriter fw = new FileWriter(h_path);
		fw.write(csvEdukiaOna);
		fw.close();
		
		File csvOna = new File(h_path);
		
		
		CSVLoader loader = new CSVLoader();
        loader.setSource(csvOna);
        
        Instances data = loader.getDataSet();

        // save as an  ARFF (output file)
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(j_path));
       // saver.setDestination(new File(j_path));
        saver.writeBatch();
        System.out.println("Arff fitxategia sortu da");

	}

}
