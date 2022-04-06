package aurreprozesamendua;

import java.awt.image.ImagingOpException;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RenameAttribute;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.ReplaceWithMissingValue;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Aurre {
	
	public static void main(String[] args) throws Exception {
		
		/*Argumentuak:
		 0- .csv
		 1- .arff
		 2- .csv garbia
		 3- Hiztegia
		 4- trainBOWFSS
		 5- DevBOWFSS
		*/
		//train_raw prozesatu
		bihurketa(args[0], args[1], args[2]);
		ArffLoader aLoader = new ArffLoader();
		File f = new File(args[1]);
		aLoader.setFile(f);
		Instances data = aLoader.getDataSet();
		data.setClassIndex(data.numAttributes()-1);
		Instances[] instantziak = zatiketa(data);
		Instances[] bowInstantziak = bagOfWords(instantziak[0], instantziak[1], args[3]);
		Instances[] bowFssInstantziak = featureSS(bowInstantziak[0], bowInstantziak[1]);
		ArffSaver aSaver = new ArffSaver();
		aSaver.setInstances(bowFssInstantziak[0]);
		File trainBF = new File(args[4]);
		aSaver.setFile(trainBF);
		aSaver.writeBatch();
		aSaver = new ArffSaver();
		aSaver.setInstances(bowFssInstantziak[1]);
		File devBF = new File(args[5]);
		aSaver.setFile(devBF);
		aSaver.writeBatch();
		System.out.println("Train_BOW_FSS eta Dev_BOW_FSS gorde egin dira!");
	}
	
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	// -- .csv fitxategi bat .arff-ra bihurtzeko metodoa --
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
	
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	// -- Datu sorta osoa, stratified hold-out baten bidez bi multzotan (train eta dev) zatitu --
	public static Instances[] zatiketa(Instances datuak) throws Exception {
		
		//Klasea ezarri? Antes o despues de dividir la data en train y test?
		//data.setClassIndex(data.numAttributes()-1);
				
		//Datuak estratifikatu ???
		StratifiedRemoveFolds srf = new StratifiedRemoveFolds();
		srf.setInputFormat(datuak);
		srf.setInvertSelection(true);
		srf.setNumFolds(10);
		srf.setFold(7);
				
		Instances train_str = Filter.useFilter(datuak, srf);
		System.out.println(train_str.size());
				
				
		srf = new StratifiedRemoveFolds();
				
		srf.setInputFormat(datuak);
		srf.setInvertSelection(false);
		srf.setNumFolds(10);
		srf.setFold(3);
				
		Instances dev_str = Filter.useFilter(datuak, srf);
		System.out.println(dev_str.size());
				
		// Datu sortak (train_str eta testDataBlind) idatzi (args[1] eta args[2]-n)
		//SerializationHelper.write(args[1], train_str);
		//SerializationHelper.write(args[2], testDataBlind);
		
		Instances[] instantziak = new Instances[2];
		instantziak[0] = train_str;
		// edo testDataBlind?
		instantziak[1] = dev_str;
		System.out.println("Train eta dev sortu egin dira");
		return instantziak;
	}
	
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	// -- Emandako bi multzoak BoW formatura bilakatzen ditu --
	public static Instances[] bagOfWords(Instances train_raw, Instances dev_raw, String hiztegi_path) throws Exception {
		
		System.out.println("Train_raw-ren atributu kop: " + train_raw.numAttributes());
		
		NominalToString nts = new NominalToString();
		nts.setAttributeIndexes("6");
		nts.setInputFormat(train_raw);
		Instances dataTrainRawStr = Filter.useFilter(train_raw, nts);
		
		RenameAttribute ra = new RenameAttribute();
		ra.setAttributeIndices("2");
		ra.setReplace("moduleAttr");
		ra.setInputFormat(dataTrainRawStr);
		dataTrainRawStr = Filter.useFilter(dataTrainRawStr, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("3");
		ra.setReplace("ageAttr");
		ra.setInputFormat(dataTrainRawStr);
		dataTrainRawStr = Filter.useFilter(dataTrainRawStr, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("4");
		ra.setReplace("siteAttr");
		ra.setInputFormat(dataTrainRawStr);
		dataTrainRawStr = Filter.useFilter(dataTrainRawStr, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("5");
		ra.setReplace("sexAttr");
		ra.setInputFormat(dataTrainRawStr);
		dataTrainRawStr = Filter.useFilter(dataTrainRawStr, ra);
		
		StringToWordVector boW = new StringToWordVector();
		File hiztegiaFile = new File(hiztegi_path);
		boW.setDictionaryFileToSaveTo(hiztegiaFile);
		boW.setInputFormat(dataTrainRawStr);
		
		Instances train_bow = Filter.useFilter(dataTrainRawStr, boW);
		System.out.println("Train_bow-ren atributu kop: " + train_bow.numAttributes());
		//SerializationHelper.write(args[3], train_bow);
		
		System.out.println(" ");
		System.out.println(" -- -- -- -- ");
		System.out.println(" ");
		//Hiztegia lortu da eta train_bow, orain hiztegiarekin dev_bow lortuko dugu
		
		nts = new NominalToString();
		nts.setAttributeIndexes("6");
		nts.setInputFormat(dev_raw);
		Instances dataDevRawStr = Filter.useFilter(dev_raw, nts);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("2");
		ra.setReplace("moduleAttr");
		ra.setInputFormat(dataDevRawStr);
		dataDevRawStr = Filter.useFilter(dataDevRawStr, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("3");
		ra.setReplace("ageAttr");
		ra.setInputFormat(dataDevRawStr);
		dataDevRawStr = Filter.useFilter(dataDevRawStr, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("4");
		ra.setReplace("siteAttr");
		ra.setInputFormat(dataDevRawStr);
		dataDevRawStr = Filter.useFilter(dataDevRawStr, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("5");
		ra.setReplace("sexAttr");
		ra.setInputFormat(dataDevRawStr);
		dataDevRawStr = Filter.useFilter(dataDevRawStr, ra);
		
		System.out.println("Dev_raw-ren atributu kop: " + dataDevRawStr.numAttributes());

		FixedDictionaryStringToWordVector fixedBoW = new FixedDictionaryStringToWordVector();
		fixedBoW.setDictionaryFile(hiztegiaFile);
		fixedBoW.setInputFormat(dataDevRawStr);
		

		Instances dev_bow = Filter.useFilter(dataDevRawStr, fixedBoW);
		System.out.println("Dev_bow-ren atributu kop: " + dev_bow.numAttributes());
		//SerializationHelper.write(args[5], dev_bow);
		
		Instances[] instantziak = new Instances[2];
		instantziak[0] = train_bow;
		instantziak[1] = dev_bow;
		return instantziak;
	}
	
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	// -- 
	public static Instances[] featureSS(Instances train_bow, Instances dev_bow) throws Exception {
		
		System.out.println(train_bow.classIndex());
		int preFSS = train_bow.numAttributes();
		
		//FSS aplikatu: (Attribute selection edo Ranker)?
		AttributeSelection aSelection = new AttributeSelection();
		aSelection.setInputFormat(train_bow);
		Instances train_bow_fss = Filter.useFilter(train_bow, aSelection);
		int postFSS = train_bow_fss.numAttributes();
		
		//Atributu kopurua FSS baino lehen eta ondoren:
		System.out.println(" ");
		System.out.println(" FSS aplikatu baino lehen: " + preFSS + " FSS aplikatu ondoren: " + postFSS);

		/*
		//Train BoW FSS erabiliz bere header-aren bidez dev_BoW_FSS lortu
		Remove remove = new Remove();
		remove.setInputFormat(train_bow_fss);
		Instances dev_bow_fss = Filter.useFilter(dev_bow, remove); //Suposatzen da setInputFormat trainarentzat eginda
		//FSS jaso duen train multzoaren atributuak begiratu eta filtroa test-ean erabiltzea bietako atributuak konparatuko
		//dituela biak berdin utziz ....?¿ */
		dev_bow.setClassIndex(5);
		int[] borratzeko =  new int[dev_bow.numAttributes()-train_bow_fss.numAttributes()];
		int aux = 0;
		for(int i=0; i< dev_bow.numAttributes()-1; i++) {
			Attribute a = Collections.list(dev_bow.enumerateAttributes()).get(i);
			if(!Collections.list(train_bow_fss.enumerateAttributes()).contains(a)) {
				borratzeko[aux] = a.index();
				aux ++;
			}
		}
		
		Remove remove = new Remove(); 
		remove.setAttributeIndicesArray(borratzeko);
		remove.setInputFormat(dev_bow);
		Instances dev_bow_fss = Filter.useFilter(dev_bow, remove);
		
		System.out.println("FSS eginda!");
		
		Reorder order = new Reorder();
		order.setAttributeIndices("first-4,6-last,5");
        order.setInputFormat(dev_bow_fss);
        dev_bow_fss = Filter.useFilter(dev_bow_fss, order);
        
        Instances[] instantziak = new Instances[2];
		instantziak[0] = train_bow_fss;
		instantziak[1] = dev_bow_fss;
		return instantziak;
		
	}
}
