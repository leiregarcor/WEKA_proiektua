package aurreprozesamendua;

import java.awt.image.ImagingOpException;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceWithMissingValue;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Aurre {
	
	public static void main(String[] args) throws Exception {
		
		/*Argumentuak:
		 1- Datu sorta osoa
		 2- Train_raw
		 3- Dev_raw
		 4- Train_BoW
		 5- dictionary.txt
		 6- Dev_BoW
		*/
		
		//train_raw prozesatu

		
	}
	
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	// -- .csv fitxategi bat .arff-ra bihurtzeko metodoa --
	public void bihurketa(String csv_a ,String j_path, String h_path) throws IOException {
		//Parametroak:
		// --> csv_a: .csv fitxategiaren path-a
		// --> j_path: csv_a-ren path berbera, baina .arff bukaerarekin
		// --> h_path: bihurturiko .arff-a gorden nahi den path-a
		
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(csv_a));
		Instances datuSorta = loader.getDataSet();
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(datuSorta);
		saver.setFile(new File(j_path));
		saver.setDestination(new File(h_path));
		saver.writeBatch();
	}
	
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	// -- Datu sorta osoa, stratified hold-out baten bidez bi multzotan (train eta dev) zatitu --
	public Instances[] zatiketa(DataSource datuak) throws Exception {
		//Datuak kargatu
		Instances data =  datuak.getDataSet();
		//Klasea ezarri? Antes o despues de dividir la data en train y test?
		//data.setClassIndex(data.numAttributes()-1);
				
		//Datuak estratifikatu ???
		StratifiedRemoveFolds srf = new StratifiedRemoveFolds();
		srf.setInputFormat(data);
		srf.setInvertSelection(true);
		srf.setNumFolds(10);
		srf.setFold(7);
				
		Instances train_str = Filter.useFilter(data, srf);
		System.out.println(train_str.size());
				
				
		srf = new StratifiedRemoveFolds();
				
		srf.setInputFormat(data);
		srf.setInvertSelection(false);
		srf.setNumFolds(10);
		srf.setFold(3);
				
		Instances test_str = Filter.useFilter(data, srf);
		System.out.println(test_str.size());
				
		// Klasea ?-rekin ordezkatu
		ReplaceWithMissingValue rpWithMissingValue = new ReplaceWithMissingValue();
		rpWithMissingValue.setProbability(1); //Atributuaren balioa beti aldatzeko?
		//?rpWithMissingValue.setAttributeIndices(Klasea);
		Instances testDataBlind = Filter.useFilter(test_str, rpWithMissingValue);//?
				
		// Datu sortak (train_str eta testDataBlind) idatzi (args[1] eta args[2]-n)
		//SerializationHelper.write(args[1], train_str);
		//SerializationHelper.write(args[2], testDataBlind);
		
		Instances[] instantziak = new Instances[2];
		instantziak[0] = train_str;
		// edo testDataBlind?
		instantziak[1] = test_str;
		return instantziak;
	}
	
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	// -- Emandako bi multzoak BoW formatura bilakatzen ditu --
	public Instances[] bagOfWords(Instances train_raw, Instances dev_raw, String hiztegi_path) throws Exception {
		
		train_raw.setClassIndex(0);
		System.out.println("Train_raw-ren atributu kop: " + train_raw.numAttributes());
		
		StringToWordVector boW = new StringToWordVector();
		File hiztegiaFile = new File(hiztegi_path);
		boW.setInputFormat(train_raw); // Seterrak baino lehen edo ondoren?
		boW.setDictionaryFileToSaveTo(hiztegiaFile);
		
		Instances train_bow = Filter.useFilter(train_raw, boW);
		System.out.println("Train_bow-ren atributu kop: " + train_bow.numAttributes());
		//SerializationHelper.write(args[3], train_bow);
		
		System.out.println(" ");
		System.out.println(" -- -- -- -- ");
		System.out.println(" ");
		//Hiztegia lortu da eta train_bow, orain hiztegiarekin dev_bow lortuko dugu
		
		dev_raw.setClassIndex(0);
		System.out.println("Dev_raw-ren atributu kop: " + dev_raw.numAttributes());

		FixedDictionaryStringToWordVector fixedBoW = new FixedDictionaryStringToWordVector();
		File hiztegia = new File(hiztegi_path);
		fixedBoW.setDictionaryFile(hiztegia);
		fixedBoW.setInputFormat(dev_raw);

		Instances dev_bow = Filter.useFilter(dev_raw, fixedBoW);
		System.out.println("Dev_bow-ren atributu kop: " + dev_bow.numAttributes());
		//SerializationHelper.write(args[5], dev_bow);
		
		Instances[] instantziak = new Instances[2];
		instantziak[0] = train_bow;
		instantziak[1] = dev_bow;
		return instantziak;
	}
	
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	// -- 
	public Instances[] featureSS(Instances train_bow, Instances dev_bow) throws Exception {
		
		train_bow.setClassIndex(train_bow.numAttributes()-1);
		int preFSS = train_bow.numAttributes();
		
		//FSS aplikatu: (Attribute selection edo Ranker)?
		AttributeSelection aSelection = new AttributeSelection();
		aSelection.setInputFormat(train_bow);
		Instances train_bow_fss = Filter.useFilter(train_bow, aSelection);
		int postFSS = train_bow_fss.numAttributes();
		
		//Atributu kopurua FSS baino lehen eta ondoren:
		System.out.println(" ");
		System.out.println(" FSS aplikatu baino lehen: " + preFSS + " FSS aplikatu ondoren: " + postFSS);

		
		//Train BoW FSS erabiliz bere header-aren bidez dev_BoW_FSS lortu
		Remove remove = new Remove();
		remove.setInputFormat(train_bow_fss);
		Instances dev_bow_fss = Filter.useFilter(train_bow, remove); //Suposatzen da setInputFormat trainarentzat eginda
		//FSS jaso duen train multzoaren atributuak begiratu eta filtroa test-ean erabiltzea bietako atributuak konparatuko
		//dituela biak berdin utziz ....?¿
		
		Instances[] instantziak = new Instances[2];
		instantziak[0] = train_bow_fss;
		instantziak[1] = dev_bow_fss;
		return instantziak;
	}
}
