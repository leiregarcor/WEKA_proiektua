package inferentzia;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.RenameAttribute;
import weka.filters.unsupervised.attribute.Reorder;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.Puk;
import weka.classifiers.functions.supportVector.RBFKernel;

/**
 * Klase honetan Aurre klasean sortutako train eta dev datu-sorten bitartez SMO algoritmoaren parametroak ekortu
 * egingo dira optimoak (f-measure altuena dutenak) lortzeko eta lortu ostean train+dev-ekin SMO.model entrenatu, ebaluatu
 * eta gorde egingo da
 * @author aitor
 * @author andoni
 * @author leire
 */
public class GetModel {
	/**
	 * @param args exekutagarria deitzean terminalean sartutako balioak
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		
		/*Argumentuak:
		 0- trainBOWFSS
		 1- DevBOWFSS
		 2- .model
		 3- evaluation.txt
		 4- data.arff
		 5- Hiztegi FSS
		*/
		
		// fitx kargatu
		DataSource trainSource = new DataSource(args[0]);
		Instances train = trainSource.getDataSet();

		// klasea azken atributua da
		train.setClassIndex(train.numAttributes()-1);

		// fitx kargatu
		DataSource devSource = new DataSource(args[1]);
		Instances dev = devSource.getDataSet();

		// klasea azken atributua da
		dev.setClassIndex(dev.numAttributes()-1);
				
		inferentzia(train, dev, args[2],args[3], args[4], args[5]);


	}
	
	/**
	 * Metodo honen bidez, datu-sortako klase minoritarioa topatuko da lehenik klase ezberdinen maiztasunak beatuz. Hau, ekorketa klase
	 * minoritarioaren f-measurearekiko egingo delako da. Irizpide honen bidez, kernel ezberdinen eta hauek erabiltzen dituzten parametroen artean
	 * ekorketa prozesu bat emango da balio optimoak topatzeko. Optimo hauek lortutakoan modeloa, kalitatearen estimazioa eta datu ezberdinak gordeko dira.
	 * @param data_BOW_FSS Train datu-sorta instantzia 
	 * @param dev_BOW_FSS Dev datu_sorta instanzia
	 * @param pathModel Modeloa non gorde nahi den adierazten duen path-a
	 * @param path_kalitate Kalitatearen estimazio non gorde nahi den adierazten duen path-a
	 * @param pathData Zatiketa egin baino lehen datu sortaren path-a
	 * @param pathHiztegia Hiztegi berria gordeta dagoen path-a
	 * @throws Exception
	 */
	public static void inferentzia(Instances data_BOW_FSS, Instances dev_BOW_FSS, String pathModel, String path_kalitate, String pathData, String pathHiztegia) throws Exception {
		
		// Ekorketa burutzeko lehenengo klase minoriatarioa aurkitu behar dugu, hau da, ez 0 direnen artean frekuentzia minimoa duen balioa.
		// ekorketa klase minoritarioaren f-measure hobetzea du helburu, horretarako kernelak eta hauek behar dituzten parametroak aldatuko dira iterazioetan
		
		int minclassIndex = 0; // adib lehenengoa
		int minClassFreq =  data_BOW_FSS.numInstances(); // balio altuenaerkin hasieratu			
		
		for(int i=0; i<data_BOW_FSS.attribute(data_BOW_FSS.classIndex()).numValues(); i++){			
			int classFreq = data_BOW_FSS.attributeStats(data_BOW_FSS.classIndex()).nominalCounts[i];			
			if (classFreq != 0 && classFreq < minClassFreq) {
				minclassIndex = i;
				minClassFreq = classFreq;
			}			
		}
		System.out.println(minclassIndex + "    " + minClassFreq);
		// sailkatzaile optimoa lortu
		System.out.println("\n Tunning parameters for SMO"); 
		minclassIndex = 37;

		double fmeasure = 0.0;
		double aux = 0.0;
		double exp = 1;
		double gamma = 1;
		double omega = 1;
		Kernel kernelOpt = null;

		

		Kernel[] kernelak = new Kernel[] {new PolyKernel(),new RBFKernel(),new Puk() };

		for (Kernel k : kernelak) {			
			if (k instanceof PolyKernel) {

				for (double i = 1; i < 6; i++) { // exponentea iteratuko dugu 1-etik 5-era

					SMO model = new SMO();

					// kernel konfiguratu
					k = new PolyKernel();

					// exponentea aldatu
					((PolyKernel) k).setExponent(i);

					// kernel SMO-ri pasatu
					model.setKernel(k);

					// eredua entrenatu
					model.buildClassifier(data_BOW_FSS); 

					// ebaluatu 
					Evaluation ev = new Evaluation(data_BOW_FSS);
					ev.evaluateModel(model, dev_BOW_FSS);

					aux = ev.fMeasure(minclassIndex);
					System.out.println("exponentea: " + i + "; fmeasure: " + aux);
					if (fmeasure < aux) {
						fmeasure = aux;
						exp = i;
						kernelOpt = k;
					}
				}

			}
			if (k instanceof RBFKernel) {
				for (double i = 1; i < 6; i++) {
					SMO model = new SMO();

					// kernel konfiguratu
					k = new RBFKernel();

					//gamma aldatu
					((RBFKernel) k).setGamma(i);					

					// kernel SMO-ri pasatu
					model.setKernel(k);
					// eredua entrenatu
					model.buildClassifier(data_BOW_FSS); 

					// ebaluatu 
					Evaluation ev = new Evaluation(data_BOW_FSS);
					ev.evaluateModel(model, dev_BOW_FSS);

					aux = ev.fMeasure(minclassIndex);
					System.out.println("gamma: " + i + "; fmeasure: " + aux);
					if (fmeasure < aux) {
						fmeasure = aux;
						gamma = i;
						kernelOpt = k;
					}
				}

			}
			if (k instanceof Puk) {
				for (double i = 1; i < 6; i++) {

					SMO model = new SMO();

					// kernel konfiguratu
					k = new Puk();

					//omega aldatu
					((Puk) k).setOmega(i);					

					// kernel SMO-ri pasatu
					model.setKernel(k);

					// eredua entrenatu
					model.buildClassifier(data_BOW_FSS); 

					// ebaluatu 
					Evaluation ev = new Evaluation(data_BOW_FSS);
					ev.evaluateModel(model, dev_BOW_FSS);

					aux = ev.fMeasure(minclassIndex);
					System.out.println("omega: " + i + "; fmeasure: " + aux);

					if (fmeasure < aux) {
						fmeasure = aux;
						omega = i;
						kernelOpt = k;
					}
				}

			}			

		}

		System.out.println("* PARAMETRO OPTIMOAK: ");
		// behin parameto optimoak lotu direla eredu optimoa gordeko da

		// modelo optimoa eraiki
		SMO model = new SMO();
		Kernel ker = null;

		if (kernelOpt instanceof PolyKernel) { 
			System.out.println("Exponent: " + exp + " klase minoritarioaren f-measure: " + aux);
			ker = new PolyKernel();
			((PolyKernel) ker).setExponent(exp);

		}else if (kernelOpt instanceof RBFKernel) {
			System.out.println("Gamma: " + gamma + " klase minoritarioaren f-measure: " + aux);	
			ker = new RBFKernel();
			((RBFKernel) ker).setGamma(gamma);
		}else {
			System.out.println("Omega: " + omega + " klase minoritarioaren f-measure: " + aux);
			ker = new Puk();
			((Puk) ker).setOmega(omega);	
		}

		// sailkatzailearen kalitatearen estimazioa egiteko train eta dev batera dituen dataset-a erabiliko da
		// kalitatearen estimazioa egiteko ebaluaketa ez zintzoa eginez goi-bornea lortuko da, gero 10-fold cv ere egingo da.
		
		DataSource dSource = new DataSource(pathData);
		Instances train_dev = dSource.getDataSet();
		train_dev.setClassIndex(train_dev.numAttributes()-1);
		
		NominalToString nts = new NominalToString();
		nts.setAttributeIndexes("6");
		nts.setInputFormat(train_dev);
		train_dev = Filter.useFilter(train_dev, nts);
		
		RenameAttribute ra = new RenameAttribute();
		ra.setAttributeIndices("2");
		ra.setReplace("moduleAttr");
		ra.setInputFormat(train_dev);
		train_dev = Filter.useFilter(train_dev, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("3");
		ra.setReplace("ageAttr");
		ra.setInputFormat(train_dev);
		train_dev = Filter.useFilter(train_dev, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("4");
		ra.setReplace("siteAttr");
		ra.setInputFormat(train_dev);
		train_dev = Filter.useFilter(train_dev, ra);
		
		ra = new RenameAttribute();
		ra.setAttributeIndices("5");
		ra.setReplace("sexAttr");
		ra.setInputFormat(train_dev);
		train_dev = Filter.useFilter(train_dev, ra);		
		
		FixedDictionaryStringToWordVector fds = new FixedDictionaryStringToWordVector();
		File hiztegiBerria = new File(pathHiztegia);
		fds.setDictionaryFile(hiztegiBerria);
		fds.setInputFormat(train_dev);
		

		train_dev = Filter.useFilter(train_dev, fds);
		System.out.println("train_dev-ren atributu kop: " + train_dev.numAttributes());
		
		
		Reorder order = new Reorder();
		order.setAttributeIndices("3,2,7,5,first,8-14,4,15-last,6");
        order.setInputFormat(train_dev);
        train_dev = Filter.useFilter(train_dev, order);
		
		model = new SMO();
		model.setKernel(ker);
		model.buildClassifier(train_dev);

		// ebaluazio ez-zintzoa
		System.out.println("Ebaluazio ez-zintzoa");
		FileWriter writer = new FileWriter(path_kalitate);
		writer.write("\n KALITATEAREN ESTIMAZIOA, ebaluazio ez zintzoa");
		Evaluation eva = new Evaluation(train_dev);
		eva.evaluateModel(model, train_dev);
		writer.write("\nKlase minoritarioren f-measure: " + eva.fMeasure(minclassIndex));
		writer.write("\nOndo klasifikatutako instantzia ehunekoa: " + eva.pctCorrect());
		System.out.println("Ebaluazio ez-zintzoa bukatuta");


		// 10-fold cross validation
		System.out.println("10 fold cross-validation");
		writer.write("\n KALITATEAREN ESTIMAZIOA, 10-fold cross validation ebaluazio eskema");
		Evaluation eva2 = new Evaluation(train_dev);
		eva2.crossValidateModel(model, train_dev, 10, new Random(1));
		writer.write("\nKlase minoritarioren f-measure: " + eva2.fMeasure(minclassIndex));
		writer.write("\nOndo klasifikatutako instantzia ehunekoa: " + eva2.pctCorrect());
		System.out.println("10 fold cross-validation bukatuta");


		writer.close();
		System.out.println("Ebaluazioa eginda eta idatzita.");
		
		System.out.println("Eredu optimoa gordetzen**********************");
		SerializationHelper.write(pathModel, model);
		System.exit(0);
	}
}
