package inferentzia;

import weka.core.Instances;
import weka.core.SerializationHelper;
import java.io.FileWriter;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;

public class GetModel {
	public void inferentzia(Instances data_BOW_FSS, Instances dev_BOW_FSS, String[] args) throws Exception {
		// sailkatzaile optimoa lortu
		System.out.println("\nSAILKATZAILE OPTIMOAREN BILA"); // Hold - out 
		double fmeasure = 0.0;
		double aux = 0.0;
		double exp = 1;
		for (double i = 1; i < 6; i++) { // exponentea iteratuko dugu 1-etik 5-era

			SMO model = new SMO();

			// kernel konfiguratu
			PolyKernel pk = new PolyKernel();
			// exponentea aldaty
			pk.setExponent(i);

			// kernel SMO-ri pasatu
			model.setKernel(pk);

			// eredua entrenatu
			model.buildClassifier(data_BOW_FSS); 

			// ebaluatu 10fcv erabiliko da
			Evaluation ev = new Evaluation(data_BOW_FSS);
			ev.evaluateModel(model, data_BOW_FSS);

			aux = ev.weightedFMeasure();
			System.out.println("Berretzailea: " + i + " weighted f-measure: " + aux);

			if (fmeasure < aux) {
				fmeasure = aux;
				exp = i;
			}						
		}
		System.out.println("\nOPTIMOA\nBerretzailea: " + exp + " weighted f-measure: " + fmeasure);



		// Eredu optimoa gorde		
		// 1- modelo optimoa 
		SMO model = new SMO();
		PolyKernel pKernel = new PolyKernel();
		pKernel.setExponent(exp);
		model.setKernel(pKernel);
		model.buildClassifier(data_BOW_FSS);

		System.out.println("Eredu optimoa gordetzen**********************");
		SerializationHelper.write(args[2], model);

		// sailkatzailearen kalitatearen estimazioa:
		FileWriter writer = new FileWriter(args[3], true);
		writer.write("\n KALITATEAREN ESTIMAZIOA, ebaluazio ez zintzoa");
		Evaluation eva = new Evaluation(data_BOW_FSS);
		eva.evaluateModel(model, data_BOW_FSS);

		// klase minoritarioaren precision recall eta fmeasure :
		int classIndex = data_BOW_FSS.classIndex(); //duplicado
		int minclassIndex = weka.core.Utils.minIndex(data_BOW_FSS.attributeStats(classIndex).nominalCounts); // clasearen frekuentzia minimo duen balioaren indizea

		writer.write("\nKlase minoritarioren precision: " + eva.precision(minclassIndex) + " recall: " + eva.recall(minclassIndex) + " f-measure: " + eva.fMeasure(minclassIndex));

		// ondo dagoela frogatzeko
		writer.write("\n frogatzeko ondo dagoela:");
		writer.write(eva.toClassDetailsString());

		// nahasmen matrizea
		writer.write(eva.toMatrixString());	

		writer.close();



	}

}
