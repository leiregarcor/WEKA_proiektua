package inferentzia;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileWriter;
import java.util.Random;


import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.Puk;
import weka.classifiers.functions.supportVector.RBFKernel;

public class GetModel {
	public static void inferentzia(Instances data_BOW_FSS, Instances dev_BOW_FSS, String pathModel, String path_kalitate) throws Exception {

		// sailkatzaile optimoa lortu
		System.out.println("\n Tunning parameters for SMO"); 

		double fmeasure = 0.0;
		double aux = 0.0;
		double exp = 1;
		double gamma = 1;
		double omega = 1;
		Kernel kernelOpt = null;

		int minclassIndex = weka.core.Utils.minIndex(data_BOW_FSS.attributeStats(data_BOW_FSS.classIndex()).nominalCounts); 

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
					//System.out.println("Exponent: " + i + " klase minoritarioaren f-measure: " + aux);

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
					//System.out.println("Gamma: " + i + " klase minoritarioaren f-measure: " + aux);

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
					//System.out.println("Omega: " + i + " klase minoritarioaren f-measure: " + aux);

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

		model.setKernel(ker);
		model.buildClassifier(data_BOW_FSS);

		System.out.println("Eredu optimoa gordetzen**********************");
		SerializationHelper.write(pathModel, model);



		// sailkatzailearen kalitatearen estimazioa:

		Instances train_dev = merge(data_BOW_FSS, dev_BOW_FSS);

		// ebaluazio ez-zintzoa
		FileWriter writer = new FileWriter(path_kalitate, true);
		writer.write("\n KALITATEAREN ESTIMAZIOA, ebaluazio ez zintzoa");
		Evaluation eva = new Evaluation(train_dev);
		eva.evaluateModel(model, train_dev);
		writer.write("\nKlase minoritarioren f-measure: " + eva.fMeasure(minclassIndex));

		/*// ondo dagoela frogatzeko
		writer.write("\n frogatzeko ondo dagoela:");
		writer.write(eva.toClassDetailsString());

		// nahasmen matrizea
		writer.write(eva.toMatrixString());	
		 */

		// 10-fold cross validation
		writer.write("\n KALITATEAREN ESTIMAZIOA, 10-fold cross validation ebaluazio eskema");
		Evaluation eva2 = new Evaluation(train_dev);
		eva2.crossValidateModel(model, train_dev, 10, new Random(1));
		writer.write("\nKlase minoritarioren f-measure: " + eva2.fMeasure(minclassIndex));



		writer.close();

	}

	public static void main(String[] args) throws Exception {
		
		// fitx kargatu
		DataSource trainSource = new DataSource(args[0]);
		Instances train = trainSource.getDataSet();

		// klasea azken atributua da
		train.setClassIndex(train.numAttributes()-1);

		// fitx kargatu
		DataSource devSource = new DataSource(args[1]);
		Instances dev = devSource.getDataSet();

		// klasea azken atributua da
		dev.setClassIndex(4);
				
		inferentzia(train, dev, args[2],args[3]);


	}
	
	public static Instances merge(Instances data1, Instances data2)
            throws Exception
        {
            // Check where are the string attributes
            int asize = data1.numAttributes();
            boolean strings_pos[] = new boolean[asize];
            for(int i=0; i<asize; i++)
            {
                Attribute att = data1.attribute(i);
                strings_pos[i] = ((att.type() == Attribute.STRING) ||
                                  (att.type() == Attribute.NOMINAL));
            }

            // Create a new dataset
            Instances dest = new Instances(data1);
            dest.setRelationName(data1.relationName() + "+" + data2.relationName());

            DataSource source = new DataSource(data2);
            Instances instances = source.getStructure();
            Instance instance = null;
            while (source.hasMoreElements(instances)) {
                instance = source.nextElement(instances);
                dest.add(instance);

                // Copy string attributes
                for(int i=0; i<asize; i++) {
                    if(strings_pos[i]) {
                        dest.instance(dest.numInstances()-1)
                            .setValue(i,instance.stringValue(i));
                    }
                }
            }

            return dest;
        }

}
