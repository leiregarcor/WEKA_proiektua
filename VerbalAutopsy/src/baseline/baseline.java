package baseline;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class baseline {

	public static void main(String[] args) throws Exception {
		DataSource ds = new DataSource(args[0]);
		Instances data = ds.getDataSet();
		
		

	}

}
