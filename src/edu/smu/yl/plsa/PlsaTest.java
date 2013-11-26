package edu.smu.yl.plsa;

import java.io.IOException;

import edu.smu.yl.conf.ParamConfig;
import edu.smu.yl.conf.PathConfig;

/**
 * Test Class for PLSA model using EM algorithm
 * 
 * @author yangliu
 * @blog http://blog.csdn.net/yangliuy
 * @mail yangliuyx@gmail.com
 */

public class PlsaTest {

    public static void main(String[] args) throws IOException {
    	Documents docSet = new Documents();
    	System.out.println("0 Read Docs ...");
    	docSet.readDocs(PathConfig.oriDataPath);
    	System.out.println("docSet: " + docSet.docs.size());
		Plsa model = new Plsa();
    	System.out.println("1 Initialize the model ...");
		model.initializeModel(docSet);
		System.out.println("2 Learning and Saving the model ...");
		model.inferenceModel(docSet);
		System.out.println("3 Output the final model ...");
		model.saveIteratedModel(ParamConfig.iteration, docSet);
		System.out.println("Done!");
    }
}
