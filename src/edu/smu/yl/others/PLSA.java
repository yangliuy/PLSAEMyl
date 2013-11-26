package edu.smu.yl.others;

//The code is taken from:
//http://code.google.com/p/mltool4j/source/browse/trunk/src/edu/thu/mltool4j/topicmodel/plsa
//I noticed some difference with original Hofmann concept in computation of P(z). It is 
//always even and actually not involved, that makes this algorithm non-negative matrix 
//factoring and not PLSA.
//Found and tested by Andrew Polar. 
//My version can be found on semanticsearchart.com or ezcodesample.com

public class PLSA {
	public static void main(String[] args) {
		ProbabilisticLSA plsa = new ProbabilisticLSA();
		//the file is not used, the hard coded data is used instead, but file name should be valid,
		//just replace the name by something valid.
		plsa.doPLSA("C:\\Users\\APolar\\workspace\\PLSA\\src\\data.txt", 2, 60);
        System.out.println("end PLSA");
    }
}
