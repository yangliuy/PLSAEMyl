package edu.smu.yl.plsa;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import edu.smu.yl.plsa.Documents.Document;

import edu.smu.yl.com.FileUtil;
import edu.smu.yl.conf.ParamConfig;
import edu.smu.yl.conf.PathConfig;

/**
 * Class for PLSA model using EM
 * 
 * @author yangliu
 * @blog http://blog.csdn.net/yangliuy
 * @mail yangliuyx@gmail.com
 */
public class Plsa {

	private int iters;
	
	private int saveStep;
	
	private int beginSaveIters;
	
    private int topicNum;

    private int N; // number of docs

    private int M; // number of terms

    private int[][] docTermMatrix; // docTermMatrix
    
    private double[][] docTopicPros;//p(z|d)

    private double[][] topicTermPros;//p(w|z)

    private double[][][] docTermTopicPros;//p(z|d,w)


    public Plsa() {
    	//init parameters
        topicNum = ParamConfig.topicNum;
        iters = ParamConfig.iteration;
        saveStep = ParamConfig.saveStep;
        beginSaveIters = ParamConfig.beginSaveIters;
    }
    
    public void initializeModel(Documents docSet) {
    	if (docSet == null) {
            throw new IllegalArgumentException("The documents set must be not null!");
        }

    	N = docSet.docs.size();
    	M = docSet.indexToTermMap.size();
    	
        //element in docTermMatrix represents times the word appear in this document 
        docTermMatrix = new int[N][M];
        //init docTermMatrix
        for (int docIndex = 0; docIndex < N; docIndex++) {
            Document doc = docSet.docs.get(docIndex);
            for (int word = 0; word < doc.docWords.length; word++) {
                int termIndex = docSet.docs.get(docIndex).docWords[word];
                docTermMatrix[docIndex][termIndex] += 1;
            }
        }

        docTopicPros = new double[N][topicNum];
        topicTermPros = new double[topicNum][M];
        docTermTopicPros = new double[N][M][topicNum];

        //init p(z|d),for each document the constraint is sum(p(z|d))=1.0
        for (int i = 0; i < N; i++) {
            double[] pros = randomProbilities(topicNum);
            for (int j = 0; j < topicNum; j++) {
                docTopicPros[i][j] = pros[j];
            }
        }
        //init p(w|z),for each topic the constraint is sum(p(w|z))=1.0
        for (int i = 0; i < topicNum; i++) {
            double[] pros = randomProbilities(M);
            for (int j = 0; j < M; j++) {
                topicTermPros[i][j] = pros[j];
            }
        }
    }

    /**
     * 
     * Plsa inference using EM
     * 
     * @param docs all documents
     * @throws IOException 
     */
    public void inferenceModel(Documents docSet) throws IOException {
        //use em to estimate params
        //save model according to model parameters
        
        if (iters < saveStep + beginSaveIters) {
			System.err.println("Error: the number of iterations should be larger than "
							+ (saveStep + beginSaveIters));
			System.exit(0);
		}
		for (int i = 0; i < iters; i++) {
			System.out.println("Interation " + i + " -------------- ");
			if ((i >= beginSaveIters)
					&& (((i - beginSaveIters) % saveStep) == 0)) {
				// Saving the model
				System.out.println("Saving model at iteration " + i + " ... ");
				saveIteratedModel(i, docSet);
			}
            em();
            System.out.println("After E step and M step, the new log likelihood is " + computeLogLikelihood(docSet));
        }

        System.out.println("done");
    }

    private double computeLogLikelihood(Documents docSet) {
		// TODO Auto-generated method stub
    	 /*
         * Compute the log likelihood of generation the corpus
         * 
         * L = sum_i {n(d_i) * [sum_j( n(d_i, w_j) / n(d_i) * log sum_k p(z_k|d_i)*p(w_j|z_k))]}
         * 
         */
    	 double L = 0.0;
    	 for (int i = 0; i < N; i++) {
    		 double docISize = docSet.docs.get(i).docWords.length;
    		 double sumM = 0.0;
    		 for (int j = 0; j < M; j++) { 
    			 double sumK = 0.0;
    			 for(int k = 0; k < topicNum; k++){
    				 sumK += docTopicPros[i][k] * topicTermPros[k][j];
    			 }
    			 //System.out.println("sumK: " + sumK);
    			 sumM += (double)docTermMatrix[i][j] / docISize * Math.log10(sumK);
    		 }
    		 //System.out.println("sumM: " + sumM);
             L += docISize * sumM;
    	 }
		return L;
	}

	/**
     * 
     * EM algorithm
     * 
     */
    private void em() {
        /*
         * E-step,calculate posterior probability p(z|d,w,&),& is
         * model params(p(z|d),p(w|z))
         * 
         * p(z|d,w,&)=p(z|d)*p(w|z)/sum(p(z'|d)*p(w|z'))
         * z' represent all posible topic
         * 
         */
        for (int docIndex = 0; docIndex < N; docIndex++) {
            for (int wordIndex = 0; wordIndex < M; wordIndex++) {
                double total = 0.0;
                double[] perTopicPro = new double[topicNum];
                for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
                    double numerator = docTopicPros[docIndex][topicIndex]
                            * topicTermPros[topicIndex][wordIndex];
                    total += numerator;
                    perTopicPro[topicIndex] = numerator;
                }

                if (total == 0.0) {
                    total = avoidZero(total);
                }

                for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
                    docTermTopicPros[docIndex][wordIndex][topicIndex] = perTopicPro[topicIndex]
                            / total;
                }
            }
        }

        //M-step
        /*
         * update p(w|z),p(w|z)=sum(n(d',w)*p(z|d',w,&))/sum(sum(n(d',w')*p(z|d',w',&)))
         * 
         * d' represent all documents
         * w' represent all vocabularies
         * 
         * 
         */
        for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
            double totalDenominator = 0.0;
            for (int wordIndex = 0; wordIndex < M; wordIndex++) {
                double numerator = 0.0;
                for (int docIndex = 0; docIndex < N; docIndex++) {
                    numerator += docTermMatrix[docIndex][wordIndex]
                            * docTermTopicPros[docIndex][wordIndex][topicIndex];
                }

                topicTermPros[topicIndex][wordIndex] = numerator;

                totalDenominator += numerator;
            }

            if (totalDenominator == 0.0) {
                totalDenominator = avoidZero(totalDenominator);
            }

            for (int wordIndex = 0; wordIndex < M; wordIndex++) {
                topicTermPros[topicIndex][wordIndex] = topicTermPros[topicIndex][wordIndex]
                        / totalDenominator;
            }
        }
        /*
         * update p(z|d),p(z|d)=sum(n(d,w')*p(z|d,w'&))/sum(sum(n(d,w')*p(z'|d,w',&)))
         * 
         * w' represent all vocabularies
         * z' represnet all topics
         * 
         */
        for (int docIndex = 0; docIndex < N; docIndex++) {
            //actually equal sum(w) of this doc
            double totalDenominator = 0.0;
            for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
                double numerator = 0.0;
                for (int wordIndex = 0; wordIndex < M; wordIndex++) {
                    numerator += docTermMatrix[docIndex][wordIndex]
                            * docTermTopicPros[docIndex][wordIndex][topicIndex];
                }
                docTopicPros[docIndex][topicIndex] = numerator;
                totalDenominator += numerator;
            }

            if (totalDenominator == 0.0) {
                totalDenominator = avoidZero(totalDenominator);
            }

            for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
                docTopicPros[docIndex][topicIndex] = docTopicPros[docIndex][topicIndex]
                        / totalDenominator;
            }
        }
    }

    /**
     * 
     * 
     * Get a normalize array
     * 
     * @param size
     * @return
     */
    public double[] randomProbilities(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("The size param must be greate than zero");
        }
        double[] pros = new double[size];

        int total = 0;
        Random r = new Random();
        for (int i = 0; i < pros.length; i++) {
            //avoid zero
            pros[i] = r.nextInt(size) + 1;

            total += pros[i];
        }

        //normalize
        for (int i = 0; i < pros.length; i++) {
            pros[i] = pros[i] / total;
        }

        return pros;
    }

    /**
     * 
     * @return
     */
    public double[][] getDocTopics() {
        return docTopicPros;
    }

    /**
     * 
     * @return
     */
    public double[][] getTopicWordPros() {
        return topicTermPros;
    }

    /**
     * 
     * Get topic number
     * 
     * 
     * @return
     */
    public Integer getTopicNum() {
        return topicNum;
    }

    /**
     * 
     * avoid zero number.if input number is zero, we will return a magic
     * number.
     * 
     * 
     */
    private final static double MAGICNUM = 0.0000000000000001;

    public double avoidZero(double num) {
        if (num == 0.0) {
            return MAGICNUM;
        }

        return num;
    }
    
    public void saveIteratedModel(int iteration, Documents docSet) throws IOException {
		// TODO Auto-generated method stub
		// model.params model.theta model.phi model.psi model.varphi model.tau
		String resPath = PathConfig.resDataPath +"model_" + iteration;
		FileUtil.write2DArray(docTopicPros, resPath + ".docTopicPros");
		FileUtil.write2DArray(topicTermPros, resPath + ".topicTermPros");

		// model.zassign

		int topNum = ParamConfig.topTopicWordNum;
		// model.zterms
		ArrayList<String> ztermsLines = new ArrayList<String>();
		for (int i = 0; i < topicNum; i++) {
			List<Integer> tWordsIndexArray = new ArrayList<Integer>();
			for (int w = 0; w < M; w++) {
				tWordsIndexArray.add(new Integer(w));
			}
			Collections.sort(tWordsIndexArray,
					new Plsa.TwordsComparable(topicTermPros[i]));
			String line = "topic=" + i + "\t";
			for (int w = 0; w < topNum; w++) {
				line += docSet.indexToTermMap.get(tWordsIndexArray.get(w)) + "\t";
			}
			ztermsLines.add(line);
		}
		FileUtil.writeLines(resPath + ".zterms", ztermsLines);
	}

	public class TwordsComparable implements Comparator<Integer> {
		public double[] sortProb; // Store probability of each word in topic k

		public TwordsComparable(double[] sortProb) {
			this.sortProb = sortProb;
		}

		@Override
		public int compare(Integer o1, Integer o2) {
			// TODO Auto-generated method stub
			// Sort topic word index according to the probability of each word
			// in topic k
			if (sortProb[o1] > sortProb[o2])
				return -1;
			else if (sortProb[o1] < sortProb[o2])
				return 1;
			else
				return 0;
		}
	}
}
