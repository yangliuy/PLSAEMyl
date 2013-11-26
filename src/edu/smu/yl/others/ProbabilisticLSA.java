package edu.smu.yl.others;

import java.io.File;
import java.util.ArrayList;

class ProbabilisticLSA
{
	private Dataset dataset = null;
    private Posting[][] invertedIndex = null;
    private int M = -1; // number of data
    private int V = -1; // number of words
    private int K = -1; // number of topics

    public ProbabilisticLSA()
    {
    }

    public boolean doPLSA(String datafileName, int ntopics, int iters)
    {
    	File datafile = new File(datafileName);
        if (datafile.exists())
        {
        	if ((this.dataset = new Dataset(datafile)) == null)
            {
        		System.out.println("doPLSA, dataset == null");
                return false;
            }
            this.M = this.dataset.size();
            this.V = this.dataset.getFeatureNum();
            this.K = ntopics;
            
             //build inverted index
            this.buildInvertedIndex(this.dataset);
            //run EM algorithm
            this.EM(iters);
            return true;
        }
        else
        {
        	System.out.println("ProbabilisticLSA(String datafileName), datafile: " + datafileName + " doesn't exist");
            return false;
        }
    }

    //Build the inverted index for M-step fast calculation. Format:
    //invertedIndex[w][]: a unsorted list of document and position which word w
    // occurs. 
    //@param ds the dataset which to be analysis
    @SuppressWarnings("unchecked")
    private boolean buildInvertedIndex(Dataset ds)
    {
    	ArrayList<Posting>[] list = new ArrayList[this.V];
        for (int k=0; k<this.V; ++k) {
        	list[k] = new ArrayList<Posting>();
        }
        	
        for (int m = 0; m < this.M; m++)
        {
        	Data d = ds.getDataAt(m);
         	for (int position = 0; position < d.size(); position++)
         	{
        		int w = d.getFeatureAt(position).dim;
         		// add posting
         		list[w].add(new Posting(m, position));
        	}
        }
        // convert to array
        this.invertedIndex = new Posting[this.V][];
        for (int w = 0; w < this.V; w++)
        {
        	this.invertedIndex[w] = list[w].toArray(new Posting[0]);
        }
        return true;
    }

    private boolean EM(int iters)
    {
    	// p(z), size: K
        double[] Pz = new double[this.K];

        // p(d|z), size: K x M
        double[][] Pd_z = new double[this.K][this.M];

        // p(w|z), size: K x V
        double[][] Pw_z = new double[this.K][this.V];

        // p(z|d,w), size: K x M x doc.size()
        double[][][] Pz_dw = new double[this.K][this.M][];

         // L: log-likelihood value
         double L = -1;

         // run EM algorithm
         this.init(Pz, Pd_z, Pw_z, Pz_dw);
         for (int it = 0; it < iters; it++)
         {
         	 // E-step
        	 if (!this.Estep(Pz, Pd_z, Pw_z, Pz_dw))
        	 {
        		 System.out.println("EM,  in E-step");
             }

             // M-step
             if (!this.Mstep(Pz_dw, Pw_z, Pd_z, Pz))
             {
            	 System.out.println("EM, in M-step");
             }

             L = calcLoglikelihood(Pz, Pd_z, Pw_z);
             System.out.println("[" + it + "]" + "\tlikelihood: " + L);
         }
                
         //print result
         for (int m = 0; m < this.M; m++)
         {
        	 double norm = 0.0;
        	 for (int z = 0; z < this.K; z++) {
        		 norm += Pd_z[z][m];
             }
        	 if (norm <= 0.0) norm = 1.0;
        	 for (int z = 0; z < this.K; z++) {
        		 System.out.format("%10.4f", Pd_z[z][m]/norm);
             }
             System.out.println();
        } 
        return false;
    }
   
    private boolean init(double[] Pz, double[][] Pd_z, double[][] Pw_z, double[][][] Pz_dw)
    {
    	// p(z), size: K
        double zvalue = (double) 1 / (double) this.K;
        for (int z = 0; z < this.K; z++)
        {
        	Pz[z] = zvalue;
        }

        // p(d|z), size: K x M
        for (int z = 0; z < this.K; z++)
        {
        	double norm = 0.0;
            for (int m = 0; m < this.M; m++)
            {
            	Pd_z[z][m] = Math.random();
                norm += Pd_z[z][m];
            }

            for (int m = 0; m < this.M; m++)
            {
            	Pd_z[z][m] /= norm;
            }
        }

        // p(w|z), size: K x V
        for (int z = 0; z < this.K; z++)
        {
        	double norm = 0.0;
            for (int w = 0; w < this.V; w++)
            {
            	Pw_z[z][w] = Math.random();
                norm += Pw_z[z][w];
            }

            for (int w = 0; w < this.V; w++)
            {
            	Pw_z[z][w] /= norm;
            }
        }

        // p(z|d,w), size: K x M x doc.size()
        for (int z = 0; z < this.K; z++)
        {
        	for (int m = 0; m < this.M; m++)
            {
         		Pz_dw[z][m] = new double[this.dataset.getDataAt(m).size()];
            }
        }
        return false;
    }

    private boolean Estep(double[] Pz, double[][] Pd_z, double[][] Pw_z, double[][][] Pz_dw)
    {
    	for (int m = 0; m < this.M; m++)
        {
    		Data data = this.dataset.getDataAt(m);
            for (int position = 0; position < data.size(); position++)
            {
            	// get word(dimension) at current position of document m
                int w = data.getFeatureAt(position).dim;

                double norm = 0.0;
                for (int z = 0; z < this.K; z++)
                {
                	double val = Pz[z] * Pd_z[z][m] * Pw_z[z][w];
                    Pz_dw[z][m][position] = val;
                    norm += val;
                }

                // normalization
                for (int z = 0; z < this.K; z++)
                {
                	Pz_dw[z][m][position] /= norm;
                }
            }
        }
        return true;
    }

    private boolean Mstep(double[][][] Pz_dw, double[][] Pw_z, double[][] Pd_z, double[] Pz)
    {
    	// p(w|z)
        for (int z = 0; z < this.K; z++)
        {
        	double norm = 0.0;
            for (int w = 0; w < this.V; w++)
            {
            	double sum = 0.0;

                Posting[] postings = this.invertedIndex[w];
                for (Posting posting : postings)
                {
                	int m = posting.docID;
                    int position = posting.pos;
                    double n = this.dataset.getDataAt(m).getFeatureAt(position).weight;
                    sum += n * Pz_dw[z][m][position];
                }
                Pw_z[z][w] = sum;
                norm += sum;
            }

            // normalization
            for (int w = 0; w < this.V; w++)
            {
            	Pw_z[z][w] /= norm;
            }
        }

        // p(d|z)
        for (int z = 0; z < this.K; z++)
        {
        	double norm = 0.0;
            for (int m = 0; m < this.M; m++)
            {
            	double sum = 0.0;
                Data d = this.dataset.getDataAt(m);
                for (int position = 0; position < d.size(); position++)
                {
                	double n = d.getFeatureAt(position).weight;
                    sum += n * Pz_dw[z][m][position];
                }
                Pd_z[z][m] = sum;
                norm += sum;
            }

            // normalization
            for (int m = 0; m < this.M; m++)
            {
            	Pd_z[z][m] /= norm;
            }
        }

        //This is definitely a bug
        //p(z) values are even, but they should not be even
        double norm = 0.0;
        for (int z = 0; z < this.K; z++)
        {
        	double sum = 0.0;
            for (int m = 0; m < this.M; m++)
            {
            	sum += Pd_z[z][m];
            }
            Pz[z] = sum;
            norm += sum;
       }

        // normalization
        for (int z = 0; z < this.K; z++)
        {
        	Pz[z] /= norm;
        	//System.out.format("%10.4f", Pz[z]);  //here you can print to see
        }
        //System.out.println();

        return true;
    }

    private double calcLoglikelihood(double[] Pz, double[][] Pd_z, double[][] Pw_z)
    {
    	double L = 0.0;
        for (int m = 0; m < this.M; m++)
        {
        	Data d = this.dataset.getDataAt(m);
            for (int position = 0; position < d.size(); position++)
            {
            	Feature f = d.getFeatureAt(position);
                int w = f.dim;
                double n = f.weight;

                double sum = 0.0;
                for (int z = 0; z < this.K; z++)
                {
                	sum += Pz[z] * Pd_z[z][m] * Pw_z[z][w];
                }
                L += n * Math.log10(sum);
            }
        }
        return L;
    }
}