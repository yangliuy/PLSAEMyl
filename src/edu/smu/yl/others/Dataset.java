package edu.smu.yl.others;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

class Dataset
{
        private ArrayList<Data> datas;

        // total number of data
        private int dataNum = -1;

        // total number of distinct features
        private int featureNum = -1;

        public Dataset()
        {
        	this.datas = new ArrayList<Data>();
            refreshStatistics();
        }

        public Dataset(ArrayList<Data> initDatas)
        {
        	this.datas = initDatas;
            refreshStatistics();
        }

// this is data matrix        
//	{9.0, 2.0, 1.0, 1.0, 1.0, 0.0},   
//	{8.0, 3.0, 2.0, 1.0, 0.0, 0.0},   
//	{3.0, 0.0, 0.0, 1.0, 2.0, 8.0},
//	{0.0, 1.0, 0.0, 2.0, 4.0, 7.0},
//	{2.0, 1.0, 1.0, 0.0, 1.0, 3.0},        
        
        public Dataset(File datafile)
        {
        	this.datas = new ArrayList<Data>();
            try
            {
            	//this is format for data matrix
            	String[] lines = {
            			"-1 0:9 1:2 2:1 3:1 4:1", 
                		"-2 0:8 1:3 2:2 3:1",  
                		"-3 0:3 3:1 4:2 5:8",
                		"-4 1:1 3:2 4:4 5:7",
                		"-5 0:3 1:2 2:2 4:1 5:6"
                		};
                //old operation reading from file
                //List<String> datalines = FileUtils.readLines(datafile);
                //is replaced by hard coded data for testing 
                List<String> datalines = Arrays.asList(lines);  
                for (int i = 0; i < datalines.size(); i++)
                {
                	datas.add(new Data(i, datalines.get(i)));
                }
                refreshStatistics();
            }
            catch (Exception e)
            {
            	System.out.println("Dataset(File datafile): " + e.getMessage());
                e.printStackTrace();
            }
        }

        private void refreshStatistics()
        {
        	this.dataNum = this.datas.size();
            HashSet<Integer> calculator = new HashSet<Integer>();
            for (Data d : datas)
            {
           	 	for (Feature f : d.getAllFeature())
                {
           	 		calculator.add(f.dim);
                }
            }
            this.featureNum = calculator.size();
        }

        public Data getDataAt(int index)
        {
        	return datas.get(index);
        }

        public ArrayList<Data> getAllData()
        {
        	return datas;
        }

        public int size()
        {
            return this.dataNum;
        }

        public int getFeatureNum()
        {
            return this.featureNum;
        }
}
