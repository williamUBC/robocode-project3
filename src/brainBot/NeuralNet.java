package brainBot;

import java.io.*;

public class NeuralNet {
	public int argNumInputs;
    public int argNumHidden;
    public int argNumOutput;
    private double argLearningRate = 0.0001;
    private double argMomentumTerm = 0.9;
    private double argA = -1.0;
    private double argB = 1.0;
    public double[][] weightsInputToHidden;
    public double[][] weightsHiddenToOutput;
    static double bias = 1;
    private double[] hiddenValues;
    private double[] outputValue;
    private boolean biPolar = true;//T means biPolar; F means binary
    double deltaWeightInToHid[][];
    double deltaWeightHidToOut[][];
    //    private double[][] inputValue = {
//            {0, 0},
//            {0, 1},
//            {1, 0},
//            {1, 1}
//    };
    //constructor
    public NeuralNet(int argNumInputs, int argNumHidden, int argNumOutput) {
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;
        this.argNumOutput = argNumOutput;
        this.weightsInputToHidden = new double[argNumInputs + 1][argNumHidden];
        this.weightsHiddenToOutput = new double[argNumHidden + 1][argNumOutput];
        this.hiddenValues = new double[argNumHidden];
        this.outputValue = new double[argNumOutput];
        this.deltaWeightInToHid = new double[argNumInputs+1][argNumHidden];
        this.deltaWeightHidToOut = new double[argNumHidden+1][argNumOutput];
        //this.initializeWeights();
    }


    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }


    public double customSigmoid(double x) {
        if(biPolar){
            return (argB - argA)/(1 + Math.exp(-x)) + argA;
        }else{
            return (1)/(1 + Math.exp(-x));
        }
    }


    public void initializeWeights() {
//        for (int i = 0; i < argNumInputs + 1; i++) {
//            for (int j = 0; j < argNumHidden; j++) {
//                this.weightsInputToHidden[i][j] = Math.random() - 0.5;
//                //System.out.println(this.weightsInputToHidden[i][j]);
//            }
//        }
//
//        for (int i = 0; i < argNumHidden + 1; i++) {
//            for (int j = 0; j < argNumOutput; j++) {
//                this.weightsHiddenToOutput[i][j] = Math.random() - 0.5;
//                //System.out.println(this.weightsHiddenToOutput[i][j]);
//            }
//        }
        for(int i = 0; i<argNumInputs+1; i++){
            for (int j=0; j<argNumHidden; j++){
                this.weightsInputToHidden[i][j] = Math.random() - 0.5;
                //System.out.println(this.weightsInputToHidden[i][j]);
            }
        }

        for(int i = 0; i<argNumHidden+1; i++){
            for (int j=0; j<argNumOutput; j++){
                this.weightsHiddenToOutput[i][j] = Math.random() - 0.5;
                //System.out.println(this.weightsHiddenToOutput[i][j]);
            }
        }

        for (int hiddenNodeIndex=0; hiddenNodeIndex<argNumHidden; hiddenNodeIndex++){
            for(int inputNodeIndex=0; inputNodeIndex<argNumInputs+1; inputNodeIndex++){
                deltaWeightInToHid[inputNodeIndex][hiddenNodeIndex] = 0.0;
            }
        }

        for(int outputNodeIndex=0; outputNodeIndex<argNumOutput; outputNodeIndex++){
            for (int hiddenNodeIndex=0; hiddenNodeIndex<argNumHidden+1; hiddenNodeIndex++){
                deltaWeightHidToOut[hiddenNodeIndex][outputNodeIndex] = 0.0;
            }
        }
    }


    public void zeroWeights() {
        for(int i = 0; i<this.argNumInputs+1; i++){
            for (int j=0; j<this.argNumHidden; j++){
                this.weightsInputToHidden[i][j] = 0.5;
                //System.out.println(this.weightsInputToHidden[i][j]);
            }
        }

        for(int i = 0; i<this.argNumHidden+1; i++){
            for (int j=0; j<this.argNumOutput; j++){
                this.weightsHiddenToOutput[i][j] = 0.5;
                //System.out.println(this.weightsHiddenToOutput[i][j]);
            }
        }
    }

    public double[] forwardFeed(double[] input) {
        //hiddenValues[] = new double[argNumHidden];
        //double outputValue[] = new double[argNumOutput];
        //this.inputValue = input;
        if(input.length != argNumInputs){
            throw new ArrayIndexOutOfBoundsException();
        }else{
            //calculate value for nodes in hidden layer
            for (int hiddenNodeIndex=1; hiddenNodeIndex<argNumHidden+1; hiddenNodeIndex++){
                hiddenValues[hiddenNodeIndex-1] = 0.0;
                for(int inputNodeIndex=1; inputNodeIndex<argNumInputs+1; inputNodeIndex++){
                    hiddenValues[hiddenNodeIndex-1] += input[inputNodeIndex-1] * weightsInputToHidden[inputNodeIndex][hiddenNodeIndex-1];
                }
                hiddenValues[hiddenNodeIndex-1] += weightsInputToHidden[0][hiddenNodeIndex-1];
                hiddenValues[hiddenNodeIndex-1] = customSigmoid(hiddenValues[hiddenNodeIndex-1]);
            }
            //calculate value for nodes in output layer
            for(int outputNodeIndex=0; outputNodeIndex<argNumOutput; outputNodeIndex++){
                outputValue[outputNodeIndex] = 0.0;
                for (int hiddenNodeIndex=1; hiddenNodeIndex<argNumHidden+1; hiddenNodeIndex++){
                    outputValue[outputNodeIndex] += hiddenValues[hiddenNodeIndex-1] * weightsHiddenToOutput[hiddenNodeIndex][outputNodeIndex];
                }
                outputValue[outputNodeIndex] += weightsHiddenToOutput[0][outputNodeIndex];
                //outputValue[outputNodeIndex] = customSigmoid(outputValue[outputNodeIndex]);
                if(outputValue[outputNodeIndex]<0) {
                	outputValue[outputNodeIndex] = 0;
                }
            }
        }
        return outputValue;
    }


    public void backPropagation(double[] expectedOut, double[] actualOut, double[] inputValue) {
        //double delta_OutputToHidden[] = new double[argNumOutput];
        double delta_Hidden[] = new double[argNumHidden];
        double delta_Output[] = new double[argNumOutput];
        double delta_Output_beforeSigmoid[] = new double[argNumOutput];
//        double deltaWeightInToHid[][] = new double[argNumInputs+1][argNumHidden];
//        double deltaWeightHidToOut[][] = new double[argNumHidden+1][argNumOutput];
        //double updateAmountIH[][] = new double[argNumInputs][argNumHidden];
        //double updateAmountHO[][] = new double[argNumHidden][argNumOutput];
        if(actualOut.length!=argNumOutput && expectedOut.length!=argNumOutput){
            throw new ArrayIndexOutOfBoundsException();
        }else{
            //get delta at output nodes

            for(int j=0; j<argNumOutput; j++){
                //delta_Output_beforeSigmoid[j] = 0.5 * Math.pow(expectedOut[j] - actualOut[j],2);
                delta_Output_beforeSigmoid[j] = expectedOut[j] - actualOut[j];
                if(biPolar){
                    //delta_Output[j] = (actualOut[j] + 1) * 0.5 * (1 - actualOut[j]) * delta_Output_beforeSigmoid[j];
                	//delta_Output[j] = delta_Output_beforeSigmoid[j];
                    if(actualOut[j]>0) {
                    	delta_Output[j] = delta_Output_beforeSigmoid[j];
                    }else {
                    	delta_Output[j] = 0;
					}
                }else{
                    delta_Output[j] = actualOut[j] * (1 - actualOut[j]) * delta_Output_beforeSigmoid[j];
                }

            }

            //get delta at hidden nodes
            for(int i=0; i<argNumHidden; i++){
                delta_Hidden[i] = 0.0;
                for(int j=0; j<argNumOutput; j++){
                    if(biPolar){
                        delta_Hidden[i] += (hiddenValues[i] + 1) * 0.5 * (1 - hiddenValues[i]) * delta_Output_beforeSigmoid[j] * weightsHiddenToOutput[i+1][j];
                    }else{
                        delta_Hidden[i] += hiddenValues[i] * (1 - hiddenValues[i]) * delta_Output_beforeSigmoid[j] * weightsHiddenToOutput[i+1][j];
                    }

                    //System.out.println(actualOut[j] * (1 - actualOut[j]));
                }
            }

            //update weights
            //calculate value for nodes in hidden layer
            for (int hiddenNodeIndex=0; hiddenNodeIndex<argNumHidden; hiddenNodeIndex++){
                for(int inputNodeIndex=1; inputNodeIndex<argNumInputs+1; inputNodeIndex++){
                    deltaWeightInToHid[inputNodeIndex][hiddenNodeIndex] = argLearningRate * delta_Hidden[hiddenNodeIndex] * inputValue[inputNodeIndex-1] + argMomentumTerm * deltaWeightInToHid[inputNodeIndex][hiddenNodeIndex];
                    weightsInputToHidden[inputNodeIndex][hiddenNodeIndex] += deltaWeightInToHid[inputNodeIndex][hiddenNodeIndex];
                }
                weightsInputToHidden[0][hiddenNodeIndex] += argLearningRate * delta_Hidden[hiddenNodeIndex] * bias;
                //System.out.println(weightsInputToHidden[0][hiddenNodeIndex]);
            }

            //calculate value for nodes in output layer

            for(int outputNodeIndex=0; outputNodeIndex<argNumOutput; outputNodeIndex++){
                for (int hiddenNodeIndex=1; hiddenNodeIndex<argNumHidden+1; hiddenNodeIndex++){
                    deltaWeightHidToOut[hiddenNodeIndex][outputNodeIndex] = argLearningRate * delta_Output[outputNodeIndex] * hiddenValues[hiddenNodeIndex-1] + argMomentumTerm * deltaWeightHidToOut[hiddenNodeIndex][outputNodeIndex];
                    weightsHiddenToOutput[hiddenNodeIndex][outputNodeIndex] += deltaWeightHidToOut[hiddenNodeIndex][outputNodeIndex];
                }
                weightsHiddenToOutput[0][outputNodeIndex] += argLearningRate * delta_Output[outputNodeIndex] * bias;
                //System.out.println(weightsHiddenToOutput[0][outputNodeIndex]);
            }
            //momentum
        }


    }


    public double outputFor(double[] X) {
        return 0;
    }


    public double train(double[] X, double argValue) {
        return 0;
    }


    public void save(File argFile) {

    }


    public void load(String argFileName) throws IOException {

    }

//    public static void train(String[] args) {
//        NeuralNet myNN = new NeuralNet(2, 10, 1);
//        double x_train[][] = {
//                {0, 0},
//                {0, 1},
//                {1, 0},
//                {1, 1}
//        };
//        double y_label[][] = {
//                {0},
//                {1},
//                {1},
//                {0}
//        };
//        //int epoch = 0;
//
//        myNN.initializeWeights();
//        double[] res = new double[0];
//        double myError = 0.0;
//        double epoch = 0.0;
//        //System.out.println(x_train[1][1]);
//        //myNN.zeroWeights();
//        //for (int epoch = 0; epoch < 50000; epoch++) {
//        do {
//            for (int i = 0; i < 4; i++) {
//                res = myNN.forwardFeed(x_train[i]);
//                myError = 0.0;
//                //System.out.println(y_label[i][0]);
//                //System.out.println(x_train[i][1]);
//                myNN.backPropagation(y_label[i], res, x_train[i]);
//                myError += 0.5 * Math.pow((res[0] - y_label[i][0]), 2);
//                //System.out.println(res[0]);
//
//            }
//            epoch += 1;
//            //System.out.println(res[0]);
//            //System.out.println("Epoch: " + epoch + "    Error: " + myError);
//            System.out.println(myError);
//        } while (myError > 0.01 && epoch < 500000);
//    }

   public void loadWeightsItoH(File file) throws IOException {
//       BufferedReader in = new BufferedReader(new FileReader("weightsInToHid.txt"));  //
//       String line;  //一行数据
//       int row=0;
//       //逐行读取，并将每个数组放入到数组中
//       while((line = in.readLine()) != null){
////           String temp = line.split("\t");
//           double temp = Double.parseDouble(line);
////           for(int j=0;j<temp.length;j++){
////               arr2[row][j] = Double.parseDouble(temp[j]);
////           }
//           for (int i=0;i<argNumInputs+1;i++){
//               for(int j=0;j<argNumHidden;j++){
//                   weightsInputToHidden[i][j] = temp;
//               }
//           }
////           row++;
//       }
//       in.close();
//
//       in = new BufferedReader(new FileReader("weightsHidToOut.txt"));  //
//       //逐行读取，并将每个数组放入到数组中
//       while((line = in.readLine()) != null){
////           String temp = line.split("\t");
//           double temp = Double.parseDouble(line);
////           for(int j=0;j<temp.length;j++){
////               arr2[row][j] = Double.parseDouble(temp[j]);
////           }
//           for (int i=0;i<argNumHidden+1;i++){
//               for(int j=0;j<argNumOutput;j++){
//                   weightsHiddenToOutput[i][j] = temp;
//               }
//           }
////           row++;
//       }
//       in.close();
       
       
       BufferedReader read = null; 
	    try 
	    { 
	    	//File file = new File("weightsInToHid.txt"); 
	    	read = new BufferedReader(new FileReader(file)); 	      
	    	    	
	    	for (int i=0;i<argNumInputs+1;i++){
              for(int j=0;j<argNumHidden;j++){
                  weightsInputToHidden[i][j] = Double.parseDouble(read.readLine());
              }
          }
        }
	    catch (IOException e) 
	    { 
	      System.out.println("IOException trying to open reader: " + e); 
	      //initialiseLUT(); 
	      nothing();
	    } 
	    catch (NumberFormatException e) 
	    { 
	      //initialiseLUT(); 
	    	nothing();
	    	System.out.println("IOException trying to open reader: " + e);
	    } 
	    finally 
	    { 
	      try 
	      { 
	        if (read != null) 
	        	read.close(); 
	        	
	      } 
	      catch (IOException e) 
	      { 
	        System.out.println("IOException trying to close reader: " + e); 
	      } 
	    } 
	    
	    //BufferedReader read = null; 
	    
   }
   public void loadWeightsHtoO(File file) throws IOException {     
     BufferedReader read = null; 
	    try 
	    { 
	    	//File file = new File("weightsInToHid.txt"); 
	    	read = new BufferedReader(new FileReader(file)); 
	      
	    	for (int i=0;i<argNumHidden+1;i++){
	    		for(int j=0;j<argNumOutput;j++){
	    			weightsHiddenToOutput[i][j] = Double.parseDouble(read.readLine());
	    		}
	    	}	    	
	    	
      }
	    catch (IOException e) 
	    { 
	      System.out.println("IOException trying to open reader: " + e); 
	      //initialiseLUT(); 
	      nothing();
	    } 
	    catch (NumberFormatException e) 
	    { 
	      //initialiseLUT(); 
	    	nothing();
	    	System.out.println("IOException trying to open reader: " + e);
	    } 
	    finally 
	    { 
	      try 
	      { 
	        if (read != null) 
	        	read.close(); 
	        	
	      } 
	      catch (IOException e) 
	      { 
	        System.out.println("IOException trying to close reader: " + e); 
	      } 
	    } 
	    
	    //BufferedReader read = null; 
	    
 }
   public void nothing() {
	   
   }
   
    public void saveWeights(int weightsKind) {
        System.out.println("Saving weights...");
        try {
            //BufferedWriter out = new BufferedWriter(new FileWriter(argFile));
            if (weightsKind==0){//weightsKind==0 means it is the weights for inputnodes to hidden nodes
                File file = new File("weightsInToHid.dat");  //存放数组数据的文件
                FileWriter out = new FileWriter(file);  //文件写入流
                for (int i=0; i<argNumInputs+1; i++){
                    for (int j=0; j<argNumHidden; j++){
                        out.write(String.valueOf(weightsInputToHidden[i][j])+"\n");
                    }
                }
                out.close();
            }else{
                File file = new File("weightsHidToOut.dat");
                FileWriter out = new FileWriter(file);  //文件写入流
                for (int i=0; i<argNumHidden+1; i++){
                    for (int j=0; j<argNumOutput; j++){
                        out.write(String.valueOf(weightsHiddenToOutput[i][j])+"\n");
                    }
                }
                out.close();
            }
            System.out.println("文件创建成功！");
        } catch (IOException e) {
        }
    }
}
