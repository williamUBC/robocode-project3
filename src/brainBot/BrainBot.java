package brainBot;

import java.awt.Color;
import java.io.BufferedReader;
import java.io.Console;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;



import robocode.AdvancedRobot;
import robocode.BattleEndedEvent;
import robocode.BulletHitEvent;
import robocode.BulletMissedEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.HitWallEvent;
import robocode.RobocodeFileOutputStream;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class BrainBot extends AdvancedRobot{
	//hyper parameters
	int maxNumElementsToStore = 500;//Initialize replay memory capacity.
    public double epsilon = 0.2;
    public int C = 20;//after c rounds, copy myNN weights to target nn's weights
    public int c = C; 
    public int hidNum = 10;
    public int batchSize = 100;
    public static double discount = 0.9;
    
    //int ROUND_NUM = 1000;
    
    public int actionNum = 4;
    NeuralNet myNN = new NeuralNet(5,hidNum,actionNum);
    NeuralNet targetNN = new NeuralNet(5,hidNum,actionNum);
    //LearningProcess qAgent = new LearningProcess();
    public boolean intermediateReward = true;
    public double myReward = 0.0;
    public double[] myRewardList = new double[maxNumElementsToStore];
    public double positiveReward = 1;
    public double negativeReward = 1;
    public double winBonus = 10;
    public double lostBonus = 5;
    double heading=0.0;
    double targetBearing=0.0;
    double targetDis=0.0;
    double xPos=0.0;
    double yPos=0.0;
    double[][] batchExperience;
    int randomPickBatch;
    double RMSError = 0.0;
    boolean ifTrain = true;
    private static final int divNum = 100;
    private static int[] mNumWinArray = new int[divNum*100];
    public double[] yLabel;
    public int roundNum = 10000;
    public double[] qError=new double[roundNum];
    public int errIndex = 0;
    //double[] RMSRecord = new double[ROUND_NUM];
    //define a network for training, and in the constructor of NeuralNet, I change it to call initialize
    //after constructed.
    CircularStringQueue experience = new CircularStringQueue(maxNumElementsToStore);
    public void run() {
        //each episode starts in run function
//    	out.println("C is now: " + c);
//        if(getRoundNum()>2000) {
//        	epsilon = 0.2;
//        }
//        
//        if(c==0){
//        	
//        	updateTarget();
//        	c = C;
//        	out.println("Update target weights!!! C is now: "+c);
//        }else {
//        	c-=1;
//        }
        
    	if(ifTrain) {
    		try {
				myNN.loadWeightsItoH(getDataFile("weightsInToHid.dat"));
				targetNN.loadWeightsItoH(getDataFile("weightsInToHid.dat"));
				myNN.loadWeightsHtoO(getDataFile("weightsHidToOut.dat"));
				targetNN.loadWeightsHtoO(getDataFile("weightsHidToOut.dat"));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				//e.printStackTrace();
			}
    	}
    	out.println("I am in run()");
        initializeState();
        //updateTarget();
        setColors(new Color(255, 233, 51), new Color(0,0,0), new Color(255, 255, 255));
        setBulletColor(new Color(63, 217, 10 ));
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        execute();
        turnRadarLeft(360);
        while (true){
        	//out.println("In While loop !!!");
        	chooseAction();
        	callTrain();
//        	out.println("In While loop !!!");
//        	//out.println("In choosing Action Part!!!");
//        	double[] state = getMyState();
//            state = dataProcessing(state);
//            //out.println(state != null);
//            //out.println("state: "+ state[0]+ state[1]+ state[2]+ state[3]+ state[4]);
//            //experience.add(state);
//            //int myAction = epsilonGreedy(epsilon,state);
//            //out.println("This is experience: " + Arrays.toString(experience.get(10)));
//            //out.println("myAction is: "+myAction);
//            int myAction = (int)(Math.random() * 4);
//            out.println("In choosing Action Part!!! Action: "+myAction);
//            switch (myAction) {
//                case 0:
//                    // int plusOrMinus = Math.random() < 0.5 ? -1 : 1;
//                    //out.println("Ahead!!!");
//                    setAhead(10);
//                    break;
//                case 1:
//                    //out.println("Back!!!");
//                    setBack(10);
//                    break;
//                case 2:
//                    //out.println("Turn!!!");
//                    turnLeft(30);
//                    setAhead(10);
//                    break;
////    			case Action.turnRight:
////    				setAhead(Action.aheadDis);
////    				turnRight(Action.turnDegree);
////    				break;
//                case 3:
//                    //out.println("Fire!!!");
//                    // turnGunLeft(getGunHeading() - getHeading() - targetBearing);
//                    setFire(5.0/(targetDis/100));
//                    // turnLeft(30);
//                    // setAhead(50);
//                    break;
//                default:
//                    fire(1);
//                    break;
//            }
//            double[] statePrime = getMyState();
//            statePrime = dataProcessing(statePrime);
//            Object[] e = new Object[4];
//            e[0] = state;
//            e[1] = myAction;
//            e[2] = myReward;
//            e[3] = statePrime;
//            experience.add(e);
//            out.println("In adding experince part!!!");
//            myReward = 0;
//            turnRadarLeft(360);
//            execute();
//            
//            for(int i=0; i<batchSize; i++){
//            	//out.println("In for loop of while loop");
//                randomPickBatch = (int)(Math.random()*maxNumElementsToStore);
//                //out.println(experience.get(randomPickBatch) instanceof double[]);
//                //batchExperience[i] = experience.get(randomPickBatch);
//                //out.println("This is state: " + Arrays.toString(experience.get(randomPickBatch)));
//                //out.println((experience.get(0)) instanceof double[]);
//                turnRadarLeft(360);                
////                train(experience.get(randomPickBatch)[0],experience.get(randomPickBatch)[2],experience.get(randomPickBatch)[3]);
//                train(experience.get(randomPickBatch));
//            }
//            out.println("After training Part!!!");
//            //train(batchExperience);
            //out.println("C is now: " + c);
            c-=1;
            
            if(c==0){
            	//out.println("Update target weights!!!");
            	//saveQError();
            	updateTarget();
            	c = C;
            	
            }
//            double[] state = getMyState();
//            state = dataProcessing(state);            
//            experience.add(state);
//            int myAction = epsilonGreedy(epsilon,state);
//            out.println("myAction is: "+myAction);            
//            switch (myAction) {
//                case 0:                    
//                    setAhead(10);
//                    break;
//                case 1:
//                
//                    setBack(10);
//                    break;
//                case 2:
//                   
//                    turnLeft(30);
//                    setAhead(10);
//                    break;
//
//                case 3:
//                    
//                    setFire(20.0/(targetDis/100));
//                 
//                    break;
//                default:
//                    fire(1);
//                    break;
//            }
//            myReward = 0;
//            turnRadarLeft(360);
//            execute();
            //out.println(randomPickBatch);
//            out.println(experience.element()[0]);
//            Object p = experience.toArray();
            //out.println("This is state: " + Arrays.toString(experience.pop()));
//            experience.get(0);
            //out.println("check instance");
        	
        	
//            for(int i=0; i<batchSize; i++){
//            	out.println("In for loop of while loop");
//                randomPickBatch = (int)(Math.random()*maxNumElementsToStore);
//                
//                turnRadarLeft(360);                
//
//                train(experience.get(randomPickBatch));
//            }
//            
//            out.println("C is now: " + C);
//            c-=1;
//            
//            if(c==0){
//            	out.println("Update target weights!!!");
//            	updateTarget();
//            	c = C;
//            }
            
        }
    }
    
    public void chooseAction() {
    	//out.println("In choosing Action Part!!!");
    	double[] state = getMyState();
        state = dataProcessing(state);
        //out.println(state != null);
        //out.println("state: "+ state[0]+ state[1]+ state[2]+ state[3]+ state[4]);
        //experience.add(state);
        int myAction = epsilonGreedy(epsilon,state);
        //out.println("This is experience: " + Arrays.toString(experience.get(10)));
        //out.println("myAction is: "+myAction);
        //int myAction = (int)(Math.random() * 4);
        switch (myAction) {
            case 0:
                // int plusOrMinus = Math.random() < 0.5 ? -1 : 1;
                //out.println("Ahead!!!");
                setAhead(50*(targetDis/100.0));
                break;
            case 1:
                //out.println("Back!!!");
                setBack(20*(targetDis/100.0));
                break;
            case 2:
                //out.println("Turn!!!");
                turnLeft(30);
                setAhead(10);
                break;
//			case Action.turnRight:
//				setAhead(Action.aheadDis);
//				turnRight(Action.turnDegree);
//				break;
            case 3:
                //out.println("Fire!!!");
                //turnGunLeft(getGunHeading() - getHeading() - targetBearing);
                setFire(5.0/(targetDis/100));
                // turnLeft(30);
                // setAhead(50);
                break;
            default:
                fire(1);
                break;
        }
        double[] statePrime = getMyState();
        statePrime = dataProcessing(statePrime);
        Object[] e = new Object[4];
        e[0] = state;
        e[1] = myAction;
        e[2] = myReward;
        e[3] = statePrime;
        experience.add(e);
        
        myReward = 0;
        turnRadarLeft(360);
        execute();
        //call Train process
//        for(int i=0; i<batchSize; i++){
//        	//out.println("In for loop of while loop");
//            randomPickBatch = (int)(Math.random()*maxNumElementsToStore);
//            //out.println(experience.get(randomPickBatch) instanceof double[]);
//            //batchExperience[i] = experience.get(randomPickBatch);
//            //out.println("This is state: " + Arrays.toString(experience.get(randomPickBatch)));
//            //out.println((experience.get(0)) instanceof double[]);
//            turnRadarLeft(360);                
////            train(experience.get(randomPickBatch)[0],experience.get(randomPickBatch)[2],experience.get(randomPickBatch)[3]);
//            train(experience.get(randomPickBatch));
//        }
//        //train(batchExperience);
//        out.println("C is now: " + C);
//        c-=1;
//        
//        if(c==0){
//        	//out.println("Update target weights!!!");
//        	updateTarget();
//        	c = C;
//        }
	}
    
    
    public void callTrain() {
//    	double[] statePrime = getMyState();
//        statePrime = dataProcessing(statePrime);
//        double[] state = getMyState();
//        state = dataProcessing(state);
//        int myAction = epsilonGreedy(epsilon,state);
//        Object[] e = new Object[4];
//        e[0] = state;
//        e[1] = myAction;
//        e[2] = myReward;
//        e[3] = statePrime;
//        experience.add(e);
    	for(int i=0; i<maxNumElementsToStore; i++){
        	//out.println("In for loop of while loop");
            randomPickBatch = (int)(Math.random()*maxNumElementsToStore);
            //out.println(experience.get(randomPickBatch) instanceof double[]);
            //batchExperience[i] = experience.get(randomPickBatch);
            //out.println("This is state: " + Arrays.toString(experience.get(randomPickBatch)));
            //out.println((experience.get(0)) instanceof double[]);
            //turnRadarLeft(360);                
//            train(experience.get(randomPickBatch)[0],experience.get(randomPickBatch)[2],experience.get(randomPickBatch)[3]);
            train(experience.get(i));
        }
        //train(batchExperience);
//        out.println("C is now: " + c);
//        c-=1;
//        
//        if(c==0){
//        	//out.println("Update target weights!!!");
//        	//updateTarget();
//        	c = C;
//        }
	}
    
    @Override
    public void onScannedRobot(ScannedRobotEvent event) {
    	//out.println("I am Scanning at the enemy!");
        targetBearing = event.getBearing();
        targetDis = event.getDistance();
        turnGunLeft(getGunHeading() - getHeading() - targetBearing);
        //fire(1);
        //setFire(5.0/(targetDis/100));
        chooseAction();
    }

    public void onBulletHit(BulletHitEvent event) {
        // TODO Auto-generated method stub
        if (intermediateReward) {
            myReward += positiveReward * 5;
        }
        // out.println("I hit a robot! My energy: " + getEnergy() + " his energy: " +
        // event.getEnergy());
        if (event.getEnergy() < 0.001) return; //ignore if enemy dead

    }

    public void onHitRobot(HitRobotEvent event) {
        // out.println("Ram you, idiot!!!");
        if (intermediateReward) {
            myReward += positiveReward;
        }
    }

    public void onHitByBullet(HitByBulletEvent event) {
        // TODO Auto-generated method stub
        turnRight(60);
        setAhead(150);
        //out.println("Oh God! I got hit! My energy: " + getEnergy());
        if (intermediateReward) {
            myReward -= negativeReward / 2;
        }

    }

    public void onBulletMissed(BulletMissedEvent event) {
        // out.println("Oh shoot! The bullet missed! My energy: " + getEnergy());
        if (intermediateReward) {
            myReward -= negativeReward / 2;
        }
    }

    public void onHitWall(HitWallEvent event) {
        // out.println("Oh I hit the wall");
        setBack(100);
        turnLeft(120);
        if (intermediateReward) {
            myReward -= negativeReward / 2;
        }
    }

    public void onWin(WinEvent event) {
        myReward += winBonus;
        callTrain();
        saveToFile();
        getRMS(RMSError);
        mNumWinArray[(getRoundNum() - 1) / divNum]++;
        
    }

    public void onDeath(DeathEvent event) {
    	callTrain();
    	getRMS(RMSError);
        myReward -= lostBonus;
        saveToFile();
        
        //out.println("This is experience: " + Arrays.toString(experience.get(10)));
    }

    public void onBattleEnded(BattleEndedEvent event) {
    	//for (int i = 0; i < maxNumElementsToStore; i++) {
        	//out.println("This is experience: " + Arrays.toString(experience.get(i)));
		//}
//    	myNN.saveWeights(0);
//        myNN.saveWeights(1);
    	saveWinRateToConsole();
    	saveToFile();
    	saveQError();
        //showQError();
    }

    public void initializeState(){
        for(int i=0; i<maxNumElementsToStore; i++){
            double[] s = {0.0,0.0,0.0,0.0,0.0};
            int a = (int)(Math.random() * actionNum);
            double rp = 0.0;
            double[] sp = {0.0,0.0,0.0,0.0,0.0};
            Object[] e = new Object[4];
            e[0] = s;
            e[1] = a;
            e[2] = rp;
            e[3] = sp;
            experience.add(e);
            myRewardList[i]=0;
        }        
    }

    public double[] getMyState(){
        double[] state = {getHeading(), targetDis, targetBearing, getX(), getY()};
        return state;
    }

    public int getMyAction(double[] state){
        double[] nnRes = myNN.forwardFeed(state);
//        out.println("state: "+state[0]+state[1]+state[2]+state[3]+state[4]);
//        out.println("nnRes: "+nnRes[0]+nnRes[1]+nnRes[2]+nnRes[3]);
//        out.println("weights: "+myNN.weightsHiddenToOutput[1][1]+myNN.weightsHiddenToOutput[2][2]+myNN.weightsHiddenToOutput[0][0]);
        int actionRes = getMaxAction(nnRes);
        //out.println("actionRes:"+ actionRes);
        return actionRes;
    }

    public int epsilonGreedy(double epsilon, double[] state) {
        double randomVal = Math.random();
        if (randomVal <= epsilon) {
            //exploratory
            return (int)(Math.random() * myNN.argNumOutput);
        }else {
            return getMyAction(state);
        }
    }

    public int getMaxAction(double[] nnRes){//取得具有最大输出的节点号
        int maxNo = 0;
        double init = nnRes[0];
        for (int i=0; i<nnRes.length; i++){
            if (init>nnRes[i]){
                init = nnRes[i];
                maxNo = i;
            }
        }
        return maxNo;
    }

    public void updateTarget(){
        targetNN.weightsInputToHidden = myNN.weightsInputToHidden;
        targetNN.weightsHiddenToOutput = myNN.weightsHiddenToOutput;
    }

//    public void train(double[][] stateBatch){
//        double myError = 0.0;
//        int epoch = 0;
//        double RMSError = 0.0;
//        double[] qCurrent;
//        double[] qTarget;
//        do{
//            myError = 0.0;
//            for (int i = 0; i < batchSize; i++) {
//
//                qCurrent = myNN.forwardFeed(stateBatch[i]);
//                qTarget = targetNN.forwardFeed(experience.get(experience.indexOf(stateBatch[i])+1));
//                //myNN.backPropagation(qTarget[i],qCurrent[i],stateBatch[i]);
//                for (int j=0;j<myNN.argNumOutput;j++) {
//                    RMSError += Math.pow((qCurrent[j] - qTarget[j]), 2);
//                }
//            }
//            epoch += 1;
////        }while (myError>0.05);
//        }while (epoch<1000);
//    }
//public void train(double[] curState, int reward, double[] statePrime){
public void train(Object[] stateBatch){
    double myError = 0.0;
    
    
    double[] qCurrent;
    double[] qTarget;
    double[] qTargetAddReward = {0,0,0,0};
    int experienceIndex;
    int actionNo = 0;
    //do{
        myError = 0.0;
        //for (int i = 0; i < batchSize; i++) {
            //experienceIndex = experience.indexOf(stateBatch);
            qCurrent = myNN.forwardFeed((double[])(stateBatch[0]));
            //out.println("Show the qcurrent: "+qCurrent);
            actionNo = getMaxAction(qCurrent);
//            if(experienceIndex<maxNumElementsToStore - 1){
//                qTarget = targetNN.forwardFeed((double[])(stateBatch[3]));
//            }else{
//                qTarget = targetNN.forwardFeed((double[])(stateBatch[3]));
//            }
            qTarget = targetNN.forwardFeed((double[])(stateBatch[3]));
//            for(int i =0; i<5;i++) {
//            	out.println("state is " + ((double[])(stateBatch[3]))[i]);
//            }
            for (int j = 0; j < qTarget.length; j++) {
//				if (j==actionNo) {
//					qTargetAddReward[j] = discount * qTarget[j] +(double)(stateBatch[2]);
//				}else {
////					qTargetAddReward[j] = qTarget[j];
//					qTargetAddReward[j] = 0;
//				}
            	qTargetAddReward[j] = discount * qTarget[j] +(double)(stateBatch[2]);
			}
            myNN.backPropagation(qTargetAddReward,qCurrent,(double[])(stateBatch[0]));
//            for (int j=0;j<myNN.argNumOutput;j++) {
//                RMSError += Math.pow((qCurrent[j] - qTarget[j]), 2);
//                getRMS(RMSError);
//                //out.println("RMS is :"+qCurrent[j]);
//            }
            //getRMS(RMSError);
        //}
            //saveQError();
//        }while (myError>0.05);
    //}while (epoch<1000);
}
	public static int getDRHeading(double x) {
		int result = 0;
		if(x >= 0&& x<(Math.PI/2)) {
			result = 0;
		}else if (x >= (Math.PI/2) && x<(Math.PI)) {
			result = 1;
		}else if (x >= (Math.PI) && x<(Math.PI*3/2)) {
			result = 2;
		}else if (x >= (Math.PI*3/2) && x<(Math.PI*2)) {
			result = 3;
		}
		return result;
	}
	
	public static int getTargetDistance(double x) {
		int result = 0;
		int distance = 0;
		int gunPower = 1;
		distance = (int)(x/100);
		switch (distance) {
			case 0:
				result = 0;//really close to the target, can ram it
				gunPower = 10;
				break;
			case 1:
				result = 1;//close
				gunPower = 5;
				break;
			case 2:
				result = 2;//far
				gunPower = 2;
				break;
			default:
				result = 3;
				gunPower = 1;
				break;
		}
		
		return result;
	}
	public static int getTargetBearing(double x) {
		int result = 0;
		x = x + Math.PI;
		if(x >= 0&& x<(Math.PI/2)) {
			result = 0;
		}else if (x >= (Math.PI/2) && x<(Math.PI)) {
			result = 1;
		}else if (x >= (Math.PI) && x<(Math.PI*3/2)) {
			result = 2;		
		}else {
			result = 3;
		}
		return result;
	}
	public static int getXPos(double x) {
		int result = 0;
		int distance = 0;
		distance = (int)(x/100);
		if(distance>=0 && distance<2) {
			result = 0;
		}else if (distance>=2 && distance<4) {
			result = 1;
		}else{
			result = 2;
		}
		return result;
	}
	public static int getYPos(double x) {
		int result = 0;
		int distance = 0;
		distance = (int)(x/100);
		if(distance>=0 && distance<2) {
			result = 0;
		}else if (distance>=2 && distance<4) {
			result = 1;
		}else {
			result = 2;
		}
		return result;
	}
    public double[] dataProcessing(double[] state){
        double norHeading = state[0] / 100 - 1.8;
        double norTDis = state[1] / 100 - 5;
        double norTBearing = (state[2]+180) / 100 - 1.8;
        double norX = state[3] / 100 - 4;
        double norY = state[4] / 100 - 3;
        double[] res = {norHeading,norTDis, norTBearing, norX, norY};
    	//double[] res = {getDRHeading(state[0]),getTargetDistance(state[1]), getTargetBearing(state[2]), getXPos(state[3]), getYPos(state[4])};
        return res;
    }
    
    public void saveToFile() {
    	PrintStream w = null;
		try {
			w = new PrintStream(
					new RobocodeFileOutputStream(getDataFile("wItoH.dat").getAbsolutePath(), true));
//			DecimalFormat dFormat = new DecimalFormat("#.##");	
			for (int i = 0; i < myNN.argNumInputs+1; i++) {
				for (int j = 0; j < myNN.argNumHidden; j++) {
					w.println(myNN.weightsInputToHidden[i][j]);
				}
			}			
			if (w.checkError())
				System.out.println("Could not save the data!");
			w.close();
			
			w = new PrintStream(
					new RobocodeFileOutputStream(getDataFile("wHtoO.dat").getAbsolutePath(), true));
	
			for (int i = 0; i < myNN.argNumHidden+1; i++) {
				for (int j = 0; j < myNN.argNumOutput; j++) {
					w.println(myNN.weightsHiddenToOutput[i][j]);
				}
			}			
			if (w.checkError())
				System.out.println("Could not save the data!");
			w.close();
			
			
		} catch (IOException e) {
			System.out.println("IOException trying to write: " + e);
		} finally {
			try {
				if (w != null)
					w.close();
			} catch (Exception e) {
				System.out.println("Exception trying to close witer: " + e);
			}
		}
	}
    
    public void getRMS(double rmsError) {
    	PrintStream w = null;
		try {
			out.println("Writing error!!!" + rmsError);
			w = new PrintStream(
					new RobocodeFileOutputStream(getDataFile("RMS.dat").getAbsolutePath(), true));
//			DecimalFormat dFormat = new DecimalFormat("#.##");	
			w.println(rmsError);		
			if (w.checkError())
				System.out.println("Could not save the data!");
			w.close();			
		} catch (IOException e) {
			System.out.println("IOException trying to write: " + e);
		} finally {
			try {
				if (w != null)
					w.close();
			} catch (Exception e) {
				System.out.println("Exception trying to close witer: " + e);
			}
		}		
	}
    public void saveWinRateToConsole() {		
		out.println("saveingWinRate");
    	for (int i = 0; i < getRoundNum()/100; i++) {			
			out.println("Winning rate: "+mNumWinArray[i]/100.0);
		}		
	}
    public void saveQError() {
    	double[] s = {0.1,1.3,0.25,0.98,2.82};
    	double errorValue = 0;
    	//for(int i=0;i<myNN.argNumOutput;i++) {
    		//errorValue +=Math.pow((myNN.forwardFeed(s)[i]-targetNN.forwardFeed(s)[i]), 2);
    		errorValue +=myNN.forwardFeed(s)[0]-targetNN.forwardFeed(s)[0];
    	//}
//    	if(Math.pow(errorValue, 2)>0.0) {    		
//    		qError[errIndex] = errorValue;
//    		errIndex++;
//    	}
		//qError[errIndex] = errorValue;
		//errIndex++;
    	getRMS(errorValue);
    }
    public void showQError() {
    	for(int i =0; i<1000; i++) {
    		//if(qError[i]!=0.0) {
    			//out.println("Qerror is: "+qError[i]);
    		//}   
    		getRMS(qError[i]);
    	}
    }
//    public static double[][] getYlabel() throws IOException {
//        double[][] yLabel = new double[trainSize][4];
//        BufferedReader in = new BufferedReader(new FileReader("roboLUT_testAuto2.dat"));  //
//        String line;  //一行数据
//        int row=0;
//        double temp = 0;
//        int i = 0;
//        int j = 0;
//        //逐行读取，并将每个数组放入到数组中
//        while((line = in.readLine()) != null){
////            for (int i=0;i<trainSize;i++){
////                for(int j=0;j<4;j++){
////                    temp = Double.parseDouble(line);
////                    yLabel[i][j] = temp;
////                }
////            }
//            temp = Double.parseDouble(line);
//            yLabel[i][j]=temp;
//            if(j<4-1){
//                j++;
//            }else{
//                j = 0;
//                i++;
//            }
//        }
//        in.close();
//        return yLabel;
//    }
}
