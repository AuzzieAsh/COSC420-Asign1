/*
	COSC420 Neural Networks Assignment 1
	Ashley Manson, 4061527
 */

import java.util.*;
import java.io.*;
import java.lang.Math.*;

public class NeuralNetwork {

    public static void main (String[] args) {
        
        // File IO stuff
        int numOfPatterns = 0;
        
        // param.txt values
        int numOfInput = 0;
        int numOfHidden = 0;
        int numOfOutput = 0;
        double learningConstant = 0.0; // alpha
        double momentumConstant = 0.0;
        double errorCriterion = 0.0;
        
        // in.txt values
        double[][] inputPatterns = new double[0][0];
        
        // teach.txt values
        double[][] teachingPatterns = new double[0][0]; // target output

        try {
            // Read data from param.txt
            Scanner scan = new Scanner(new File("param.txt"));
            numOfInput = scan.nextInt();
            numOfHidden = scan.nextInt();
            numOfOutput = scan.nextInt();
            learningConstant = scan.nextDouble();
            momentumConstant = scan.nextDouble();
            errorCriterion = scan.nextDouble();
            
            // Get the number of patterns
            scan = new Scanner(new File("in.txt"));
            while (scan.hasNextLine()) {
                scan.nextLine();
                numOfPatterns++;
            }
            
            // Read data from in.txt
            scan = new Scanner(new File("in.txt"));
            inputPatterns = new double[numOfPatterns][numOfInput];
            for (int i = 0; i < numOfPatterns; i++) {
                for (int j = 0; j < numOfInput; j++) {
                    inputPatterns[i][j] = scan.nextDouble();
                }
            }
            
            // Read data from teach.txt
            scan = new Scanner(new File("teach.txt"));
            teachingPatterns = new double[numOfPatterns][numOfOutput];
            for (int i = 0; i < numOfPatterns; i++) {
                for (int j = 0; j < numOfOutput; j++) {
                    teachingPatterns[i][j] = scan.nextDouble();
                }
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        
        int epochs = 0;
        double errorPop = 1.0;
        double errorPopSum = 0.0;
        
        double[][] hiddenPatterns = new double[numOfPatterns][numOfHidden];
        double[][] actualPatterns = new double[numOfPatterns][numOfOutput];
        
        double[][] hiddenBias = new double[numOfPatterns][numOfHidden];
        double[][] actualBias = new double[numOfPatterns][numOfOutput];
        
        double[][] inputWeights = new double[numOfPatterns][numOfHidden];
        double[][] hiddenWeights = new double[numOfPatterns][numOfOutput];
        
        double[][] inputError = new double[numOfPatterns][numOfInput];
        double[][] hiddenError = new double[numOfPatterns][numOfHidden];
        
        // Setup the initial weights and bias
        Random rnjesus = new Random();
        for (int i = 0; i < numOfPatterns; i++) {
            for (int j = 0; j < numOfHidden; j++) {
                inputWeights[i][j] = rnjesus.nextDouble() * 0.6 - 0.3;
                hiddenBias[i][j] = rnjesus.nextDouble() * 0.6 - 0.3;
            }
        }
        for (int i = 0; i < numOfPatterns; i++) {
            for (int j = 0; j < numOfOutput; j++) {
                hiddenWeights[i][j] = rnjesus.nextDouble() * 0.6 - 0.3;
                actualBias[i][j] = rnjesus.nextDouble() * 0.6 - 0.3;
            }
        }
        
        while (epochs < 10000) {
            // Compute the activiation for the hidden layer, and the output for the actual layer
            for (int i = 0; i < numOfPatterns; i++) {
                for (int j = 0; j < numOfHidden; j++) {
                    hiddenPatterns[i][j] = activationFunction(inputPatterns[i], inputWeights[i], hiddenBias[i][j]);
                }
                for (int j = 0; j < numOfOutput; j++) {
                    actualPatterns[i][j] = activationFunction(hiddenPatterns[i], hiddenWeights[i], actualBias[i][j]);
                }
            }
            
            for (int i = 0; i < numOfPatterns; i++) {
                // Compute the error of the deriative for the output layer
                for (int j = 0; j < numOfOutput; j++) {
                    for (int k = 0; k < numOfInput; k++) {
                        double errorValue = (teachingPatterns[i][j] - actualPatterns[i][j]) * actualPatterns[i][j] * (1 - actualPatterns[i][j]);
                        inputError[i][k] = errorValue;
                    }
                }

                // Compute the error for the hidden layer
                for (int j = 0; j < numOfHidden; j++) {
                    double errorWeightSum = 0.0;
                    for (int k = 0; k < numOfOutput; k++) {
                        errorWeightSum += inputError[i][k] * hiddenWeights[i][k];
                    }
                    hiddenError[i][j] = hiddenPatterns[i][j] * (1 - hiddenPatterns[i][j]) * errorWeightSum;
                }
            }
            
            // Make the Weight changes
            for (int i = 0; i < numOfPatterns; i++) {
                for (int j = 0; j < numOfOutput; j++) {
                    hiddenWeights[i][j] = (learningConstant * hiddenError[i][j] * hiddenPatterns[i][j]) + (momentumConstant * hiddenWeights[i][j]);
                    actualBias[i][j] += (learningConstant * hiddenError[i][j] * 1);
                }
                
                for (int j = 0; j < numOfInput; j++) {
                    inputWeights[i][j] = (learningConstant * inputError[i][j] * inputPatterns[i][j]) + (momentumConstant * inputWeights[i][j]);
                    hiddenBias[i][j] += (learningConstant * inputError[i][j] * 1);
                }
            }
            
            for (int i = 0; i < numOfPatterns; i++) {
                double patternError = 0.0;
                for (int j = 0; j < numOfOutput; j++) {
                    patternError += Math.pow(teachingPatterns[i][j] - actualPatterns[i][j], 2);
                }
                errorPopSum += patternError;
            }
            
            errorPop = (double)(errorPopSum / (numOfOutput * numOfPatterns));
            errorPopSum = 0.0;
            
            epochs++;
            if (errorPop < errorCriterion) break;
        }

        
        System.out.println("Error Pop: " + errorPop);
        printDouble2DArray(hiddenPatterns);
        System.out.println();
        printDouble2DArray(actualPatterns);
        System.out.println();
        printDouble2DArray(hiddenBias);
        System.out.println();
        printDouble2DArray(actualBias);
        
        System.out.println("\nEpochs: " + epochs);
        
    }
    
    public static double activationFunction(double[] inputLayer, double[] weights, double bias) {
        
        double sum = 0.0;
        for (int i = 0; i < inputLayer.length; i++) {
            for (int j = 0; j < weights.length; j++) {
                sum += (inputLayer[i] * weights[j]);
            }
        }
        sum += bias;
        return (double)(1/(1 + Math.pow(Math.E, -sum)));
    }
    
    /* Print Functions */
    public static void printDouble2DArray(double[][] toPrint) {
        for (int i = 0; i < toPrint.length; i++) {
            for (int j = 0; j < toPrint[i].length; j++) {
                System.out.print(toPrint[i][j] + " ");
            }
            System.out.println("");
        }
    }
    
}