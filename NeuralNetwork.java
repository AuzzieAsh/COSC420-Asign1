/*
	File: NeuralNetwork.java
	Author: Ashley Manson
	Description: This is the majority of the program. It will create a new
	Neural Network and then proceed to go through a number of different epochs
	to solve a given problem. Three files are required for the program to run,
	param.txt, in.txt, and teach.txt. The final result is printed out when the
	population error is less than user specified error criterion.
	Developed with Java Version: 1.8.0_45
*/

import java.util.*;
import java.io.*;
import java.lang.Math.*;

public class NeuralNetwork {

    public void run(boolean show_epochs, boolean show_weights, 
					boolean show_activation) {
        
		// row and col used many times
        int row, col;
        int epochs = 0;
        double error_pop = 0.0;
        
		// param.txt variables
        int num_of_patterns = 0;
        int num_of_input = 0;
        int num_of_hidden = 0;
        int num_of_output = 0;
        double learning_constant = 0.0;
        double momentum_constant = 0.0;
        double error_criterion = 0.0;
        
		// in.txt variables
        NeuralNode[][] input_layers = new NeuralNode[0][0];
		// teach.txt variables
        double[][] teaching_patterns = new double[0][0];
		
        NeuralNode[][] hidden_layers = new NeuralNode[0][0];
        NeuralNode[][] output_layers = new NeuralNode[0][0];
        
        try {
			// Read in the variables from param.txt
            Scanner param = new Scanner(new File("param.txt"));
            num_of_input = param.nextInt();
            num_of_hidden = param.nextInt();
            num_of_output = param.nextInt();
            learning_constant = param.nextDouble();
            momentum_constant = param.nextDouble();
            error_criterion = param.nextDouble();
            
			// Get the number of patterns
            Scanner line_read = new Scanner(new File("in.txt"));
            while (line_read.hasNextLine()) {
                line_read.nextLine();
                num_of_patterns++;
            }
            
			// Initialise the neural network
            input_layers = new NeuralNode[num_of_patterns][num_of_input];
            hidden_layers = new NeuralNode[num_of_patterns][num_of_hidden];
            output_layers = new NeuralNode[num_of_patterns][num_of_output];
            teaching_patterns = new double[num_of_patterns][num_of_output];
            
			for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_input; col++) {
                    input_layers[row][col] = new NeuralNode(num_of_hidden);
                }
                for (col = 0; col < num_of_hidden; col++) {
                    hidden_layers[row][col] = new NeuralNode(num_of_output);
                }
                for (col = 0; col < num_of_output; col++) {
                    output_layers[row][col] = new NeuralNode(0);
                }
            }
			
			// Proceed to read in the values from in.txt and teach.txt
            Scanner in = new Scanner(new File("in.txt"));
            Scanner teach = new Scanner(new File("teach.txt"));
            Random rng = new Random();
			
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_input; col++) {
                    input_layers[row][col].set_pattern(in.nextDouble());
                }
                for (col = 0; col < num_of_output; col++) {
                    teaching_patterns[row][col] = teach.nextDouble();
                }
            }
        }
        catch(IOException e) {
            System.err.println("IO Error: Ensure all the texts files are there" 
				+ " and are correct.");
			System.exit(-1);
        }
		
		// Run infinitely
        for (;;) {
            epochs++;
			
			if (show_epochs && epochs % 100 == 0) {
				System.out.printf("Epochs: %d\nPopulation Error: %.6f\n", 
					epochs, error_pop);
			}
			
			// Set all the activations for the hidden layer
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_hidden; col++) {
                    hidden_layers[row][col].set_pattern(activation_function(
						input_layers[row], col, hidden_layers[row][col].bias));
                }
            }
			// Set all the activations for the output layer
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_output; col++) {
                    output_layers[row][col].set_pattern(activation_function(
						hidden_layers[row], col, output_layers[row][col].bias));
                }
            }
            
			// Find and set the error for the output layer
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_output; col++) {
                    double pattern_value = output_layers[row][col].pattern;
                    double error_value = (teaching_patterns[row][col] -
						pattern_value) * pattern_value * (1 - pattern_value);
						output_layers[row][col].set_error(error_value);
                }
            }
            
			// Find and set the error for the hidden layer
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_hidden; col++) {
                    double pattern_value = hidden_layers[row][col].pattern;
                    double error_x_weight_sum = 0.0;
                    for (int err = 0; err < num_of_output; err++) {
                        error_x_weight_sum += output_layers[row][err].error *
							hidden_layers[row][col].weight(err);
                    }
                    double error_value = pattern_value * (1 - pattern_value) *
						error_x_weight_sum;
						hidden_layers[row][col].set_error(error_value);
                }
            }
            
			// Weight and Bias adjustment for the output layer
            for (row = 0; row < num_of_patterns; row++) {
				// For each output node in a row, set its bias
				for (col = 0; col < num_of_output; col++) {
                    output_layers[row][col].set_bias(
						output_layers[row][col].bias += (learning_constant * 
						output_layers[row][col].error * 1));
						
					// For each hidden node: "prev" = before each output node
					for (int prev = 0; prev < num_of_hidden; prev++) {
						// For each weight for a hidden node, set its weight: 
						// "next" = all the output nodes connected to a hidden 
						// node	
						for (int next = 0; next < num_of_output; next++) {
							// Get the weight change
							double weight_change = (learning_constant * 
								output_layers[row][next].error * 
								hidden_layers[row][prev].pattern) + 
								(momentum_constant * 
								hidden_layers[row][prev].change(next));
							
							hidden_layers[row][prev].set_weight(next, 
								hidden_layers[row][prev].weight(next) + 
								weight_change);
								
							hidden_layers[row][prev].set_change(next, 
								weight_change);
						}
					}
                }
            }
			// Weight and Bias adjustment for the hidden layer
			for (row = 0; row < num_of_patterns; row++) {
                // For each hidden node in a row, set its bias
				for (col = 0; col < num_of_hidden; col++) {
                    hidden_layers[row][col].set_bias(
						hidden_layers[row][col].bias += (learning_constant *
						hidden_layers[row][col].error * 1));
						
					// For each input node: "prev" = before each hidden node
					for (int prev = 0; prev < num_of_input; prev++) {
						// For each weight for an input node, set its weight:
						// "next" = all the hidden nodes connected to an input 
						// node
						for (int next = 0; next < num_of_hidden; next++) {
							// Get the weight change
							double weight_change = (learning_constant *
								hidden_layers[row][next].error * 
								input_layers[row][prev].pattern) + 
								(momentum_constant * 
								input_layers[row][prev].change(next));
								
							input_layers[row][prev].set_weight(next, 
								input_layers[row][prev].weight(next) + 
								weight_change);
								
							input_layers[row][prev].set_change(next, 
								weight_change);
						}
					}
                }
            }
			
			// Calculate the population error
            double error_pop_sum = 0.0;
            for (row = 0; row < num_of_patterns; row++) {
                double pattern_error = 0.0;
                for (col = 0; col < num_of_output; col++) {
                    pattern_error += Math.pow(teaching_patterns[row][col] - 
						output_layers[row][col].pattern, 2);
                }
                error_pop_sum += pattern_error;
            }
            error_pop = error_pop_sum / (num_of_output * num_of_patterns);
			
			// Check to see if the population error is less then the error 
			// criterion
            if (error_pop < error_criterion) {
				if (show_epochs) System.out.println();
				break;
			}
        }
		
		// Successfully learned, print the results
		if (show_weights) {
			System.out.printf("Hidden Weights\n");
			for (row = 0; row < num_of_patterns; row++) {
				for (col = 0; col < num_of_input; col++) {
					for (int next = 0; next < num_of_hidden; next++) {
						System.out.printf("%.3f ", 
							input_layers[row][col].weight(next));
					}
					System.out.println();
				}
				System.out.println();
			}
			System.out.printf("Output Weights\n");
			for (row = 0; row < num_of_patterns; row++) {
				for (col = 0; col < num_of_hidden; col++) {
					for (int next = 0; next < num_of_output; next++) {
						System.out.printf("%.3f ", 
							hidden_layers[row][col].weight(next));
					}
					System.out.println();
				}
				System.out.println();
			}
		}
		
		if (show_activation) {
			System.out.printf("Input Activations\n");
			for (row = 0; row < num_of_patterns; row++) {
				for (col = 0; col < num_of_input; col++) {
					System.out.printf("%.1f ", input_layers[row][col].pattern);
				}
				System.out.println();
			}
			System.out.printf("\nHidden Activations\n");
			for (row = 0; row < num_of_patterns; row++) {
				for (col = 0; col < num_of_hidden; col++) {
					System.out.printf("%.3f ", hidden_layers[row][col].pattern);
				}
				System.out.println();
			}
			System.out.printf("\nOutput Activations\n");
			for (row = 0; row < num_of_patterns; row++) {
				for (col = 0; col < num_of_output; col++) {
					System.out.printf("%.3f ", output_layers[row][col].pattern);
				}
				System.out.println();
			}
			System.out.println();
		}
		
		System.out.printf("Final Result\nEpochs: %d\nPopulation Error: %.6f\n",
			epochs, error_pop);
    }
    
	/* 
		Activation function used to calculate the activation for a node
		Uses the previous input layer, an index to represent the node of the 
		layer the activation is being calculated for, and a bias.
	*/
    public double activation_function(NeuralNode[] input_layer, int index, 
									  double bias) {
        
        double sum = 0.0;
        for (int node = 0; node < input_layer.length; node++) {
            sum += (input_layer[node].pattern * 
				input_layer[node].weight(index));
        }
        sum += bias;
        return 1 / (1 + Math.pow(Math.E, -sum));
    }
}