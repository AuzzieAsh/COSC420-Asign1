
import java.util.*;
import java.io.*;
import java.lang.Math.*;

public class NewNetwork {

    public static void main(String[] args) {
        
        int row, col;
        int epochs = 0;
        double error_pop = 0.0;
        
        int num_of_patterns = 0;
        int num_of_input = 0;
        int num_of_hidden = 0;
        int num_of_output = 0;
        double learning_constant = 0.0;
        double momentum_constant = 0.0;
        double error_criterion = 0.0;
        
        NeuralNode[][] input_layers = new NeuralNode[0][0];
        NeuralNode[][] hidden_layers = new NeuralNode[0][0];
        NeuralNode[][] output_layers = new NeuralNode[0][0];
        double[][] teaching_patterns = new double[0][0];
        
        try {
            Scanner param = new Scanner(new File("param.txt"));
            num_of_input = param.nextInt();
            num_of_hidden = param.nextInt();
            num_of_output = param.nextInt();
            learning_constant = param.nextDouble();
            momentum_constant = param.nextDouble();
            error_criterion = param.nextDouble();
            
            Scanner line_read = new Scanner(new File("in.txt"));
            while (line_read.hasNextLine()) {
                line_read.nextLine();
                num_of_patterns++;
            }
            
            input_layers = new NeuralNode[num_of_patterns][num_of_input];
            hidden_layers = new NeuralNode[num_of_patterns][num_of_hidden];
            output_layers = new NeuralNode[num_of_patterns][num_of_output];
            teaching_patterns = new double[num_of_patterns][num_of_output];
            
            Scanner in = new Scanner(new File("in.txt"));
            Scanner teach = new Scanner(new File("teach.txt"));
            Random rng = new Random();
            
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
            
            for (row = 0; row < num_of_patterns; row++) {
                
                for (col = 0; col < num_of_input; col++) {
                    input_layers[row][col].set_pattern(in.nextDouble());
                }
                for (col = 0; col < num_of_output; col++) {
                    teaching_patterns[row][col] = teach.nextDouble();
                }
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        for (;;) {
        //while (epochs < 10000) {
            epochs++;
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_hidden; col++) {
                    hidden_layers[row][col].set_pattern(
                                            activation_function(input_layers[row], col, hidden_layers[row][col].bias));
                }
            }
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_output; col++) {
                    output_layers[row][col].set_pattern(
                                            activation_function(hidden_layers[row], col, output_layers[row][col].bias));
                }
            }
            
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_output; col++) {
                    double pattern_value = output_layers[row][col].pattern;
                    double error_value = (teaching_patterns[row][col] - pattern_value) * pattern_value * (1 - pattern_value);
                    output_layers[row][col].set_error(error_value);
                }
            }
            
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_hidden; col++) {
                    double pattern_value = hidden_layers[row][col].pattern;
                    double error_x_weight_sum = 0.0;
                    for (int err = 0; err < num_of_output; err++) {
                        error_x_weight_sum += output_layers[row][err].error * hidden_layers[row][col].weight(err);
                    }
                    double error_value = pattern_value * (1 - pattern_value) * error_x_weight_sum;
                    hidden_layers[row][col].set_error(error_value);
                }
            }
            
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_output; col++) {
                    output_layers[row][col].set_bias(output_layers[row][col].bias += (learning_constant * output_layers[row][col].error * 1));
                    for (int next = 0; next < num_of_output; next++) {
                        double weight_change = (learning_constant * output_layers[row][col].error * hidden_layers[row][col].pattern) + (momentum_constant * hidden_layers[row][col].change(next));
                        hidden_layers[row][col].set_weight(next, hidden_layers[row][col].weight(next) + weight_change);
                        hidden_layers[row][col].set_change(next, weight_change);
                    }
                }
            }
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_hidden; col++) {
                    hidden_layers[row][col].set_bias(hidden_layers[row][col].bias += (learning_constant * hidden_layers[row][col].error * 1));
                    for (int next = 0; next < num_of_hidden; next++) {
                        double weight_change = (learning_constant * hidden_layers[row][col].error * input_layers[row][col].pattern) + (momentum_constant * input_layers[row][col].change(next));
                        input_layers[row][col].set_weight(next, input_layers[row][col].weight(next) + weight_change);
                        input_layers[row][col].set_change(next, weight_change);
                    }
                }
            }
            
            double error_pop_sum = 0.0;
            for (row = 0; row < num_of_patterns; row++) {
                double pattern_error = 0.0;
                for (col = 0; col < num_of_output; col++) {
                    pattern_error += Math.pow(teaching_patterns[row][col] - output_layers[row][col].pattern, 2);
                }
                error_pop_sum += pattern_error;
            }
            
            error_pop = (double)(error_pop_sum / (num_of_output * num_of_patterns));
            if (error_pop < error_criterion) break;
        }
        System.out.println("Error Pop: " + error_pop);
        System.out.println("Epochs: " + epochs);
        print_output_teacher(output_layers, teaching_patterns);
    }
    
    public static double activation_function(NeuralNode[] input_layer, int index, double bias) {
        
        double sum = 0.0;
        for (int node = 0; node < input_layer.length; node++) {
            sum += (input_layer[node].pattern * input_layer[node].weight(index));
        }
        sum += bias;
        return (double)(1/(1 + Math.pow(Math.E, -sum)));
    }
    
    public static void print_output_teacher(NeuralNode[][] print_output, double[][] print_teacher) {
        for (int i = 0; i < print_output.length; i++) {
            for (int j = 0; j < print_output[i].length; j++) {
                System.out.print(print_output[i][j].pattern + " == " + print_teacher[i][j]);
            }
            System.out.println();
        }
    }
}