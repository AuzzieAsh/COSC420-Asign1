
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
        
        double[][] input_patterns = new double[0][0];
        double[][] hidden_patterns = new double[0][0];
        double[][] teaching_patterns = new double[0][0];
        double[][] output_patterns = new double[0][0];
        
        double[][] input_weights = new double[0][0];
        double[][] hidden_weights = new double[0][0];
        
        double[][] input_error = new double[0][0];
        double[][] hidden_error = new double[0][0];
        
        double[][] input_changes = new double[0][0];
        double[][] hidden_changes = new double[0][0];
        double[][] input_prev = new double[0][0];
        double[][] hidden_prev = new double[0][0];
        
        double[][] hidden_bias = new double[0][0];
        double[][] output_bias = new double[0][0];
        
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
            
            input_patterns = new double[num_of_patterns][num_of_input];
            hidden_patterns = new double[num_of_patterns][num_of_hidden];
            teaching_patterns = new double[num_of_patterns][num_of_output];
            output_patterns = new double[num_of_patterns][num_of_output];
            
            input_weights = new double[num_of_patterns][num_of_hidden];
            hidden_weights = new double[num_of_patterns][num_of_output];
            
            input_error = new double[num_of_patterns][num_of_hidden];
            hidden_error = new double[num_of_patterns][num_of_output];
            
            input_changes = new double[num_of_patterns][num_of_hidden];
            hidden_changes = new double[num_of_patterns][num_of_output];
            input_prev = new double[num_of_patterns][num_of_hidden];
            hidden_prev = new double[num_of_patterns][num_of_output];
            
            hidden_bias = new double[num_of_patterns][num_of_hidden];
            output_bias = new double[num_of_patterns][num_of_output];
            
            Scanner in = new Scanner(new File("in.txt"));
            Scanner teach = new Scanner(new File("teach.txt"));
            Random rng = new Random();
            
            for (row = 0; row < num_of_patterns; row++) {
                
                for (col = 0; col < num_of_input; col++) {
                    input_patterns[row][col] = in.nextDouble();
                }
                
                for (col = 0; col < num_of_hidden; col++) {
                    hidden_patterns[row][col] = 0.0;
                    input_weights[row][col] = rng.nextDouble() * 0.6 - 0.3;
                    input_error[row][col] = 0.0;
                    hidden_bias[row][col] = rng.nextDouble() * 0.6 - 0.3;
                }
                
                for (col = 0; col < num_of_output; col++) {
                    teaching_patterns[row][col] = teach.nextDouble();
                    output_patterns[row][col] = 0.0;
                    hidden_weights[row][col] = rng.nextDouble() * 0.6 - 0.3;
                    hidden_error[row][col] = 0.0;
                    output_bias[row][col] = rng.nextDouble() * 0.6 - 0.3;
                }
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        
        for (;;) {
        //while (epochs < 1000000) {
            epochs++;
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_hidden; col++) {
                    hidden_patterns[row][col] = activation_function(input_patterns[row], input_weights[row], hidden_bias[row][col]);
                }
                for (col = 0; col < num_of_output; col++) {
                    output_patterns[row][col] = activation_function(hidden_patterns[row], hidden_weights[row], output_bias[row][col]);
                }
            }
            
            for (row = 0; row < num_of_patterns; row++) {
                for (col = 0; col < num_of_output; col++) {
                    double error_value = (teaching_patterns[row][col] - output_patterns[row][col]) * output_patterns[row][col] * (1 - output_patterns[row][col]);
                    hidden_error[row][col] = error_value;
                    hidden_changes[row][col] = (learning_constant * hidden_error[row][col] * hidden_patterns[row][col])* (momentum_constant * hidden_prev[row][col]);
                }
                for (col = 0; col < num_of_hidden; col++) {
                    double error_x_weight_sum = 0.0;
                    for (int err = 0; err < num_of_output; err++) {
                        error_x_weight_sum += hidden_error[row][err] * hidden_weights[row][err];
                    }
                    double error_value = hidden_patterns[row][col] * (1 - hidden_patterns[row][col]) * error_x_weight_sum;
                    input_error[row][col] = error_value;
                    input_changes[row][col] = (learning_constant * input_error[row][col] * input_patterns[row][col]) + (momentum_constant * input_prev[row][col]);
                }
            }
            
            for (row = 0; row < num_of_output; row++) {
                for (col = 0; col < num_of_output; col++) {
                    hidden_weights[row][col] += hidden_changes[row][col];
                    hidden_prev[row][col] = hidden_changes[row][col];
                    output_bias[row][col] += (learning_constant * hidden_error[row][col] * 1);
                }
                for (col = 0; col < num_of_hidden; col++) {
                    input_weights[row][col] += input_changes[row][col];
                    input_prev[row][col] = input_changes[row][col];
                    hidden_bias[row][col] += (learning_constant * input_error[row][col] * 1);
                }
            }
            
            double error_pop_sum = 0.0;
            for (row = 0; row < num_of_patterns; row++) {
                double pattern_error = 0.0;
                for (col = 0; col < num_of_output; col++) {
                    pattern_error += Math.pow(teaching_patterns[row][col] - output_patterns[row][col], 2);
                }
                error_pop_sum += pattern_error;
            }
            
            error_pop = (double)(error_pop_sum / (num_of_output * num_of_patterns));
            if (error_pop < error_criterion) break;
            System.out.println(error_pop);
        }
        System.out.println("Error Pop: " + error_pop);
        System.out.println("Epochs: " + epochs);
        /*
        System.out.println("\nInput Weights");
        print_double_2D_array(input_weights);
        System.out.println("\nInput Weights");
        print_double_2D_array(hidden_weights);
        System.out.println("\nHidden Bias");
        print_double_2D_array(hidden_bias);
        System.out.println("\nOutput Bias");
        print_double_2D_array(output_bias);*/
    }
    
    public static double activation_function(double[] input_layer, double[] weights, double bias) {
        
        double sum = 0.0;
        for (int node = 0; node < input_layer.length; node++) {
            for (int weight = 0; weight < weights.length; weight++) {
                sum += (input_layer[node] * weights[weight]);
            }
        }
        sum += bias;
        return (double)(1/(1 + Math.pow(Math.E, -sum)));
    }
    
    public static void print_double_2D_array(double[][] to_print) {
        for (int i = 0; i < to_print.length; i++) {
            for (int j = 0; j < to_print[i].length; j++) {
                System.out.print(to_print[i][j] + " ");
            }
            System.out.println();
        }
    }
}