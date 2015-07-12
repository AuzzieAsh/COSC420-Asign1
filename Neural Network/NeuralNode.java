/*
	File: NeuralNode.java
	Author: Ashley Manson
	Description: This class represents a single Node for the NeuralNetwork 
	class.
	Developed with Java Version: 1.8.0_45
*/

import java.util.Random;

public class NeuralNode {
    
    public double pattern;
    public double bias;
    public double error;
    public double weights[];
    public double changes[];
    
	// Initialise a new Node
    public NeuralNode(int num_of_next_layer) {
        Random rng = new Random();
        pattern = 0.0;
        bias = rng.nextDouble() * 0.6 - 0.3;
        error = 0.0;
        weights = new double[num_of_next_layer];
        changes = new double[num_of_next_layer];
        for (int i = 0; i < num_of_next_layer; i++) {
            weights[i] = rng.nextDouble() * 0.6 - 0.3;
            changes[i] = 0.0;
        }
    }
    
	// Getters and Setters
    public double pattern() {
        return pattern;
    }
    
    public double bias() {
        return bias;
    }
    
    public double error() {
        return error;
    }
    
    public double weight(int node) {
        return weights[node];
    }
    
    public double change(int node) {
        return changes[node];
    }
    
    public void set_pattern(double pattern) {
        this.pattern = pattern;
    }
    
    public void set_bias(double bias) {
        this.bias = bias;
    }
    
    public void set_error(double error) {
        this.error = error;
    }
    
    public void set_weight(int node, double weight) {
        this.weights[node] = weight;
    }
    
    public void set_change(int node, double change) {
        this.changes[node] = change;
    }
}