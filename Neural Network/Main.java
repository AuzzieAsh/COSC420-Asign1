/*
  File: Main.java
  Author: Ashley Manson
  Description: Simple class to run the Neural Network, can take arguments 
  from the command line.
  Developed with Java Version: 1.8.0_45
*/

public class Main {
	
    public static void main(String[] args) {
		
	int repeat_for = 1;
	// flags
	boolean do_loop = false;
	boolean show_epochs = false;
	boolean show_weights = false;
	boolean show_activation = false;
	boolean only_help = false;
		
	if (args.length > 0) {
	    if (args[0].equals("help")) {
		only_help = true;
		System.out.println("Program takes either 1 or 2 arguments\n" + 
				   "First argument should consist of chars\n" + 
				   "e = Show Population Error and Epochs every 100 epochs\n" + 
				   "w = Show the final Weights\n" +
				   "a = Show the activation of all nodes in the Network\n" + 
				   "l = Loop the program for a specified number of times\n" +
				   "Second argument should be an integer\n" +
				   "Program usage: \"java Main awel 10\" for example\n" + 
				   "If no arguments are given, the program will run once");
	    }
	    if (!only_help) {
		for (int i = 0; i < args[0].length(); i++) {
		    if (args[0].charAt(i) == 'l') {
			try {
			    do_loop = true;
			    repeat_for = Integer.parseInt(args[1]);
			}
			catch (ArrayIndexOutOfBoundsException |
			       NumberFormatException e) {
			    do_loop = false;
			    repeat_for = 1;
			}
		    }
					
		    else if (args[0].charAt(i) == 'e') {
			show_epochs = true;
		    }
		    else if (args[0].charAt(i) == 'w') {
			show_weights = true;
		    }
		    else if (args[0].charAt(i) == 'a') {
			show_activation = true;
		    }
		}
	    }
	}
		
	if (!only_help) {
	    for (int run = 0; run < repeat_for; run++) {
		if (do_loop) System.out.printf("=== Run %d ===\n", run + 1);
		NeuralNetwork network = new NeuralNetwork();
		network.run(show_epochs, show_weights, show_activation);
		if (do_loop) System.out.println();
	    }
	}
    }
}
