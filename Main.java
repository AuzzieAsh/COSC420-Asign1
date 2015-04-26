
/* Simple class to run the Neural Network */
public class Main {
	
	public static void main(String[] args) {

		boolean loop = false;
		boolean show_epochs = false;
		boolean show_weights = false;
		boolean show_activation = false;
		int repeat_for = 1;
		
		if (args.length > 0) {
			for (int i = 0; i < args[0].length(); i++) {
				if (args[0].charAt(i) == 'l') {
					try {
						loop = true;
						repeat_for = Integer.parseInt(args[1]);
					}
					catch (ArrayIndexOutOfBoundsException | NumberFormatException e) {
						System.err.printf("Second argument not a number.\n");
						loop = false;
					}
				}
				
				if (args[0].charAt(i) == 'e') {
					show_epochs = true;
				}
				if (args[0].charAt(i) == 'w') {
					show_weights = true;
				}
				if (args[0].charAt(i) == 'a') {
					show_activation = true;
				}
			}
		}
		// show weights, print pop error and epochs, test the input patterns and see the activation of all units
		if (loop) {
			for (int i = 0; i < repeat_for; i++) {
				NeuralNetwork network = new NeuralNetwork();
				network.run(false, false, false);
			}
			System.out.println();
		}
		else {
			NeuralNetwork network = new NeuralNetwork();
			network.run(show_epochs, show_weights, show_activation);
			
		}
		
	}
	
}