
/* Simple class to run the Neural Network */
public class Main {
	
	public static void main(String[] args) {
		/*
		NeuralNetwork network = new NeuralNetwork();
		network.run(false);
		*/
		
		boolean loop = false;
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("l")) {
				loop = true;
			}
		}
		
		if (loop) {
			for (int i = 0; i < 30; i++) {
				NeuralNetwork network = new NeuralNetwork();
				network.run(false);
			}
			System.out.println();
		}
		else {
			NeuralNetwork network = new NeuralNetwork();
			network.run(true);
			
		}
		
	}
	
}