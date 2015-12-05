
package cs475.loopMRF;

public class LoopyBP {

	private LoopMRFPotentials potentials;
	private int iterations;

	/** Arrays below will store messages **/

	private double[][] nodeToFactorRight;
	private double[][] factorToNodeRight;

	private double[][] nodeToFactorLeft;
	private double[][] factorToNodeLeft;

	public LoopyBP(LoopMRFPotentials p, int iterations) {
		this.potentials = p;
		this.iterations = iterations;

		int n = p.loopLength();
		int k = p.numXValues();

		// Messages for first loop (i.e. part (a) in algorithm)
		this.nodeToFactorRight = new double[n + 1][k + 1];
		this.factorToNodeRight = new double[n + 1][k + 1];

		// Messages for second loop (i.e. part (b) in algorithm)
		this.nodeToFactorLeft = new double[n + 1][k + 1];
		this.factorToNodeLeft = new double[n + 1][k + 1];

		computeMessages();
	}

	private void computeMessages() {
		int n = this.potentials.loopLength();
		int k = this.potentials.numXValues();

		// Initialize messages
		for (int i = 1; i <= k; i++) {
			nodeToFactorRight[1][i] = 1; // mu_x_1 -> f_n+1 = 1
			nodeToFactorLeft[1][i] = 1; // mu_x_1 -> f_2n = 1
		}

		for (int t = 1; t <= this.iterations; t++) {
			// Messages for first loop (i.e. part (a) in algorithm)
			for (int i = 1; i <= n; i++) {
				factorToNodeRight[1 + (i % n)] = factorToNodeMsgRight(n + i, 1 + (i % n));
				nodeToFactorRight[1 + (i % n)] = nodeToFactorMsgRight(1 + (i % n));
			}

			// Messages for second loop (i.e. part (b) in algorithm)
			for (int i = n; i >= 1; i--) {
				factorToNodeLeft[i] = factorToNodeMsgLeft(n + i, i);
				nodeToFactorLeft[i] = nodeToFactorMsgLeft(i);
			}
		}
	}

	private double[] factorToNodeMsgRight(int f_i, int x_i) {
		int n = this.potentials.loopLength();
		int k = this.potentials.numXValues();
		
		double[] nodeToFactorMsg = null;
		
		if (f_i == 2 * n && x_i == 1) {
			nodeToFactorMsg = nodeToFactorRight[n];
		} else {
			nodeToFactorMsg = nodeToFactorRight[x_i - 1];
		}
		
		double[] result = new double[k + 1];
		for (int i = 1; i <= k; i++) {
			result[i] = 0;
			for (int j = 1; j <= k; j++) {
				result[i] += this.potentials.potential(f_i, j, i) * nodeToFactorMsg[j];
			}
		}
		
		return result;
	}

	private double[] factorToNodeMsgLeft(int f_i, int x_i) {
		int n = this.potentials.loopLength();
		int k = this.potentials.numXValues();
		
		double[] nodeToFactorMsg = null;
		
		if (f_i == 2 * n && x_i == n) {
			nodeToFactorMsg = nodeToFactorLeft[1];
		} else {
			nodeToFactorMsg = nodeToFactorLeft[x_i + 1];
		}
		
		double[] result = new double[k + 1];
		for (int i = 1; i <= k; i++) {
			result[i] = 0;
			for (int j = 1; j <= k; j++) {
				result[i] += this.potentials.potential(f_i, i, j) * nodeToFactorMsg[j];
			}
		}
		
		return result;
	}

	private double[] nodeToFactorMsgRight(int x_i) {
		return multiply(factorToNodeRight[x_i], getPotential(x_i));
	}

	private double[] nodeToFactorMsgLeft(int x_i) {
		return multiply(factorToNodeLeft[x_i], getPotential(x_i));
	}

	private double[] getPotential(int x_i) {
		int k = this.potentials.numXValues();

		double[] unary = new double[k + 1];

		for (int i = 1; i <= k; i++) {
			unary[i] = this.potentials.potential(x_i, i);
		}

		return unary;
	}
	
	private double[] multiply(double[] leftArray, double[] rightArray) {
		double[] result = new double[leftArray.length];
		for (int i = 0; i < leftArray.length; i++) {
			result[i] = leftArray[i] * rightArray[i];
		}
		return result;
	}

	public double[] marginalProbability(int x_i) {
		int k = this.potentials.numXValues();
		
		double z = 0.0;
		
		double[] probabilities = new double[k + 1];
		
		for (int i = 1; i <= k; i++) {
			probabilities[i] = factorToNodeLeft[x_i][i] * getPotential(x_i)[i] * factorToNodeRight[x_i][i];
			z += probabilities[i];
		}
		
		// Normalize
		
		for (int i = 1; i <= k; i++) {
			probabilities[i] /= z;
		}

		return probabilities;
	}

}
