package cs475.RBM2;

import java.util.Random;

public class RBMEnergy {
	
	private RBMParameters parameters;
	private int num_samples;

	private int[][] hiddenSample;

	public RBMEnergy(RBMParameters parameters, int numSamples) {
		this.parameters = parameters;
		this.num_samples = numSamples;
		computeSamples();
	}
	
	public double computeMarginal(int j) {
		int count = 0;

		for (int i = 1; i <= this.num_samples; i++) {
			if (this.hiddenSample[i][j] == 1) {
				count++;
			}
		}

		return (double) count / this.num_samples;
	}

	private void computeSamples() {
		Random random = new Random(0);

		int numVisibleNodes = this.parameters.numVisibleNodes();
		int numHiddenNodes = this.parameters.numHiddenNodes();

		this.hiddenSample = new int[this.num_samples + 1][numHiddenNodes + 1];

		// Initialize X samples
		int[] xSamples = new int[numVisibleNodes + 1];
		for (int i = 1; i <= numVisibleNodes; i++) {
			xSamples[i] = i % 2 == 0 ? 1 : 0;
		}

		// Generate samples
		for (int i = 1; i <= this.num_samples; i++) {
			this.hiddenSample[i] = generateHiddenSample(xSamples, random);
			xSamples = generateVisibleSample(this.hiddenSample[i], random);
		}
	}

	private int[] generateHiddenSample(int[] visibleSample, Random random) {
		int numHiddenNodes = this.parameters.numHiddenNodes();
		
		int[] hiddenSample = new int[numHiddenNodes + 1];

		for (int j = 1; j <= numHiddenNodes; j++) {
			double innerProduct = getWeightColInnerProduct(j, visibleSample);

			double probability = computeSigmoid(innerProduct + this.parameters.hiddenBias(j));

			if (random.nextDouble() < probability) {
				hiddenSample[j] = 1;
			} else {
				hiddenSample[j] = 0;
			}
		}
		
		return hiddenSample;
	}

	private int[] generateVisibleSample(int[] hiddenSample, Random random) {
		int numVisibleNodes = this.parameters.numVisibleNodes();
		
		int[] visibleSample = new int[numVisibleNodes + 1];

		for (int i = 1; i <= numVisibleNodes; i++) {
			double innerProduct = getWeightRowInnerProduct(i, hiddenSample);

			double probability = computeSigmoid(innerProduct + this.parameters.visibleBias(i));

			if (random.nextDouble() < probability) {
				visibleSample[i] = 1;
			} else {
				visibleSample[i] = 0;
			}
		}
		
		return visibleSample;
	}

	private double getWeightColInnerProduct(int colIndex, int[] xSamples) {
		double innerProduct = 0.0;
		for (int i = 1; i <= this.parameters.numVisibleNodes(); i++) {
			innerProduct += this.parameters.weight(i, colIndex) * xSamples[i];
		}
		return innerProduct;
	}

	private double getWeightRowInnerProduct(int rowIndex, int[] hSamples) {
		double innerProduct = 0.0;
		for (int i = 1; i <= this.parameters.numHiddenNodes(); i++) {
			innerProduct += this.parameters.weight(rowIndex, i) * hSamples[i];
		}
		return innerProduct;
	}

	private double computeSigmoid(double value) {
		return 1 / (1 + Math.exp(-value));
	}
}
