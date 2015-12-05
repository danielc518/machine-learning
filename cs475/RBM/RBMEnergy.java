package cs475.RBM;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RBMEnergy {
	
	private RBMParameters _parameters;
	private int _iters;
	private double _eta;
	
	public RBMEnergy(RBMParameters parameters, int iters, double eta) {
		this._parameters = parameters;
		this._iters = iters;
		this._eta = eta;
	}
	
	public void learning() {
		initializeParameters();
		
		int numExamples = this._parameters.numExamples();
		int numVisibleNodes = this._parameters.numVisibleNodes();
		int numHiddenNodes = this._parameters.numHiddenNodes();
		
		List<List<Integer>> allPossibleVisibleVectors = getAllPossibleBinaryVectors(numVisibleNodes);
		List<List<Integer>> allPossibleHiddenVectors = getAllPossibleBinaryVectors(numHiddenNodes);
		
		for (int iter = 0; iter < _iters; iter++) {
			Map<Integer, Map<Integer, Double>> sigmoidMap = new HashMap<Integer, Map<Integer, Double>>();
			
			/** Compute Z **/
			double zValue = 0.0;
			
			for (List<Integer> vVector : allPossibleVisibleVectors) {
				for (List<Integer> hVector : allPossibleHiddenVectors) {
					zValue += Math.exp(-computeEnergyFunction(vVector, hVector));
				}
			}
			
			/** Compute gradients of W **/
			
			double[][] weightGradients = new double[numVisibleNodes][numHiddenNodes];
			
			for (int i = 0; i < numVisibleNodes; i++) {
				for (int j = 0; j < numHiddenNodes; j++) {
					
					// Compute left term and right term in Eq. (17)
					
					double leftTermSum = 0.0;
					
					Map<Integer, Double> map = null;
					
					for (int t = 0; t < numExamples; t++) {
						double innerProduct = computeDataWeightInnerProduct(t, j);
						double sigmoidValue = computeSigmoid(innerProduct + this._parameters.getHiddenBias(j));
						
						if (map == null) {
							map = new HashMap<Integer, Double>();
							sigmoidMap.put(j, map);
						}
						
						map.put(t, sigmoidValue);
						
						leftTermSum += (sigmoidValue * (this._parameters.getExample(t, i)));
					}
					
					double rightTermSum = 0.0;
					
					for (List<Integer> vVector : allPossibleVisibleVectors) {
						if (vVector.get(i) == 1) {
							for (List<Integer> hVector : allPossibleHiddenVectors) {
								if (hVector.get(j) == 1) {
									rightTermSum += (Math.exp(-computeEnergyFunction(vVector, hVector)) / zValue);
								}
							}
						}
					}
					
					weightGradients[i][j] = leftTermSum - (numExamples * rightTermSum);
				}
			}
			
			/** Compute gradients of b **/
			double[] bGradients = new double[numVisibleNodes];
			
			for (int i = 0; i < numVisibleNodes; i++) {
				double leftTermSum = 0.0;
				
				for (int t = 0; t < numExamples; t++) {
					leftTermSum += this._parameters.getExample(t, i);
				}
				
				double rightTermSum = 0.0;
				
				for (List<Integer> vVector : allPossibleVisibleVectors) {
					if (vVector.get(i) == 1) {
						for (List<Integer> hVector : allPossibleHiddenVectors) {
							rightTermSum += (Math.exp(-computeEnergyFunction(vVector, hVector)) / zValue);
						}
					}
				}
				
				bGradients[i] = leftTermSum - (numExamples * rightTermSum);
			}
			
			/** Compute gradients of d **/
			double[] dGradients = new double[numHiddenNodes];
			
			for (int j = 0; j < numHiddenNodes; j++) {
				double leftTermSum = 0.0;
				
				for (int t = 0; t < numExamples; t++) {
					leftTermSum += sigmoidMap.get(j).get(t);
				}
				
				double rightTermSum = 0.0;
				
				for (List<Integer> vVector : allPossibleVisibleVectors) {
					for (List<Integer> hVector : allPossibleHiddenVectors) {
						if (hVector.get(j) == 1) {
							rightTermSum += (Math.exp(-computeEnergyFunction(vVector, hVector)) / zValue);
						}
					}
				}
				
				dGradients[j] = leftTermSum - (numExamples * rightTermSum);
			}
			
			/** Update parameters **/
			for (int j = 0; j < numHiddenNodes; j++) {
				this._parameters.setHiddenBias(j, this._parameters.getHiddenBias(j) + (_eta * dGradients[j]));
			}
			
			for (int i = 0; i < numVisibleNodes; i++) {
				for (int j = 0; j < numHiddenNodes; j++) {
					this._parameters.setWeight(i, j, this._parameters.getWeight(i, j) + (_eta * weightGradients[i][j]));
				}
				
				this._parameters.setVisibleBias(i, this._parameters.getVisibleBias(i) + (_eta * bGradients[i]));
			}
		}
	}
	
	private void initializeParameters() {
		int numVisibleNodes = this._parameters.numVisibleNodes();
		int numHiddenNodes = this._parameters.numHiddenNodes();
		
		for (int i = 0; i < numVisibleNodes; i++) {
			for (int j = 0; j < numHiddenNodes; j++) {
				this._parameters.setWeight(i, j, (j + 1) % 2 == 0 ? 1 : 0);
			}
		}
		
		for (int i = 0; i < numVisibleNodes; i++) {
			this._parameters.setVisibleBias(i, (i + 1) % 2 == 0 ? 1 : 0);
		}
		
		for (int j = 0; j < numHiddenNodes; j++) {
			this._parameters.setHiddenBias(j, (j + 1) % 2 == 0 ? 1 : 0);
		}
	}
	
	private double computeDataWeightInnerProduct(int exampleIndex, int fixedColumn) {
		double innerProduct = 0.0;
		
		int numVisibleNodes = this._parameters.numVisibleNodes();
		
		for (int i = 0; i < numVisibleNodes; i++) {
			double weight = this._parameters.getWeight(i, fixedColumn);
			double example = this._parameters.getExample(exampleIndex, i);
			
			innerProduct += (weight * example);
		}
		
		return innerProduct;
	}
	
	private double computeDataWeightInnerProduct(List<Integer> vVector, int fixedColumn) {
		double innerProduct = 0.0;
		
		int numVisibleNodes = this._parameters.numVisibleNodes();
		
		for (int i = 0; i < numVisibleNodes; i++) {
			innerProduct += (this._parameters.getWeight(i, fixedColumn) * vVector.get(i));
		}
		
		return innerProduct;
	}
	
	private double computeSigmoid(double value) {
		return 1 / (1 + Math.exp(-value));
	}
	
	private double computeEnergyFunction(List<Integer> vVector, List<Integer> hVector) {
		int numVisibleNodes = this._parameters.numVisibleNodes();
		int numHiddenNodes = this._parameters.numHiddenNodes();
		
		double leftTerm = 0.0; // -x^{T}Wh
		for (int j = 0; j < numHiddenNodes; j++) {
			leftTerm += (computeDataWeightInnerProduct(vVector, j) * hVector.get(j));
		}
		
		double middleTerm = 0.0; // b^{T}x
		for (int i = 0; i < numVisibleNodes; i++) {
			middleTerm += (this._parameters.getVisibleBias(i) * vVector.get(i));
		}
		
		double rightTerm = 0.0; // d^{T}h
		for (int j = 0; j < numHiddenNodes; j++) {
			rightTerm += (this._parameters.getHiddenBias(j) * hVector.get(j));
		}
		
		return -leftTerm - middleTerm - rightTerm;
	}
	
	private List<List<Integer>> getAllPossibleBinaryVectors(int size) {
		List<List<Integer>> possibleList = new ArrayList<List<Integer>>();
		
		for(int k = 0; k < (1 << size); k++){
			List<Integer> hVector = new ArrayList<Integer>(size);
			
			String binaryString = Integer.toBinaryString(k);
			
			int length = binaryString.length();
			
			int n = 0;
			
			while (n < size) {
				if (n < length) {
					hVector.add(Character.getNumericValue(binaryString.charAt(n)));
				} else {
					hVector.add(0, 0);
				}
				
				n++;
			}
			
			possibleList.add(hVector);
		}
		
		return possibleList;
	}

	public void printParameters() {
		//NOTE: Do not modify this function
		for (int i=0; i<_parameters.numVisibleNodes(); i++)
			System.out.println("b_" + i + "=" + _parameters.getVisibleBias(i));
		for (int i=0; i<_parameters.numHiddenNodes(); i++)
			System.out.println("d_" + i + "=" + _parameters.getHiddenBias(i));
		for (int i=0; i<_parameters.numVisibleNodes(); i++)
			for (int j=0; j<_parameters.numHiddenNodes(); j++)
				System.out.println("W_" + i + "_" + j + "=" + _parameters.getWeight(i,j));
	}
	
}
