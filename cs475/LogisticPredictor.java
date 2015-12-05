package cs475;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class LogisticPredictor extends Predictor {
	
	/**
	 * Initial eta value
	 */
	private double sgdEta = 0.01;
	
	/**
	 * Number of iterations
	 */
	private int sgdIterations = 20;
	
	/**
	 * Map between feature index and its corresponding weight
	 */
	private Map<Integer, Double> weightMap = new HashMap<Integer, Double>();
	
	/**
	 * Map between training example index and a map between feature index and gradient value
	 */
	private Map<Integer, Map<Integer, Double>> gradientMap = new HashMap<Integer, Map<Integer, Double>>();

	@Override
	public void train(List<Instance> instances) {
		if (instances == null) {
			return;
		}
		
		for (int iteration = 0; iteration < sgdIterations; iteration++) {
			for (int i = 0; i < instances.size(); i++) {
				Instance instance = instances.get(i);
				
				Label label = instance.getLabel();
				
				if (label instanceof ClassificationLabel) {
					int labelValue = ((ClassificationLabel) label).getLabel();
					
					FeatureVector featureVector = instance.getFeatureVector();
					
					Set<Entry<Integer, Double>> entrySet = featureVector.getEntrySet();
					
					double weightExampleProduct = getWeightExampleProduct(entrySet.iterator());
					
					Iterator<Entry<Integer, Double>> iterator = entrySet.iterator();
					
					while (iterator.hasNext()) {
						Entry<Integer, Double> entry = iterator.next();
						
						int j = entry.getKey(); // Feature index
						
						Double exampleValue = entry.getValue();
						
						if (exampleValue != null) {
							double gradientValue = getGradientValue(labelValue, weightExampleProduct, exampleValue);
							
							if (!gradientMap.containsKey(i)) {
								gradientMap.put(i, new HashMap<Integer, Double>());
							}
							
							gradientMap.get(i).put(j, gradientValue);
							
							double etaValue = getEtaValue(i, j);
							
							double updateValue = etaValue * gradientValue;
							
							if (weightMap.containsKey(j)) {
								Double currentWeight = weightMap.get(j);
								
								if (currentWeight == null) {
									weightMap.put(j, updateValue);
								} else {
									weightMap.put(j, currentWeight + updateValue);
								}
							} else {
								weightMap.put(j, updateValue);
							}
						}
					}
				}
			}
		}
	}

	@Override
	public Label predict(Instance instance) {
		if (instance == null) {
			return null;
		}
		
		FeatureVector featureVector = instance.getFeatureVector();
		
		Set<Entry<Integer, Double>> entrySet = featureVector.getEntrySet();
		
		double weightExampleProduct = getWeightExampleProduct(entrySet.iterator());
		
		double logisticValue = getLogisticValue(weightExampleProduct);
		
		if (logisticValue < 0.5) {
			return new ClassificationLabel(0);
		} else {
			return new ClassificationLabel(1);
		}
	}
	
	private double getWeightExampleProduct(Iterator<Entry<Integer, Double>> iterator) {
		double innerProduct = 0.0;
		
		while (iterator.hasNext()) {
			Entry<Integer, Double> entry = iterator.next();
			
			Double exampleValue = entry.getValue();
			
			if (exampleValue != null) {
				if (weightMap.containsKey(entry.getKey())) {
					Double weightValue = weightMap.get(entry.getKey());
					if (weightValue != null) {
						innerProduct += (weightValue * exampleValue);
					}
				}
			}
		}
		
		return innerProduct;
	}
	
	private double getGradientValue(double labelValue, double weightExampleProduct, Double exampleValue) {
		return (labelValue * getLogisticValue(-weightExampleProduct) * exampleValue) 
				+ ((1 - labelValue) * getLogisticValue(weightExampleProduct) * (-exampleValue));
	}
	
	private double getEtaValue(int i, int j) {
		double etaValue = 0.0;
		
		double partialGradientSumOfSquares = 0.0;
		
		for (int t = 0; t < i + 1; t++) {
			Map<Integer, Double> map = gradientMap.get(t);
			if (map != null && map.containsKey(j)) {
				partialGradientSumOfSquares += Math.pow(map.get(j), 2);
			}
		}
		
		etaValue = sgdEta / Math.pow(1 + partialGradientSumOfSquares, 0.5);
		
		return etaValue;
	}
	
	private double getLogisticValue(double param) {
		return 1 / (1 + Math.exp(-param));
	}

	public void setSgdEta(double sgdEta) {
		this.sgdEta = sgdEta;
	}

	public void setSgdIterations(int sgdIterations) {
		this.sgdIterations = sgdIterations;
	}

}
