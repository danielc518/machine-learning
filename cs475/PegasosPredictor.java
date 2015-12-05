package cs475;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class PegasosPredictor extends Predictor {

	/**
	 * Lambda
	 */
	private double pegasosLambda = 0.0001;

	/**
	 * Number of iterations
	 */
	private int sgdIterations = 20;

	/**
	 * Map of weights
	 */
	private final Map<Integer, Double> weightMap = new HashMap<Integer, Double>();

	@Override
	public void train(List<Instance> instances) {
		if (instances == null) {
			return;
		}

		int t = 1;

		for (int iteration = 0; iteration < sgdIterations; iteration++) {
			for (Instance instance : instances) {
				FeatureVector featureVector = instance.getFeatureVector();

				Set<Entry<Integer, Double>> featureEntrySet = featureVector.getEntrySet();

				ClassificationLabel currLabel = (ClassificationLabel) instance.getLabel();

				int labelValue = currLabel.getLabel() == 0 ? -1 : currLabel.getLabel();

				int indicatorValue = ((double) labelValue * getInnerProduct(featureVector, weightMap)) < 1 ? 1 : 0;

				Iterator<Entry<Integer, Double>> weightIterator = weightMap.entrySet().iterator();
				while (weightIterator.hasNext()) {
					Entry<Integer, Double> entry = weightIterator.next();
					weightMap.put(entry.getKey(), entry.getValue() * (1 - ((double) 1 / t)));
				}

				Map<Integer, Double> updateMap = new HashMap<Integer, Double>();
				Iterator<Entry<Integer, Double>> featureIterator = featureEntrySet.iterator();
				while (featureIterator.hasNext()) {
					Entry<Integer, Double> entry = featureIterator.next();
					updateMap.put(entry.getKey(), indicatorValue * labelValue * entry.getValue() / (pegasosLambda * t));
				}
				
				Iterator<Entry<Integer, Double>> updateIterator = updateMap.entrySet().iterator();
				while (updateIterator.hasNext()) {
					Entry<Integer, Double> entry = updateIterator.next();
					
					Integer key = entry.getKey();
					Double value = entry.getValue();
					
					if (weightMap.containsKey(key)) {
						weightMap.put(key, weightMap.get(key) + value);
					} else {
						if (value != null) {
							weightMap.put(key, value);
						}
					}
				}

				t++;
			}
		}
	}

	@Override
	public Label predict(Instance instance) {
		if (instance == null) {
			return null;
		}

		double innerProduct = getInnerProduct(instance.getFeatureVector(), weightMap);

		if (innerProduct < 0) {
			return new ClassificationLabel(0);
		} else {
			return new ClassificationLabel(1);
		}
	}

	private double getInnerProduct(FeatureVector featureVector, Map<Integer, Double> weightMap) {
		double innerProduct = 0.0;

		if (featureVector != null && weightMap != null) {
			Iterator<Entry<Integer, Double>> iterator = featureVector.getEntrySet().iterator();

			while (iterator.hasNext()) {
				Entry<Integer, Double> entry = iterator.next();

				Integer key = entry.getKey();
				
				if (weightMap.containsKey(key)) {
					innerProduct += (entry.getValue() * weightMap.get(key));
				}
			}
		}

		return innerProduct;
	}

	public void setPegasosLambda(double pegasosLambda) {
		this.pegasosLambda = pegasosLambda;
	}

	public void setSgdIterations(int sgdIterations) {
		this.sgdIterations = sgdIterations;
	}

}
