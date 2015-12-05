package cs475;

import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

public class EvenOddPredictor extends Predictor {

	@Override
	public void train(List<Instance> instances) {
	}

	@Override
	public Label predict(Instance instance) {
		FeatureVector featureVector = instance.getFeatureVector();
		
		// Iterate over feature vector values
		
		Set<Entry<Integer, Double>> entrySet = featureVector.getEntrySet();
		
		Iterator<Entry<Integer, Double>> iterator = entrySet.iterator();
		
		double evenSum = 0.0;
		double oddSum = 0.0;
		
		while (iterator.hasNext()) {
			Entry<Integer, Double> entry = iterator.next();
			if (entry.getKey() % 2 == 0) {
				evenSum += entry.getValue();
			} else {
				oddSum += entry.getValue();
			}
		}
		
		if (evenSum >= oddSum) {
			return new ClassificationLabel(1);
		} else {
			return new ClassificationLabel(0);
		}
	}

}
