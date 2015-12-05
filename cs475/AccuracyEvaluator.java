package cs475;

import java.util.List;

public class AccuracyEvaluator extends Evaluator {

	@Override
	public double evaluate(List<Instance> instances, Predictor predictor) {
		if (instances != null && predictor != null) {
			int numOfMatch = 0; // Number of matches
			
			for (Instance instance : instances) {
				Label trueLabel = instance.getLabel();
				Label predictedLabel = predictor.predict(instance);
				
				if (trueLabel == null || predictedLabel == null) {
					continue;
				}
				
				if (trueLabel.toString().equals(predictedLabel.toString())) {
					numOfMatch += 1;
				}
			}
			
			return (double) numOfMatch / (double) instances.size();
		}
		
		return 0;
	}

}
