package cs475;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MajorityPredictor extends Predictor {
	
	private int majorityLabel;

	@Override
	public void train(List<Instance> instances) {
		if (instances == null) {
			return;
		}
		
		/* 
		 * Assumes classification labels ONLY
		 */
		
		// Will keep track of frequency of 1's and 0's for training data
		Map<Integer, Integer> histogram = new HashMap<Integer, Integer>();
		histogram.put(0, new Integer(0));
		histogram.put(1, new Integer(0));

		for (Instance instance : instances) {
			Label label = instance.getLabel();
			if (label instanceof ClassificationLabel) {
				int labelValue =((ClassificationLabel) label).getLabel();
				if (labelValue == 0 || labelValue == 1) {
					Integer integer = histogram.get(labelValue);
					histogram.put(labelValue, integer + 1);
				}
			}
		}
		
		int numOfZero = histogram.get(0);
		int numOfOne = histogram.get(1);
		
		if (numOfZero > numOfOne) {
			majorityLabel = 0;
		} else if (numOfOne > numOfZero) {
			majorityLabel = 1;
		} else {
			majorityLabel = 1;
		}
	}

	@Override
	public Label predict(Instance instance) {
		return new ClassificationLabel(majorityLabel);
	}

}
