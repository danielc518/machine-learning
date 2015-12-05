package cs475;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Set;

public class FeatureVector implements Serializable {
	
	/**
	 * Map for storing feature data
	 */
	private final HashMap<Integer, Double> fvMap = new HashMap<Integer, Double>();

	public void add(int index, double value) {
		fvMap.put(index, value);
	}
	
	public double get(int index) {
		if (fvMap.containsKey(index)) {
			return fvMap.get(index);
		} else {
			return -1.0;
		}
	}
	
	public Set<Entry<Integer, Double>> getEntrySet() {
		return fvMap.entrySet();
	}

}
