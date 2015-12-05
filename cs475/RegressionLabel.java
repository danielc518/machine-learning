package cs475;

import java.io.Serializable;

public class RegressionLabel extends Label implements Serializable {

	private final double label;
	
	public RegressionLabel(double label) {
		this.label = label;
	}

	@Override
	public String toString() {
		return Double.toString(label);
	}
	
	public double getLabel() {
		return label;
	}

}
