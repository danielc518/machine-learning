package cs475;

import java.io.Serializable;

public class ClassificationLabel extends Label implements Serializable {

	private final int label;
	
	public ClassificationLabel(int label) {
		this.label = label;
	}

	@Override
	public String toString() {
		return Integer.toString(label);
	}

	public int getLabel() {
		return label;
	}

}
