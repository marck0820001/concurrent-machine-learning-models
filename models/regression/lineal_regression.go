package regression

// Linear Regression Model
// This is a simple implementation of linear regression in Go
type LinearRegression struct {
	Weights      []float64
	Bias         float64
	LearningRate float64
	Epochs       int
}

// NewLinearRegression function
func NewLinearRegression(learningRate float64, epochs int) *LinearRegression {
	return &LinearRegression{
		LearningRate: learningRate,
		Epochs:       epochs,
	}
}

// Train function
func (lr *LinearRegression) Train(X [][]float64, Y []float64) {
	//create the length of the samples (row) and the number of features (columns)
	// and initialize the weights
	numSamples := len(X)
	numFeatures := len(X[0])
	//create the array of weights with the len of the features for the frist row
	lr.Weights = make([]float64, numFeatures)
	//Loop of epochs
	for epoch := 0; epoch < lr.Epochs; epoch++ {
		// Initialize gradients
		// create the array of gradients with the len of the number of features
		// and the bias with 0.0
		gradW := make([]float64, numFeatures)
		gradB := 0.0
		// Loop of the samples
		for i := 0; i < numSamples; i++ {
			// make the predict of the first row of the sample
			yPred := lr.PredictOne(X[i])
			// calculate the error of the predict with the real valu
			// this is part of the gradient descent, with mathematical operations
			error := yPred - Y[i]
			// Loop of the features
			for j := 0; j < numFeatures; j++ {
				// calculate the gradient of the weight with the error and the sample
				gradW[j] += error * X[i][j]
			}
			// calculate the gradient of the bias with the error
			gradB += error
		}

		// Update weights and bias using gradients
		for j := 0; j < numFeatures; j++ {

			lr.Weights[j] -= lr.LearningRate * 2 * gradW[j] / float64(numSamples)
		}
		lr.Bias -= lr.LearningRate * 2 * gradB / float64(numSamples)
	}

}

// Predict functions for multiple samples
func (lr *LinearRegression) Predict(X [][]float64) []float64 {

	predictions := make([]float64, len(X))
	for i := 0; i < len(X); i++ {
		predictions[i] = lr.PredictOne(X[i])
	}
	return predictions
}

// Predict function for one sample
func (lr *LinearRegression) PredictOne(X []float64) float64 {
	return lr.dotProduct(lr.Weights, X) + lr.Bias
}

// dotProduct function for the weights and the sample
func (lr *LinearRegression) dotProduct(vec1, vec2 []float64) float64 {
	sum := 0.0
	for i := 0; i < len(vec1); i++ {
		sum += vec1[i] * vec2[i]
	}
	return sum
}
