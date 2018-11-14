package Network

import "math"

func Sigmoid(input float64) (float64){
	return (1.0/(1 + math.Exp(input)))
}

func DiffrentialSigmoid(input float64) (float64)  {
	return (1.0 - Sigmoid(input)) * Sigmoid(input)
}