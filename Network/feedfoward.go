package Network

import (
	"compress/gzip"
	"encoding/gob"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"os"
)

type Layer struct {
	//weights between current layer and next
	Weight *mat.Dense

	//bias of the layer
	Biases *mat.Dense

	//activation of the layer
	Activation *mat.Dense
}


type Network struct {
	Layers []Layer
}


func CreateNetwork(layer_init []int) (*Network) {
	layers := []Layer{}
	for i := 1; i < len(layer_init); i++ {
		NNodePrev := layer_init[i-1]
		NNoddes := layer_init[i]

		weightData := make([]float64, NNoddes*NNodePrev)
		for i := range weightData {
			weightData[i] = rand.NormFloat64()
		}

		biasData := make([]float64, NNoddes)
		for i := range biasData {
			biasData[i] = rand.NormFloat64()
		}

		layer := Layer{
			mat.NewDense(NNodePrev, NNoddes, weightData),
			mat.NewDense(1, NNoddes, biasData),
			mat.NewDense(1, NNoddes, nil),
		}
		layers = append(layers, layer)
	}
	return &Network{layers}
}

func (n *Network) Predict(input[] float64)  {
	result := mat.NewDense(0,len(input),input)
	for i := range n.Layers {
		result.Mul(n.Layers[i].Weight,result)
		result.Add(result,n.Layers[i].Biases);

		n.Layers[i].Activation.Copy(result)

		result.Apply(func(i, j int, v float64) float64 {
			return Sigmoid(v);
		},result)
	}
}

func (n *Network) BackPropigation(answer[] float64)  {
}

func (n* Network) SaveNetwork(file string)  error{
	fi, err := os.Create(file)
	if err != nil{
		return err
	}
	fz := gzip.NewWriter(fi)
	defer fi.Close()

	encoder := gob.NewEncoder(fz)
	err = encoder.Encode(n.Layers)
	if err != nil {
		return err
	}
	return nil
}

func (n* Network) LoadNetwork(file string)  error{
	fi, err := os.Open(file)
	if err != nil{
		return err
	}
	fz, err := gzip.NewReader(fi)
	defer fi.Close()


	if err != nil {
		return err
	}
	decoder := gob.NewDecoder(fz)
	err = decoder.Decode(n.Layers)
	if err != nil {
		return err
	}
	return nil
}