from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Reshape
from keras import backend as K


x = [[0],[1],[2],[3],[4]]

model = Sequential()
# model.add(Flatten(input_shape =((1,) + env.observation_space.shape)))
model.add(Embedding(input_dim=5, output_dim=5, input_length=1, embeddings_initializer='identity'))
model.add(Reshape((5,)))
layer0_output_fcn = K.function([model.layers[0].input], [model.layers[0].output])
layer1_output_fcn = K.function([model.layers[0].input], [model.layers[1].output])
layer0_output = layer0_output_fcn([x])[0]
layer1_output = layer1_output_fcn([x])[0]
print 'Input'
print x
print 'Layer 0 Output'
print layer0_output
print 'Layer 1 Output'
print layer1_output
