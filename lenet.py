from   keras.layers  import  Conv2D,Dense,MaxPooling2D,Flatten,Activation
from keras.models import Sequential

from keras  import   backend  as K




class  LeNet:
    def build(width,height,depth,classes):
        model = Sequential()
        inputShare = (height,width,depth)
        if K.image_data_format() == 'channels_first':

            inputShare = (depth,height, width)
        model.add(Conv2D(20,(5,5),padding='same',input_shape=inputShare,activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(50,(5,5),padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides= (2,2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model



if __name__ == '__main__':
    LeNet = LeNet()


