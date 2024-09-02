# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomUniform
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import time

# The import of the dataset
def preprocess_image(folder_paths,target_size=(64,64)):
    images_paths = []
    for filename in os.listdir(folder_paths):
        if filename.endswith('.jpg'):
            image_filename = os.path.join(folder_paths,filename)
            images_paths.append(image_filename)
    images=[]
    for path in images_paths:
        image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if image is not None:
           image=cv2.resize(image,target_size)
           images.append(image/255.0)
        else:
            print("Image at path ",path,"cannot be read and it is skipoed.")
    return np.array(images)

#Visulization
def plot_grayscale_histogram(images,num_cols=5,cmap='gray'):
    num_images=len(images)
    num_rows=(num_images+num_cols-1)//num_cols
    fig,axes=plt.subplots(num_rows,num_cols,figsize=(15,5*num_rows))
    axes=axes.flatten()
    for i,(ax,image) in enumerate(zip(axes,images)):
        ax.imshow(image,cmap=cmap)
        #ax.axis('off')
        ax.set_title(f'Image{i+1} grayscale intensity')

    plt.tight_layout()
    plt.show()

#The foundation of the model

class ReservoirLayer(Layer):
    def __init__(self, input_dim, reservoir_dim, **kwargs):
        super(ReservoirLayer,self).__init__(**kwargs)
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.states=None

    def build(self, input_shape):
        self.input_dim=input_shape[0]*input_shape[1]*input_shape[2]
        self.W_in=self.add_weight(#This is for the wight of the input to the reservoir
            name='W_in',
            shape=(self.input_dim,self.reservoir_dim),
            initializer='random_uniform',
            trainable=False
        )
        self.W=self.add_weight(#This is the weight in the reservoir
            shape=(self.reservoir_dim,self.reservoir_dim),
            initializer='random_uniform',
            trainable=False,
            name='W'
        )
        super(ReservoirLayer,self).build(input_shape)
    def call(self, inputs):#The method of forward propagation can be decided in this class function
        if inputs.shape.rank==4:
           inputs_flat=tf.reshape(inputs,[-1,self.input_dim])
        else:
            raise ValueError("inputs must have rank 4, but got {}".format(inputs.shape.rank))
        x=tf.matmul(inputs_flat,self.W_in)
        expand_mask=tf.expand_dims(self.states,axis=0)
        h=tf.matmul(expand_mask,self.W)
        h=tf.reshape(h,[-1,self.reservoir_dim])
        if x.shape.rank==h.shape.rank:
           res=tf.add(x,h)
        else:
            raise ValueError("The dimensions of inputs must match the dimensions of reservoir, but the shape of x is ",x.shape.rank," and the shape of h is ",h.shape.rank)
    #Refresh the state of the reservoir
        #self.states=tf.tanh(res)
        new_states=tf.tanh(res)
        return new_states

def create_esn(input_shape, reservoir_dim,num_classes):
 # Input layer
    inputs=Input(shape=input_shape)
   # print(inputs.shape)
 # Initialize the state of the reservoir
    reservoir_layer=ReservoirLayer(input_dim=np.prod(input_shape),reservoir_dim=reservoir_dim)
    reservoir_layer.build(input_shape)
    states=tf.zeros([reservoir_layer.reservoir_dim])
    reservoir_layer.states=states #set the original states

    reservoir_output=reservoir_layer(inputs)#The state of the reservoir refreshing   Problem!!!
    #reservoir_output_2d=tf.reshape(reservoir_output,[-1,reservoir_layer.reservoir_dim])
    outputs=Dense(num_classes,activation='softmax')(reservoir_output)#Output layer
    model=Model(inputs=inputs,outputs=outputs)
    return model

#Random layers for the convolution
class Randomlayer(Layer):
    def __init__(self, output_dim, **kwargs):
        super(Randomlayer,self).__init__(**kwargs)
        self.output_dim=output_dim
    def build(self, input_shape):
        self.w=self.add_weight(
            shape=(input_shape[-1], self.output_dim),
            initializer='random_uniform',
            trainable=True
        )
        self.b=self.add_weight(
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )
    def call(self, inputs):
        weight_input=tf.matmul(inputs,self.w)
        bias_expand=tf.expand_dims(self.b,axis=0)
        bias_expand=tf.tile(bias_expand,[tf.shape(weight_input)[0],1])
        new_situation=tf.add(weight_input,bias_expand)
        return new_situation
#The structure of the convolution model
def create_esn2(input_shape, num_classes):
    inputs=Input(shape=input_shape)
    #The operation of the first convolution layer
    conv1=Conv2D(32,(3,3),activation='relu')(inputs)
    flat1=Flatten()(conv1)
    x1=Randomlayer(256)(flat1)
    #The second one
    conv2=Conv2D(64,(3,3),activation='relu')(inputs)
    flat2=Flatten()(conv2)
    x2=Randomlayer(128)(flat2)
    combined=tf.keras.layers.concatenate([x1,x2],axis=-1)
    fc=Dense(64,activation='relu')(combined)
    outputs=Dense(num_classes,activation='softmax')(fc)
    model=Model(inputs=inputs,outputs=outputs)
    return model

def create_cnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    # The operation of the first convolution layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    pool1=MaxPooling2D((2,2))(conv1)
    x11 = Flatten()(pool1)
    x1=Randomlayer(256)(x11)
    # The second one
    conv2 = Conv2D(64, (3, 3), activation='relu')(inputs)
    pool2=MaxPooling2D((2,2))(conv2)
    x22 = Flatten()(pool2)
    x2=Randomlayer(128)(x22)
    combined = tf.keras.layers.concatenate([x1, x2], axis=-1)
    flat = Flatten()(combined)
    fc=Dense(128,activation='relu')(flat)
    outputs = Dense(num_classes, activation='softmax')(fc)
    model = Model(inputs=inputs, outputs=outputs)
    return model
#The class for LSMs
class LiquidMachineLayer(Layer):
    def __init__(self,input_dim, reservoir_dim, decay_rate, current_time,**kwargs):
        super(LiquidMachineLayer,self).__init__(**kwargs)
        self.input_dim=input_dim
        self.reservoir_dim=reservoir_dim
        self.decay_rate=decay_rate  # The declining rate
        self.current_time=current_time
        self.states=None


    def build(self, input_shape):
        #Initialize the weights in our reservoir
        self.input_weight=self.add_weight(
            shape=(self.input_dim, self.reservoir_dim),
            initializer='random_uniform',
            trainable=False,
            name='W_in'
        )
        self.recurrent_weight=self.add_weight(
            shape=(self.reservoir_dim, self.reservoir_dim),
            initializer='random_uniform',
            trainable=False,
            name='W'
        )
        # self.states=self.add_weight(
        #     shape=(self.reservoir_dim,),
        #     initializer='zero',
        #     trainable=False,
        #     name='states'
        # )
        # Here initialize the state of our reservoir
        super(LiquidMachineLayer,self).build(input_shape)

    # The method of forward propagation
    def call(self, inputs, training=False):

        # Renew the weights firstly
        New_W_in=self.input_weight*(self.decay_rate**self.current_time)
        New_recurrent_weight=self.recurrent_weight*(self.decay_rate**self.current_time)

        if inputs.shape.rank==4:
            inputs_flat=tf.reshape(inputs,[-1,self.input_dim])
        else:
            raise ValueError("inputs must have rank 4, but got {}".format(inputs.shape.rank))

        # Calculate new situation
        x=tf.matmul(inputs_flat,New_W_in)
        expand_mask=tf.expand_dims(self.states, axis=0)
        h=tf.matmul(expand_mask,New_recurrent_weight)
        h=tf.reshape(h,[-1, self.reservoir_dim])

        res=tf.add(x,h)
        new_states=tf.tanh(res)
        if training:
            self.current_time=self.current_time+1
        return new_states
# Create the LSM model
def create_LSM(input_shape,reservoir_dim, num_classes,decay_rate):
    inputs=Input(shape=input_shape)

    reservoir_layer=LiquidMachineLayer(decay_rate=decay_rate,
                                       reservoir_dim=reservoir_dim,
                                       input_dim=np.prod(input_shape),
                                       current_time=0,
                                       )
    reservoir_layer.build(input_shape)
    reservoir_layer.states=tf.zeros([reservoir_layer.reservoir_dim])
    reservoir_output=reservoir_layer(inputs)
    outputs=Dense(num_classes,activation='softmax')(reservoir_output)
    model=Model(inputs=inputs,outputs=outputs)
    return model

# The class for snn
class SpikingNeuronLayer(Layer):
    def __init__(self, num_neurons, rate, current_time, threshold=1.0, reset_value=0.0, **kwargs):
        super(SpikingNeuronLayer,self).__init__(**kwargs)
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.reset_value = reset_value
        self.current_time = current_time
        self.rate = rate
        self.initialized = False
        self.spike_state = None
        self.membrane_potential = None
        # 使用 callable 来初始化状态变量
        # self.spike_state_initializer = lambda: tf.zeros([1, self.num_neurons], dtype=tf.float32)
        # self.membrane_potential_initializer = lambda: tf.zeros([1, self.num_neurons], dtype=tf.float32)
        # 在图初始化阶段调用 callable 来创建变量
        # self.spike_state = tf.Variable(initial_value=self.spike_state_initializer(), trainable=False)
        # self.membrane_potential = tf.Variable(initial_value=self.membrane_potential_initializer(), trainable=False)

    def build(self,input_shape):
        # Initialize the state of neurons
        self.kernel=self.add_weight(shape=(np.prod(input_shape),self.num_neurons),
                                    initializer='random_uniform',
                                    trainable=False,
                                    name='kernel',
                                    )
        # self.spike_state=tf.Variable(tf.zeros((32,self.num_neurons)),trainable=False)
        # self.membrane_potential=tf.Variable(tf.zeros((32,self.num_neurons)),trainable=False)
        super(SpikingNeuronLayer,self).build(input_shape)

    def call(self, inputs,training=False):
        new_kernel = self.kernel*tf.pow(self.rate,self.current_time)
        if inputs.shape.rank == 4:
            inputs_flat = tf.reshape(inputs, [-1,np.prod(inputs.shape[1:])])
        else:
            raise ValueError("inputs must have rank 4, but got {}".format(inputs.shape.rank))
        batch_size=tf.shape(inputs)[0]
        new_potential = tf.matmul(inputs_flat,new_kernel)
        # Based on tf.init_scope and create the original states
        if not self.initialized:
            self.initialized = True
            self.spike_state=tf.zeros((batch_size,self.num_neurons))
            self.membrane_potential=tf.zeros((batch_size,self.num_neurons))
            # self.spike_state=tf.tile(tf.expand_dims(self.spike_state[0],0),[batch_size,1])
            # self.membrane_potential=tf.tile(tf.expand_dims(self.membrane_potential[0],0),[batch_size,1])
            # self.spike_state=spike_state
            # self.membrane_potential=membrane_potential
        # spike_state=tf.tile(spike_state_initializer[0],[batch_size,1])
        # membrane_potential=tf.tile(membrane_potential_initializer[0],[batch_size,1])
        # if self.spike_state is None or self.membrane_potential is None:
        #     self.spike_state=spike_state
        #     self.membrane_potential=membrane_potential
        # new_potential=tf.ensure_shape(new_potential,self.membrane_potential.shape)
        # if self.membrane_potential.shape==new_potential.shape:
        self.membrane_potential = self.membrane_potential+new_potential
        # else:
        #     raise ValueError("membrane_potential must have same shape as inputs, but we have ",self.membrane_potential.shape,"for mem and ",new_potential.shape,"for inputs"," and the truth for initialization is ",self.initialized)
        spikes = tf.cast(tf.greater_equal(self.membrane_potential, self.threshold), tf.float32)
        self.membrane_potential=self.reset_value*spikes
        new_potential=self.membrane_potential
        current_state=self.spike_state
        new_states = current_state+spikes
        # else:
        #     raise ValueError("Spike state and membrane potential must have same shape, but the shapes of the state and spike are ", self.spike_state.shape,"and", spikes.shape, "respectively.")
        # new_states = tf.tan(snn_state)
        if training:
            self.current_time=self.current_time+1
        return new_states,new_potential
def creat_snn_model(input_shape, num_neurons, rate, current_time, threshold=1.0, reset_value=0.0, num_classes=2):
    inputs=Input(shape=input_shape)
    spiking_layer=SpikingNeuronLayer(num_neurons=num_neurons,
                                     rate=rate,
                                     current_time=current_time,
                                     threshold=threshold,
                                     reset_value=reset_value)
    #Initializition
    spiking_layer.build(input_shape)
    spiking_layer.spike_state=tf.zeros((tf.shape(inputs)[0],num_neurons))
    spiking_layer.membrane_potential=tf.zeros((tf.shape(inputs)[0],num_neurons))
    spiking_output=spiking_layer(inputs)[0]
    dense_layer=Dense(num_classes,activation='softmax')
    outputs=dense_layer(spiking_output)
    model=Model(inputs=inputs,outputs=outputs)
    return model

# Display the parameters of the Dense of the model
def replace_output_layer_params_with_ltp(model, ltp_values, output_layer_name):
    """
    使用LTP值替换模型输出层参数。

    参数:
    - model: 训练好的Keras模型。
    - ltp_values: 从Excel表格读取的LTP值列表。
    - output_layer_name: 需要替换参数的输出层的名称。

    返回:
    - 替换后的模型。
    """
    # 将LTP值转换为NumPy数组
    ltp_values = np.array(ltp_values)

    # 获取模型输出层的权重和偏置
    output_layer = model.get_layer(name=output_layer_name)
    weights, biases = output_layer.get_weights()
    weights_flatten = weights.flatten()

    # 找到最接近的LTP值并替换权重和偏置
    replaced_weights_flatten = np.array([np.interp(w, ltp_values, ltp_values) for w in np.nditer(weights_flatten)])
    replaced_biases = np.array([np.interp(b, ltp_values, ltp_values) for b in np.nditer(biases)])

    replaced_weights = replaced_weights_flatten.reshape(weights.shape)

    # 更新输出层的权重和偏置
    output_layer.set_weights([replaced_weights, replaced_biases])

    return model


#Visulization of the accuracy and loss
#Visulization of the accuracy
def description(history):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'],label='Training accuracy')#Make the figure of the training accuracy
    plt.plot(history.history['val_accuracy'],label='Validation accuracy')#Make the figure of the Validation accuracy
    plt.title('Training and validation accuracy over the epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #Visulization of loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'],label='Training loss')#Figure of training loss
    plt.plot(history.history['val_loss'],label='Validation loss')#Figure of validation loss
    plt.title('Training and validation loss over the epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

#The test of the final performance of the model
def some_performance(model,test_image):
    model.predict(test_image[:10])
    start_time=time.time()
    predictions=model.predict(test_image)
    end_time=time.time()

    latency=end_time-start_time
    throughput=test_image.shape[0]/latency
    return latency,throughput

def model_main(model,num_classes,displace=False):
    # loading the dataset
    smile_images = preprocess_image('/home/huawei/files/ML/reservior/archive/smile')
    non_smile_images = preprocess_image('/home/huawei/files/ML/reservior/archive/non_smile')
    test_images = preprocess_image('/home/huawei/files/ML/reservior/archive/test')
    print(smile_images.shape, "\n", non_smile_images.shape, "\n", test_images.shape)
    # Divide the datasets to training set and testing set

    X_smile_train, X_smile_test, y_smile_train, y_smile_test = train_test_split(smile_images,
                                                                                np.ones((len(smile_images), 1)),
                                                                                test_size=0.2,
                                                                                random_state=42)
    X_non_smile_train, X_non_smile_test, y_non_smile_train, y_non_smile_test = train_test_split(non_smile_images,
                                                                                                np.zeros((
                                                                                                    len(non_smile_images),
                                                                                                    1)),
                                                                                                test_size=0.2,
                                                                                                random_state=42)
    test_labels = np.array(
        [1] * 3 + [0] * 5 + [1] * 1 + [0] * 1 + [1] * 1 + [0] * 5 + [1] * 1 + [0] * 1 + [1] * 1 + [0] * 1 + [1] * 1 + [
            0] * 1 + [1] * 4 + [0] * 1 + [1] * 1 + [0] * 2 + [1] * 1 + [0] * 3 + [1] * 1 + [0] * 1 + [1] * 5 + [
            0] * 1 + [1] * 3 + [0] * 4 + [1] * 1)
    # Show the shape of the training set and the shape of the testing set
    print(X_smile_train.shape, X_smile_test.shape)
    print(y_smile_train.shape, y_smile_test.shape)
    print(X_non_smile_train.shape, X_non_smile_test.shape)
    print(y_non_smile_train.shape, y_non_smile_test.shape)
    # Rearrange the training set and the testing set to make sure that there are bothe smiling faces and non-smiling faces in both sets
    X_train_i = []
    y_train_i = []
    for smile_im, non_smile_im, smile_label, non_smile_label in zip(X_smile_train, X_non_smile_train, y_smile_train,
                                                                    y_non_smile_train):
        X_train_i.append(smile_im)
        y_train_i.append(smile_label)
        X_train_i.append(non_smile_im)
        y_train_i.append(non_smile_label)
    X_train = np.array(X_train_i)
    y_train = np.array(y_train_i)
    print(X_train.shape, y_train.shape)
    X_test_i = []
    y_test_i = []
    for smile_ima, non_smile_ima, smile_lab, non_smile_lab in zip(X_smile_test, X_non_smile_test, y_smile_test,
                                                                  y_non_smile_test):
        X_test_i.append(smile_ima)
        y_test_i.append(smile_lab)
        X_test_i.append(non_smile_ima)
        y_test_i.append(non_smile_lab)
    X_test = np.array(X_test_i)
    y_test = np.array(y_test_i)
    print(X_test.shape, y_test.shape)
    # Visulization of each set
    # plot_grayscale_histogram(smile_images[:15])
    # plot_grayscale_histogram(non_smile_images[:15])
    # plot_grayscale_histogram(X_train[:15])
    # plot_grayscale_histogram(X_test[:15])
    # plot_grayscale_histogram(test_images[:50])


    # We need let the lable contains "0" and "1"
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes)

    # Let's train the model hahahahaahahahaha !!!!!!!
    # Edit the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    # Training the model! and see the aaccuracy and the loss
    history = model.fit(X_train, y_train_one_hot, batch_size=32, epochs=40,
                                        validation_data=(X_test, y_test_one_hot))
    description(history)
    if displace:
       params_df = pd.read_csv("/home/huawei/files/ML/reservior/archive/data_transformation/LTP.csv", dtype=float)
       excel_params = params_df.values.flatten()
       model_fited=replace_output_layer_params_with_ltp(model,excel_params,'dense')
    else:
        model_fited=model
    # The final performance of the model
    test_loss, test_acc = model_fited.evaluate(test_images[:50], test_labels_one_hot,
                                                         verbose=2)  # Change the labels, too!
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Test loss: {test_loss * 100:.2f}%")
    j = 0
    for i, image in enumerate(test_images[:50]):
        prediction = model_fited.predict(np.expand_dims(image, axis=0))
        predict_class = np.argmax(prediction)
        true_class = np.argmax(test_labels_one_hot[i])  # Change the labels to the Test data labels
        print("Image ", i + 1)
        print(f"Predicted class: {predict_class}")
        print(f"True class: {true_class}")
        if predict_class == true_class:
            j = j + 1
    print(f"There are {j} images which are similarly predicted by the model.")
    print(f"And {50 - j} images are not the same.")
    latency = some_performance(model_fited, test_images)[0]
    throughput = some_performance(model_fited, test_images)[1]
    print("The latency of the model for predicting the first 10 images is: ", latency)
    print("The throughput of the model is: ", throughput)




# The parameter of model
input_shape = (64, 64, 1)  # 64*64 gary image
reservoir_dim = 1000  # The dimension of reservoir
num_classes = 2  # The classes number
decay_rate = 0.5


# Creat SNN model
# model_snn=creat_snn_model(input_shape,reservoir_dim,decay_rate,0,1.0,0.0,num_classes)
# model_snn.summary()
# model_main(model_snn,num_classes)


# Creat LSM model
# model_LSM=create_LSM(input_shape,reservoir_dim,num_classes,decay_rate)
# model_LSM.summary()
# model_main(model_LSM,num_classes,True)

# Creat ESN model
# model_traditional_esn = create_esn(input_shape,reservoir_dim, num_classes)
# model_traditional_esn.summary()
# model_main(model_traditional_esn,num_classes,True)

# Create CNN with reservoir computing without pooling layers
model_NoPooling_cnn=create_esn2(input_shape, num_classes)
model_NoPooling_cnn.summary()
model_main(model_NoPooling_cnn,num_classes,True)

# Create CNN with reservoir computing and pooling layers
# model_Pooling_cnn=create_cnn(input_shape, num_classes)
# model_Pooling_cnn.summary()
# model_main(model_Pooling_cnn,num_classes,True)


# Here we create a common cnn model

# Create
# model=Sequential([
#     layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(num_classes, activation='softmax')
# ])
# # print the summary of the model
# model.summary()
# model_main(model,num_classes,True)
