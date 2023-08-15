import streamlit as st
from demo import (
    mlp_model,
    cnn_model,
    get_mlp_model_summary_path,
    get_cnn_model_summary_path,
)


st.header("MLP model architecture")
with st.expander("MLP architecture summary image"):
    mlp_model_summary_path = get_mlp_model_summary_path(mlp_model)
    st.image(mlp_model_summary_path)
st.markdown("""
    ```
    Model: "svhn_classifier_mlp"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_layer (Flatten)     (None, 1024)              0         
    
     dense_layer_1 (Dense)       (None, 512)               524800    
    
     dense_layer_2 (Dense)       (None, 256)               131328    
    
     dense_layer_3 (Dense)       (None, 128)               32896     
    
     dense_layer_4 (Dense)       (None, 64)                8256      
    
     softmax_classifier (Dense)  (None, 10)                650       
    
    =================================================================
    Total params: 697,930
    Trainable params: 697,930
    Non-trainable params: 0
    _________________________________________________________________
    ```
""")

st.header("CNN model architecture")
with st.expander("CNN architecture summary image"):
    cnn_model_summary_path = get_cnn_model_summary_path(cnn_model)
    st.image(cnn_model_summary_path)
st.markdown("""
    ```
    Model: "svhn_classifier_cnn"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     convolution_1 (Conv2D)      (None, 32, 32, 16)        160       
                                                                     
     max_pool_1 (MaxPooling2D)   (None, 16, 16, 16)        0         
                                                                     
     batch_normalization (BatchN  (None, 16, 16, 16)       64        
     ormalization)                                                   
                                                                     
     dropout (Dropout)           (None, 16, 16, 16)        0         
                                                                     
     convolution_2 (Conv2D)      (None, 16, 16, 8)         1160      
                                                                     
     max_pool_2 (MaxPooling2D)   (None, 8, 8, 8)           0         
                                                                     
     batch_normalization_1 (Batc  (None, 8, 8, 8)          32        
     hNormalization)                                                 
                                                                     
     dropout_1 (Dropout)         (None, 8, 8, 8)           0         
                                                                     
     flatten_layer (Flatten)     (None, 512)               0         
                                                                     
     dense_layer_1 (Dense)       (None, 256)               131328    
                                                                     
     dropout_2 (Dropout)         (None, 256)               0         
                                                                     
     dense_layer_2 (Dense)       (None, 128)               32896     
                                                                     
     dropout_3 (Dropout)         (None, 128)               0         
                                                                     
     softmax_classifier (Dense)  (None, 10)                1290      
                                                                     
    =================================================================
    Total params: 166,930
    Trainable params: 166,882
    Non-trainable params: 48
    _________________________________________________________________
    ```
""")
