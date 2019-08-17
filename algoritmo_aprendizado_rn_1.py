
#Aprendizado - Retinopatias

############
# Referencia: https://inclass.kaggle.com/htoukour/neural-networks-to-predict-diabetes?scriptVersionId=6116987
############

from keras.models  import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD

shape = array_img.length

nn1 = Sequential([
    Dense(12, input_shape=(shape,), activation='sigmoid'),
    Dense(1, activation='sigmoid')
])

X = eyes_df.iloc[:, :-1].values
y = eyes_df["LABEL"].value # Label, pex: 'Tem Doença'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11111)

nn1.summary()

nn1.compile(SGD(lr = .003), "binary_crossentropy", metrics=["accuracy"])
run_hist_1 = neural_network_d.fit(X_train, y_train, epochs=500, \
                                  validation_data=(data_test, target_test), \
                                  verbose=False, shuffle=False)


