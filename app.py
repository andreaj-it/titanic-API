# main imports
import pandas as pd
import numpy as np
import streamlit as st
import random as rd
import pickle

#import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report, confusion_matrix

import altair as alt


#from sklearn.model_selection import cross_val_score #only when testing
np.random.seed(13) #random seed to keep predictions consistent

forest_clf = pickle.load(open("model.pickle",'rb'))

#model = pickle.load(open('titanic_model.pickle', 'rb'))
#model = LogisticRegression()
#model.fit(X, y)

## STREAMLIT
st.write("""
         # ¿Hubieras sobrevivido al desastre del Titanic?""")
st.image("https://media1.faz.net/ppmedia/aktuell/83311481/1.1703919/format_top1_breit/der-untergang-der-titanic-1912.jpg",
         caption = "este desastre")
#st.image("http://3.bp.blogspot.com/-C6WJdUOdAaA/UJ7WLxU6lUI/AAAAAAAAAXs/GkjXrSqV2go/s320/titanic2.jpg", caption = "Este no....")

st.write("""
         ## Cómo funciona:
         Dadas algunas entradas, el algoritmo le dará una predicción para su supervivencia.
         
         ### Alguna información para ayudar en su selección
         
         #### Las mujeres y los niños primero
         
         *Solo sobrevivió alrededor del 32% de los pasajeros.*
         
         Si eras un hombre, tus posibilidades de supervivencia eran mucho menores. La tasa de supervivencia general para los hombres fue de alrededor del 20%.
         Para las mujeres fue del 74% y para los niños del 52%.
         
         
         #### Los precios de las entradas del Titanic fueron:
         
         - Primera clase (suite parlor) = £ 870/$ 4350 ($ 113 075,78 2018)
         - Primera clase (litera) = £30/$150 ($3899,16 2018)
         - Segunda Clase = £12/$60 ($1,559.67 2018)
         - Tercera clase = £3 - £8/$15 - $40 ($389,92 a $1039,78 2018)
                
         El pasajero de más edad a bordo era Johan Svensson, de 74 años.
         """)

#st.markdown("[You can find more facts about the Titanic here](https://www.telegraph.co.uk/travel/lists/titanic-fascinating-facts/#:~:text=1.,2.)")
#st.markdown("[and here](https://titanicfacts.net/titanic-survivors/)")
#st.markdown("[Could Jack have lived? More about the famous door scene from the Titanic Movie](http://colgatephys111.blogspot.com/2012/11/could-jack-have-lived.html)")
        
st.sidebar.header("Parámetros de entrada del usuario")

### input needs to be scaled! geht raw X_train and then scale

def user_input_features():
    age = st.sidebar.slider("Edad", 1,75,30)
    fare = st.sidebar.slider("Tarifa",15,500,40)
    SibSp = st.sidebar.selectbox("¿Cuántos hermanos o cónyuges viajan con usted?",[0,1,2,3,4,5,6,7,8])
    Parch = st.sidebar.selectbox("¿Cuántos padres o hijos viajan con usted?",[0,1,2,3,4,5,6,7,8])
    cabin_multiple = st.sidebar.selectbox("¿Cuántas cabinas adicionales ha reservado??",[0,1,2,3,4])
    numeric_ticket = rd.randint(0, 1)
    Sex = st.sidebar.selectbox("Seleccione su género",["male","female"])
    Sex_female = 0 if Sex == "male" else 1
    Sex_male = 0 if Sex == "female" else 1
    Pclass = st.sidebar.selectbox("¿De qué clase es su billete?", [1,2,3])
    Pclass_1 = 1 if Pclass == 1 else 0; Pclass_2 = 1 if Pclass == 2 else 0; Pclass_3 = 1 if Pclass == 3 else 0
    boarding = st.sidebar.selectbox("¿Dónde abordaste el Titanic?", ["Cherbourg","Queenstown","Southampton"])
    Embarked_C = 1 if boarding == "Cherbourg" else 0; Embarked_Q = 1 if boarding == "Queenstown" else 0; Embarked_S = 1 if boarding == "Southampton" else 0
    
    data = {"Age": age,"norm_fare":np.log(fare+1),"SibSp":SibSp,"Parch":Parch,"cabin_multiple": cabin_multiple,
            "numeric_ticket":numeric_ticket,"Sex_female":Sex_female,"Sex_male":Sex_male,"Pclass_1": Pclass_1, "Pclass_2": Pclass_2, "Pclass_3": Pclass_3,
            "Embarked_C":Embarked_C,"Embarked_Q":Embarked_Q, "Embarked_S":Embarked_S}
    data = pd.DataFrame(data, index = [0])
    #scaler.transform(data)
    #age,np.log(fare+1), SibSp, Parch, cabin_multiple, numeric_ticket, Sex_female,Sex_male, Pclass_1,Pclass_2,Pclass_3, Embarked_C, Embarked_Q, Embarked_S
    return data

user_data = user_input_features()
#st.subheader("Sus parámetros de entrada:")
#st.dataframe(user_data.to_dict())

prediction = forest_clf.predict(user_data)

#input_data = (3,0,35,0,0)

#input_data_as_numpy_array = np.asarray(input_data)
#input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#prediction = model.predict(input_data_reshap)


st.title("Predicción de supervivencia")
#st.write(prediction[0])
if prediction[0] == 1:
    st.write("**Probablemente lo habrías logrado!**")
    st.image("https://thumbs-prod.si-cdn.com/pn1W-PCw0pwa_EpefSOduW74gcM=/fit-in/1072x0/https://public-media.si-cdn.com/filer/Titanic-survivors-drifting-2.jpg")
else: 
    st.write("Bueno... **Probablemente estés más seguro viendo la película!**")
    st.image("https://i2-prod.irishmirror.ie/incoming/article9830920.ece/ALTERNATES/s615b/0_Kate-Winslet-as-Rose-DeWitt-Bukater-and-Leonardo-DiCaprio-as-Jack-Dawson-in-Titanic.jpg")


######TESTING#####

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

#streamlit run app.py
#http://localhost:8501/