import streamlit as st
import pickle5 as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np



#graph_objects gives you a higher level of customization on your charts

#We need to import another object such as we did with the model and scaler pickle objects
#For now we also need to know the maximum values for each columns because one we make our sliders
#   we want them to be proportional to their respective maximums, so they do not look confusing to use

# To do this, we will import our data from the csv file. We note the this is not standard procedure in real production
#   In the real world scenario we would import the csv as a pickle object under the "main.py" and then we would import it right below
# But to make it simpler for now, we will not do that 



def get_clean_data():
    data = pd.read_csv('data/data.csv')
    
    #we will drop column unamed 32 because it is all NAN's. Also need to drop the id column
    data = data.drop(['Unnamed: 32', 'id'], axis = 1)

    #We also need to drop the diagnosis column
    #We want Malicious (M) cells to be represented with "1" and Benine (B) will now be represented with 0

    #To do this, we will use a map function with key-value pairs. M  key, value  = 1
    data['diagnosis']  =  data['diagnosis'].map({'M' : 1, 'B' : 0})



    return data





def add_sidebar():
    st.sidebar.header('Cell Nuclei Measurements') 

    #We use this dtata here, because we want the clean columns for our slide bars
    data = get_clean_data()

    #Now we want the column namess of the sidebars inside of a list and we need the labels
    #The idea is that we have the label which is the value of the slider, the second paramter is the column name

    slider_labels = [
    ("Radius (mean)", "radius_mean"),
    ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"),
    ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"),
    ("Concave points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"),
    ("Fractal dimension (mean)", "fractal_dimension_mean"),
    ("Radius (se)", "radius_se"),
    ("Texture (se)", "texture_se"),
    ("Perimeter (se)", "perimeter_se"),
    ("Area (se)", "area_se"),
    ("Smoothness (se)", "smoothness_se"),
    ("Compactness (se)", "compactness_se"),
    ("Concavity (se)", "concavity_se"),
    ("Concave points (se)", "concave points_se"),
    ("Symmetry (se)", "symmetry_se"),
    ("Fractal dimension (se)", "fractal_dimension_se"),
    ("Radius (worst)", "radius_worst"),
    ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"),
    ("Concave points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"),
    ("Fractal dimension (worst)", "fractal_dimension_worst"),
]



    #Now we need to loop through all of these labels and column names and create a slider for each
    # in our list, there are only two total VALUES because this is a value pair list. We have the label ( that we chose the names of
    #   and then we have the "key" which is represented by the ACTUAL column names)
    # So these respective pairs are represented as VALUE PAIRS

    # This list is NOT to be confused with a dictionary, which consists of key:value pairs

    #This is the loop

    #this step is to create the dictionary to store out slider mean values
    input_dict = {} 

    for label, key in slider_labels:  
    #slider is a method of the sidebar function
    #The first argument that  the slider method takes is "label". and we are repsenting this with the label names we created...called "label"


        #Note that we could also just do st.slider. But we want to make sure the slider is inside the sidebar so we do it like this
        #Also note that this information here creates our mean value for our slider

        # When we put key after the dictionary, we are say that this key equals this label value
        input_dict[key] = st.sidebar.slider(
            label = label,        
            min_value = float(0),
            #using key selected the column associated with the label on our sliders
            max_value = float(data[key].max()),
            value=float(data[key].mean() )
            )

    return input_dict
    #
    # up above key = column name,    and value = mean value from our slider

#We need to make sure we scale the values so that they all range between 0 and 1
#We will need a dictionary to do this

def get_scaled_values(input_dict):
    #lets get our data first from our old function
    data = get_clean_data()
    #We only need the predictors
    X = data.drop(['diagnosis'], axis = 1)

    scaled_dict = {}

    #Here is our loop that will go through our dictionary key-value pairs
    #   including .items() , ensures that we loop through all of our items in the key-values pairs 
    #we use basic arithmetic to create a proportion
    for key, value in input_dict.items() : 
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val) 
        scaled_dict[key] = scaled_value 

    return scaled_dict    




def get_radar_chart(input_data): 

    #adjusting out input_data to be inbetween 0 and 1 for our Radar chart inputs
    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture' 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                  'Symmetry', 'Fractoinal Dimension']

    fig = go.Figure()


        
    fig.add_trace(go.Scatterpolar(
        #r represents the radius values. We must replace these with the values that come from our side bar
        #recall that input data is the data that are the values returned from side_bar
            # the sidebar returns a dictionary key value pair
            # key = name of the column, value  = value of the number (mean) in the sidebar
            # SO...We will call the "values" which are the MEANS..... input_data['value']
            #This will be a list of all of our elements
            # This is all comes from the object that we exported from the sidebar
            #NOTICE THAT WE ARE ONLY INCLUDING VALUES THAT HAVE THE WORD MEAN INCLUDED IN THE SIDEBAR
      r =   [   
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'], 
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'], 
          input_data['fractal_dimension_mean']

      ],
      #theta represents the categories on our chart
      theta=categories,
      #fill gives the shape color on the chart
      fill='toself',
      #name is the name of the traces for mean, SD, and max
      name='Mean Value'
))
    

    #We repeat calling the values from our dictionary but for the SE values
    fig.add_trace(go.Scatterpolar(
      r = [
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
],

      theta=categories,
      fill='toself',
      name='Standard Error'
))
    #Now for worst value
    fig.add_trace(go.Scatterpolar(
      r =   [
         input_data['radius_worst'] , input_data['texture_worst'] , input_data['perimeter_worst'] , input_data['area_worst'],
           input_data['smoothness_worst'] , input_data['compactness_worst'] , input_data['concavity_worst'],
         input_data['concave points_worst'] , input_data['symmetry_worst'] , input_data['fractal_dimension_worst'] 

      ],
    
      theta=categories,
      fill='toself',
      name='Worst Value'
))

    fig.update_layout(
    polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
    showlegend=True

)
 
    return fig




#Note that we do not need to fig.show() from the plotly documentation, ebcause streamlit has its own way of interacting with plotly
#   So we just need to do ' return fig '

def add_predictions(input_data) : 
    #Remeber that we are importing the model and the scaler because we are NOT  training the model in our application. We are IMPORTING IT
    #r = read mode, b = binary mode
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))

    #out model takes a  2-dimensional array****
    #on streamlit, this puts every single variable in one columnn, just as the original data set is.
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    #IMPORTANT STEP....We must scale our prediction values, because if not, the model will not work
    input_array_scaled = scaler.transform(input_array) 

    #Now that our values are all scaled, we can use these values for our predictions
    #****** THIS IS THE MACHINE LEARNING STEP!!!!!!!!! **************
    prediction = model.predict(input_array_scaled)

    #FINAL STEP
    #Make more user friendly with header and subheader
    st.subheader('Cell Cluster Prediction')
    st.write('The cell cluster is:')



    #Instead of just using st.write, lets make the prediction more user friendly
    #Note the prediction is an ARRAY... MUST USE []
    if prediction[0] == 0:
        #Lets make benign turn green on the page and malicious turn red on our page
        #In order for this html string to be parsed we must add the unsafe_allow_html = True
        st.write("<span class= 'diagnosis benign'>Benign</span>", unsafe_allow_html=True) 
    else:
        st.write("<span class= 'diagnosis malicious'>Malicious</span>",  unsafe_allow_html=True) 

    #Now we will find the probabilty of it being benign
    #predict_proba returns an array with two element. The first is the prob(0) and the second is prob(1)
    #so for this first st.write, we make the second element in brackets [0]
    #Likewise, we change the second index of the array, to [1], representing malicious
    st.write('The probability of being benign: ', model.predict_proba(input_array_scaled)[0][0])
    st.write('The probability of being malicious: ', model.predict_proba(input_array_scaled)[0][1])

    st.write('This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis')
    

    #note that when the scaler function works, it considers the mean as a value of 0. 
    #Any adjustments below are negative, any adjusments above are positive. This is just how it works
    #st.write(prediction)  



    #Recall that input_data is a dictionary of key-values pairs
    # Where the categoy name is the key, and the actual respective mean number is value
    #We need to be able to get all of these values in one place, as an array






def main():


    #Here, we are setting up the configuration of our home page
    st.set_page_config(
        page_title = 'Breast Cancer Predictor' , 
        page_icon = 'female-doctor:' ,
        layout = 'wide' , 
        initial_sidebar_state = 'expanded'

    )

    #f for file
    #This is essentially a hack because streamlit is not made to accept css
    with open('assets/style.css') as f:
        #style text, in side the style text add the file. f.read adds the contents of our file
        #allow_unsafe allows it to parse it as an html
        #This should import our styles file as a markdown file and update the style of our box
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)




    #Update the sidebar with out new dictionary
    #Because we created this input_data, everytime that we change a value on our slider it will adjust our value of the inpur dictionary up above
    #the sidebar is returning the data from those inputs when the slider changes
    input_data = add_sidebar() 

    #Check if we can see the physical values change
    #THIS TEP IS IMPORTANT BECAUSE THIS FLUCTUATING DATA IS WHAT WE WILL BUILD THE MODEL OFF OF ****************
    #st.write(input_data)

    
    #We want to make sure that everything is contained in one area or contain
    #There we will use the streamlit container function

    with st.container() :
        st.title("Breast Cancer Predictor")
        #this line represents the P element in html code ( just a description of the application for the users)
        st.write('Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learninig model whether a breast mass is benign or malignant based on the measurements it recieves from your cytosis lab. You can also update the measurements by hand using the slidebars in the sidebar.')

    col1, col2 = st.columns([4,1])

    with col1:
        #we need the figure that get_radar_chart returns by passing through our input data
        radar_chart = get_radar_chart(input_data)
        #we then pass in the figure element that we got from our function
        st.plotly_chart(radar_chart)
    with col2:
        #time to add our predictions
        add_predictions(input_data) 

    




if __name__ == '__main__' :
    main()
