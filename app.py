import plotly.express as px
import streamlit as st
from streamlit_folium import folium_static
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import requests
import sklearn
from PIL import Image
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn import preprocessing
PRIMARY_COLOR = "#1E88E5"
SECONDARY_COLOR = "#EEEEEE"
TEXT_COLOR = "#FFFFFF"
from streamlit_option_menu import option_menu
import folium


st.set_page_config(
    page_title="My Streamlit App",
    page_icon="assets/icon.jpeg",

)



selected =option_menu(menu_title=None,options=["Basic EDA","Advance EDA","ML Model"],orientation="horizontal",)



if selected=="Basic EDA":
    # Define the coordinates of your points
    
    points = {
        'Aundh-Sangvi 28': (18.5689,73.8175),
        'Baner-32': (18.5590,73.7868),
        'Chinchwad-3': (18.6298, 73.7997),
        'Dapodi-31': (18.5853, 73.8334),
        'Dattawadi-19': (18.4961, 73.8384),
        'DP road-38': (18.5296, 73.8760 ),
        'Pimple gurav-32': (18.5866, 73.8134),
        'Kalewadi-10': (18.6171, 73.7903),
        'kothrud-44': (18.5074, 73.8077),
        'pashan-1': (18.5415, 73.7925),
        'pimpri-25': (18.6298, 73.7997),
        'Yerwada rto -17': (18.5529, 73.8797),
        'Shivajinagar-76': (18.5314, 73.8446),
        'Someshwarwadi-7': (18.5467, 73.8022),
        'Vitthalwadi-19': (18.4818, 73.8296),
        'warje-29': (18.4865, 73.7968),
        'Yerwada-22': (18.5529, 73.8797),
    }

    # Create a map object centered on the first point
    m = folium.Map(location=list(points.values())[0], zoom_start=13)

    # Add markers for each point
    for name, coord in points.items():
        folium.Marker(coord, tooltip=name).add_to(m)

    # Display the map on Streamlit
    st.title("Map that displays the locations along with their corresponding total number of positive cases over a period of time. ")
    folium_static(m)
    df = pd.read_csv('assets/pmc_2.csv')

    # convert the "Date" column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # extract the month and year information from the "Date" column
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # filter the rows to only include the entries from the year 2022
    df_2022 = df[df['Year'] == 2021]

    # calculate the number of positive cases for each month
    cases_per_month = {}
    for i in range(1, 13):
        cases_per_month[i] = df_2022[(df_2022['Month'] == i) & (df_2022['dailysamples'] >6800)].shape[0]

    # plot the bar graph
    fig, ax = plt.subplots()
    bars = ax.bar(cases_per_month.keys(), cases_per_month.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'teal'])
    plt.title('no of cases posive in 2022')
    plt.xlabel('Month')
    plt.xticks(list(cases_per_month.keys()))
    plt.ylabel('Number of Cases')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_names)
    # add the text labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, height, ha='center', va='bottom')

        
    # Define the points
    points = {
        'Aundh-Sangvi 28': (18.5689,73.8175),
        'Baner-32': (18.5590,73.7868),
        'Chinchwad-3': (18.6298, 73.7997),
        'Dapodi-31': (18.5853, 73.8334),
        'Dattawadi-19': (18.4961, 73.8384),
        'DP road-38': (18.5296, 73.8760 ),
        'Pimple gurav-32': (18.5866, 73.8134),
        'Kalewadi-10': (18.6171, 73.7903),
        'kothrud-44': (18.5074, 73.8077),
        'pashan-1': (18.5415, 73.7925),
        'pimpri-25': (18.6298, 73.7997),
        'Yerwada rto -17': (18.5529, 73.8797),
        'Shivajinagar-76': (18.5314, 73.8446),
        'Someshwarwadi-7': (18.5467, 73.8022),
        'Vitthalwadi-19': (18.4818, 73.8296),
        'warje-29': (18.4865, 73.7968),
        'Yerwada-22': (18.5529, 73.8797),
    }

    # Calculate pairwise distances
    distances = np.zeros((len(points), len(points)))
    for i, (name_i, coord_i) in enumerate(points.items()):
        for j, (name_j, coord_j) in enumerate(points.items()):
            if i != j:
                distances[i, j] = np.sqrt((coord_i[0]-coord_j[0])**2 + (coord_i[1]-coord_j[1])**2)

    # Create heatmap
    st.header('Pairwise distances between points')
    figg, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(distances, cmap='viridis', annot=True, xticklabels=list(points.keys()), yticklabels=list(points.keys()))
    st.pyplot(figg)        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    # convert the Matplotlib figure to a Streamlit chart
    
    col1, col2 = st.columns(2)

    # Put content in the left column
    with col1:
        st.pyplot(fig)

    # Put content in the right column
    with col2:
        df = pd.read_csv('assets/data4.csv')

        # convert the "Date" column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # extract the month and year information from the "Date" column
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year

        # filter the rows to only include the entries from the year 2022
        df_2022 = df[df['Year'] == 2021]

        # calculate the number of positive cases for each month
        cases_per_month = {}
        for i in range(1, 13):
            cases_per_month[i] = df_2022[(df_2022['Month'] == i) & (df_2022['PCR_Result'] == 'Positive')].shape[0]

        # plot the bar graph
        fig, ax = plt.subplots()
        bars = ax.bar(cases_per_month.keys(), cases_per_month.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'teal'])
        plt.title('Number of Positive Cases per Month in 2021')
        plt.xlabel('Month')
        plt.xticks(list(cases_per_month.keys()))
        plt.ylabel('Number of Cases')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names)
        # add the text labels above each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, height, ha='center', va='bottom')


        st.pyplot(fig)
        
    # Load the dataset
    df = pd.read_csv('assets/data1.csv')

    # Convert Date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Group the data by Date and count the number of Positive cases
    daily_cases = df[df['PCR_Result'] == 'Positive'].groupby('Date').count()['PCR_Result']
    # Group the data by Location and count the number of Positive cases
    location_cases = df[df['PCR_Result'] == 'Positive'].groupby('Location').count()['PCR_Result']

    # Plot the location-wise positive cases
    st.set_option('deprecation.showPyplotGlobalUse', False) # Hide deprecation warning
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=location_cases.index, y=location_cases, ax=ax)
    plt.xticks(rotation=90)
    plt.title('Location-wise Positive Cases')
    plt.xlabel('Location')
    plt.ylabel('Number of Cases')

    # Display the plot in Streamlit
    st.pyplot(fig)    
    
    
    
    
    
    
    
if selected=="Advance EDA":    
    sel2 =option_menu(menu_title=None,options=["Comparitive graph between rise in positive cases and pcr result being positive","Graph to see flow of water in from the location points in pune","Corelation between two points"],orientation="vertical",)
    if sel2=="Comparitive graph between rise in positive cases and pcr result being positive":
        df1 = pd.read_csv('assets/pmc_2.csv')
        df2 = pd.read_csv('assets/data2.csv')

        # Convert the date columns to datetime format
        df1['Date'] = pd.to_datetime(df1['Date'], format='%d/%m/%Y')
        df2['Date'] = pd.to_datetime(df2['Date'], format='%d/%m/%Y')

        # Filter the DataFrame to only include rows with Positive results
        positive_df = df2[df2['PCR_Result'] == 'Positive']

        # Group the positive_df by date and count the number of positive cases
        positive_count = positive_df.groupby('Date').size().reset_index(name='NumPositive')

        # Set the title of the app
        st.title('COVID-19 Dashboard')

        # Allow user to select date range for the first graph
        date_range1 = st.sidebar.selectbox('Select Date Range (Graph 1)', ['1w', '2w', '1m', '3m', 'All'])

        if date_range1 == '1w':
            filtered_df1 = df1[df1['Date'] >= df1['Date'].max() - pd.Timedelta(days=7)]
        elif date_range1 == '2w':
            filtered_df1 = df1[df1['Date'] >= df1['Date'].max() - pd.Timedelta(days=14)]
        elif date_range1 == '1m':
            filtered_df1 = df1[df1['Date'] >= df1['Date'].max() - pd.DateOffset(months=1)]
        elif date_range1 == '3m':
            filtered_df1 = df1[df1['Date'] >= df1['Date'].max() - pd.DateOffset(months=3)]
        else:
            filtered_df1 = df1

        # Create the first interactive line graph using Plotly
        fig1 = px.line(filtered_df1, x='Date', y='dailyconfirmed', title='Daily Confirmed Cases')

        # Allow user to select date range for the second graph
        date_range2 = st.sidebar.selectbox('Select Date Range (Graph 2)', ['1w', '2w', '1m', '3m', 'All'])

        if date_range2 == '1w':
            filtered_df2 = positive_count[positive_count['Date'] >= positive_count['Date'].max() - pd.Timedelta(days=7)]
        elif date_range2 == '2w':
            filtered_df2 = positive_count[positive_count['Date'] >= positive_count['Date'].max() - pd.Timedelta(days=14)]
        elif date_range2 == '1m':
            filtered_df2 = positive_count[positive_count['Date'] >= positive_count['Date'].max() - pd.DateOffset(months=1)]
        elif date_range2 == '3m':
            filtered_df2 = positive_count[positive_count['Date'] >= positive_count['Date'].max() - pd.DateOffset(months=3)]
        else:
            filtered_df2 = positive_count

        # Create the second interactive line graph using Plotly
        fig2 = px.line(filtered_df2, x='Date', y='NumPositive', title='Positive Cases over Time')

        # Display the graphs side by side
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.plotly_chart(fig2, use_container_width=True)






    if sel2=="Graph to see flow of water in from the location points in pune":


        # Load the dataset
        df = pd.read_csv('assets/wastewater_flow.csv')

        # Create a function to generate the map
        def generate_map(df):
            # Create a map centered on Pune
            pune_map = folium.Map(location=[18.5204, 73.8567], zoom_start=12)

            # Add markers and lines for the flow data
            for index, row in df.iterrows():
                # Create a line connecting the source and destination points
                folium.PolyLine(locations=[[row['source_latitude'], row['source_longitude']],
                                           [row['destination_latitude'], row['destination_longitude']]],
                                color='blue',
                                weight=2.5,
                                opacity=0.7
                               ).add_to(pune_map)

                # Add a marker for the source location
                folium.Marker(location=[row['source_latitude'], row['source_longitude']],
                              popup=row['source_location'],
                              icon=folium.Icon(color='red', icon='info-sign')
                             ).add_to(pune_map)

                # Add a marker for the destination location
                folium.Marker(location=[row['destination_latitude'], row['destination_longitude']],
                              popup=row['destination_location'],
                              icon=folium.Icon(color='blue', icon='info-sign')
                             ).add_to(pune_map)

            return pune_map

        # Display the map using Streamlit
        st.title('Wastewater flow map')
        folium_static(generate_map(df))
        st.write("The above map shows the flow of waste water from different points where the sample is collected")

                
        
    
    if sel2=="Corelation between two points": 
       
        
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.title("Correlation Heatmap")

        # Ask user to select CSV files for the datasets
        filename1 = st.selectbox("Select CSV file for dataset 1:", ['Aundh', 'Baner', 'Chinchwad', 'Dapodi', 'Dattawadi', 'dp_road', 'gurav', 'Kalewadi', 'kothrud', 'pashan', 'pimpri', 'rto', 'shantinagar', 'shivajinagar', 'Someshwarwadi', 'Vitthalwadi', 'warje'])
        filename2 = st.selectbox("Select CSV file for dataset 2:",["yerwada",'Aundh', 'Baner', 'Chinchwad', 'Dapodi', 'Dattawadi', 'dp_road', 'gurav', 'Kalewadi', 'kothrud', 'pashan', 'pimpri', 'rto', 'shantinagar', 'shivajinagar', 'Someshwarwadi', 'Vitthalwadi', 'warje'])

        # Load the datasets from the CSV files
        df1 = pd.read_csv("assets/"+filename1+".csv")

        df2 = pd.read_csv("assets/"+filename2+".csv")

        # Calculate the correlation matrix between the two datasets
        corr = pd.concat([df1, df2]).corr()

        # Create a sidebar for the heatmap settings
        cmap = st.sidebar.selectbox("Choose a color map:", ("coolwarm", "viridis", "magma", "inferno"))
        annot = st.sidebar.checkbox("Show annotations", value=True)
        vmin = st.sidebar.number_input("Enter the minimum value for the color scale:", value=-1.0)
        vmax = st.sidebar.number_input("Enter the maximum value for the color scale:", value=1.0)

        # Plot the correlation heatmap
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=annot, fmt=".2f", cmap=cmap, cbar_kws={"orientation": "horizontal", "ticks": [-1, -0.5, 0, 0.5, 1]}, vmin=vmin, vmax=vmax, center=0, ax=ax)
        plt.title("Correlation Heatmap between the provided Datasets")
        st.pyplot(fig)    


if selected=="ML Model":    
    st.title("COVID-19 Test Result Predictor")

    # Load the data
    data = pd.read_csv("Data1.csv")

    # Split into features and target variable
    X = data[["CtN", "CtE", "CtRdRp"]]
    y = data["PCR_Result"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the logistic regression classifier and fit to the training data
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv('shivajinagar.csv')

    # Prompt the user to input a date in the specified format
    date_str = st.text_input('Enter a date in the format dd//mm//yy: ')

    if date_str:
        # Filter the DataFrame to only include rows with the specified date
        filtered_df = df[df['Date'] == date_str]

        if not filtered_df.empty:
            # Extract the values of CtN, CtE, and CtRdRp for the filtered rows
            ctn_values = filtered_df['CtN'].values
            cte_values = filtered_df['CtE'].values
            ctrdrp_values = filtered_df['CtRdRp'].values

            # Print the predicted values of CtN, CtE, and CtRdRp
            x = ctn_values.mean()
            y = cte_values.mean()
            z = ctrdrp_values.mean()
            st.write(f'Predicted values of CtN: {x}')
            st.write(f'Predicted values of CtE: {y}')
            st.write(f'Predicted values of CtRdRp: {z}')

            # Predict the PCR_Result for new CtN, CtE, and CtRdRp values
            new_data = pd.DataFrame({"CtN": [x], "CtE": [y], "CtRdRp": [z]})
            result = clf.predict(new_data)
            st.write(f'Predicted PCR_Result: {result[0]}')
        else:
            st.warning(f"No data available for the specified date: {date_str}")