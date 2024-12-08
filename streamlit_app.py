import streamlit as st
import pandas as pd
import openai
import numpy as np
from flag import flag
import plotly.express as px
import pycountry
import json
from questions import questions

# Set up OpenAI API key
openai.api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

st.set_page_config(page_title="Big Five", layout="wide")

def calculate_big_five_scores(responses):
    """Calculate Big Five scores based on user responses."""
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    scores = {trait: sum(responses[i::5]) / len(responses[i::5]) for i, trait in enumerate(traits)}
    return scores

def rescale_t_scores_to_one_five(t_scores):
    """Rescale T-scores to a 1-5 scale."""
    t_min = t_scores.min()
    t_max = t_scores.max()
    return 1 + 4 * (t_scores - t_min) / (t_max - t_min)

def calculate_similarity(user_scores, country_scores):
    """Calculate similarity between user scores and country scores using Euclidean distance."""
    user_vector = np.array([user_scores[trait] for trait in user_scores.keys()])
    country_vector = np.array([country_scores[trait] for trait in user_scores.keys()])
    distance = np.linalg.norm(user_vector - country_vector)  # Euclidean distance
    return distance

def alpha2_to_alpha3(alpha2):
    """Convert Alpha-2 country code to Alpha-3."""
    try:
        return pycountry.countries.get(alpha_2=alpha2).alpha_3
    except AttributeError:
        return None

def get_flag_emoji(country_code):
    """Get the country flag emoji using the `flag` package."""
    try:
        return flag(country_code.upper())
    except Exception:
        return "❓"

def get_negotiation_advice(scores):
    """Use OpenAI to provide negotiation advice based on personality scores."""
    personality_profile = ", ".join([f"{trait}: {score:.2f}" for trait, score in scores.items()])
    prompt = f"Based on this personality profile: {personality_profile}, provide negotiation advice tailored to this person's strengths and weaknesses."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while fetching negotiation advice: {e}")
        return "Unable to generate advice due to an error."

def render_country_table(df, title):
    """Render a country table with emoji flags."""
    st.markdown(f"### {title}")
    table_html = "<table>"
    table_html += "<tr><th>Country</th><th>Similarity</th><th>Flag</th></tr>"
    for _, row in df.iterrows():
        table_html += (
            f"<tr>"
            f"<td>{row['country']}</td>"
            f"<td>{row['similarity']:.2f}</td>"
            f"<td>{get_flag_emoji(row['country_code'])}</td>"
            f"</tr>"
        )
    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)

def render_similarity_map(similarity_df):
    """Render a choropleth map based on similarity scores."""
    st.subheader("Global Similarity Map")

    # Load GeoJSON file 
    with open("countries.geo.json", "r") as geojson_file:
        geojson_data = json.load(geojson_file)

    # Convert Alpha-2 to Alpha-3 codes
    similarity_df['country_code_alpha_3'] = similarity_df['country_code'].apply(alpha2_to_alpha3)

    map_data = similarity_df.rename(columns={"country_code_alpha_3": "iso_alpha", "similarity": "Similarity"})

    fig = px.choropleth(
        map_data,
        geojson=geojson_data,
        locations="iso_alpha",
        color="Similarity",
        hover_name="country",
        hover_data=["Similarity"],
        color_continuous_scale="Viridis",
        title="Similarity of Countries Based on Big Five Traits"
    )
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="gray",  # Coastline color for contrast
        showland=True,
        landcolor="black",  # Land color
        showocean=True,
        oceancolor="black",  # Ocean color
        fitbounds="locations"
    )
    fig.update_layout(
        autosize=False,
        paper_bgcolor="black",  # Background color
        plot_bgcolor="black",  # Plot area background color
        font_color="white",  # Font color for text
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_colorbar={
            "title": "Similarity",
            "tickfont": {"color": "white"},  # Tick font color
            "titlefont": {"color": "white"}  # Title font color
        }
    )
    st.plotly_chart(fig, use_container_width=True)

# Inside compare_with_dataset function
def compare_with_dataset(scores, dataset_path):
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Normalize column names in the dataset
        df.columns = [col.lower() for col in df.columns]

        # Check if the required columns exist
        t_score_columns = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        if not all(col in df.columns for col in t_score_columns + ["country", "country_code"]):
            raise ValueError("Dataset is missing required columns, including 'country' and 'country_code'.")

        # Drop rows with missing values in Big Five columns
        df = df.dropna(subset=t_score_columns)

        # Ensure the `country_code` column contains valid country codes
        if "country_code" in df.columns:
            df["country_code"] = df["country_code"].apply(lambda x: str(x).strip().upper() if isinstance(x, str) else "UNKNOWN")
        else:
            raise ValueError("Dataset is missing the 'country_code' column.")

        # Extract relevant Big Five columns and rescale them
        df_rescaled = df[t_score_columns].apply(rescale_t_scores_to_one_five)
        df_rescaled = pd.concat([df_rescaled, df[["country", "country_code"]].reset_index(drop=True)], axis=1)  # Add country and country_code columns

        # Convert user scores to lowercase keys for comparison
        scores = {k.lower(): v for k, v in scores.items()}

        # Calculate similarity scores for each country
        similarities = []
        for _, row in df_rescaled.iterrows():
            country_scores = row[t_score_columns].to_dict()
            similarity = calculate_similarity(scores, country_scores)
            similarities.append({"country": row["country"], "country_code": row["country_code"], "similarity": similarity})

        # Convert to DataFrame and sort by similarity
        similarity_df = pd.DataFrame(similarities)
        similarity_df = similarity_df.sort_values(by="similarity", ascending=True)  # Lower distance = more similar

        # Create the map visualization
        render_similarity_map(similarity_df)

        # Create a centered container for tables
        with st.container():
            col1, col2 = st.columns((0.1, 0.1))

            with col1:
                render_country_table(similarity_df.head(5), "Top 5 Most Similar Countries")

            with col2:
                render_country_table(similarity_df.tail(5), "Top 5 Least Similar Countries")

    except FileNotFoundError:
        st.error(f"Dataset not found at {dataset_path}. Please upload the correct file.")
    except ValueError as ve:
        st.error(f"Dataset format error: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Streamlit App
def main():
    st.title("Big Five Personality Test & Negotiation Advisor")

    # Initialize session state variables
    if "test_submitted" not in st.session_state:
        st.session_state.test_submitted = False
    if "advice_generated" not in st.session_state:
        st.session_state.advice_generated = False

    # Step 1: Big Five Questionnaire
    with st.expander("Step 1: Take the Personality Test", expanded=True):
        responses = {}

        # Display questions with sliders grouped by trait
        for trait, trait_questions in questions.items():
            st.subheader(trait)
            st.caption("Rate on a scale of 1 (Strongly Disagree) to 5 (Strongly Agree).")
            responses[trait] = []
            for question in trait_questions:
                response = st.slider(question, 1, 5, 3, key=f"{trait}_{question}")
                responses[trait].append(response)

        # Step 1 Button: Submit Test
        if st.button("Submit Test"):
            st.session_state.test_submitted = True
            st.session_state.advice_generated = False  # Reset advice state when test is resubmitted

    if st.session_state.test_submitted:
        # Flatten responses for scoring
        flattened_responses = [response for trait_responses in responses.values() for response in trait_responses]

        # Step 2: Calculate Scores
        scores = calculate_big_five_scores(flattened_responses)
        st.subheader("Your Personality Profile")

        # Create a bar graph with distinct colors for each bar
        traits = list(scores.keys())
        values = list(scores.values())
        colors = ["#C2185B", "#7C4DFF", "#536DFE", "#00BCD4", "#64FFDA"]  # Example color palette

        fig = px.bar(
            x=traits,
            y=values,
            labels={"x": "Traits", "y": "Score"},
            title="Your Big Five Personality Profile",
            color=traits,  # Add a color dimension
            color_discrete_sequence=colors  # Assign custom colors
        )

        fig.update_layout(
            xaxis_title="Traits",
            yaxis_title="Score",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)


        # Step 3: Compare with Dataset
        st.header("Step 2: Country Similarities")
        dataset_path = "/workspaces/chatbot/data/global_bigfive_data.csv"  # Replace with your actual dataset path
        compare_with_dataset(scores, dataset_path)

        # Step 4: Get Negotiation Advice
        st.header("Step 3: Personalized Negotiation Advice")
        if st.button("Generate Advice"):
            st.session_state.advice_generated = True

        if st.session_state.advice_generated:
            advice = get_negotiation_advice(scores)
            st.write(advice)

if __name__ == "__main__":
    main()
