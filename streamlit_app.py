import streamlit as st
import pandas as pd
import openai
import numpy as np
from flag import flag

# Set up OpenAI API key
openai.api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

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
            f"<td>{get_flag_emoji(row['country'])}</td>"
            f"</tr>"
        )
    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)

def compare_with_dataset(scores, dataset_path):
    """Compare user scores with a dataset of Big Five test results."""
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Normalize column names in the dataset
        df.columns = [col.lower() for col in df.columns]

        # Check if the required columns exist
        t_score_columns = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        if not all(col in df.columns for col in t_score_columns + ["country"]):
            raise ValueError("Dataset is missing required columns, including 'country'.")

        # Drop rows with missing values in Big Five columns
        df = df.dropna(subset=t_score_columns)

        # Ensure the `country` column contains valid country codes
        if "country" in df.columns:
            df["country"] = df["country"].apply(lambda x: str(x).strip().upper() if isinstance(x, str) else "UNKNOWN")
        else:
            raise ValueError("Dataset is missing the 'country' column.")

        # Extract relevant Big Five columns and rescale them
        df_rescaled = df[t_score_columns].apply(rescale_t_scores_to_one_five)
        df_rescaled = pd.concat([df_rescaled, df["country"].reset_index(drop=True)], axis=1)  # Add the country column

        # Convert user scores to lowercase keys for comparison
        scores = {k.lower(): v for k, v in scores.items()}

        # Calculate similarity scores for each country
        similarities = []
        for _, row in df_rescaled.iterrows():
            country_scores = row[t_score_columns].to_dict()
            similarity = calculate_similarity(scores, country_scores)
            similarities.append({"country": row["country"], "similarity": similarity})

        # Convert to DataFrame and sort by similarity
        similarity_df = pd.DataFrame(similarities)
        similarity_df = similarity_df.sort_values(by="similarity", ascending=True)  # Lower distance = more similar

        # Add emoji flags
        similarity_df["flag"] = similarity_df["country"].apply(lambda x: get_flag_emoji(str(x).upper()))

        # Top 5 most similar countries
        most_similar = similarity_df.head(5)
        render_country_table(most_similar, "Top 5 Most Similar Countries")

        # Top 5 least similar countries
        least_similar = similarity_df.tail(5)
        render_country_table(least_similar, "Top 5 Least Similar Countries")

    except FileNotFoundError:
        st.error(f"Dataset not found at {dataset_path}. Please upload the correct file.")
    except ValueError as ve:
        st.error(f"Dataset format error: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Streamlit App
def main():
    st.title("Big Five Personality Test & Negotiation Advisor")

    # Step 1: Big Five Questionnaire
    st.header("Step 1: Take the Personality Test")
    
    # Define Big Five questions (10 for each trait for better granularity)
    questions = {
        "Openness": [
            "I have a vivid imagination.",
            "I am interested in abstract ideas.",
            "I enjoy artistic and creative experiences.",
            "I value aesthetic and artistic qualities.",
            "I prefer variety to routine.",
            "I enjoy philosophical discussions.",
            "I am open to trying new activities.",
            "I enjoy thinking about complex concepts.",
            "I am curious about many different things.",
            "I have an active imagination."
        ],
        "Conscientiousness": [
            "I like to plan ahead.",
            "I pay attention to details.",
            "I complete tasks on time.",
            "I am highly organized.",
            "I follow through with commitments.",
            "I set and stick to goals.",
            "I am self-disciplined.",
            "I strive for excellence in everything I do.",
            "I keep my workspace clean and orderly.",
            "I work hard to achieve my ambitions."
        ],
        "Extraversion": [
            "I feel comfortable around people.",
            "I start conversations easily.",
            "I am outgoing and sociable.",
            "I enjoy being the center of attention.",
            "I make friends easily.",
            "I enjoy social gatherings.",
            "I find it easy to express my opinions.",
            "I am energized by interacting with others.",
            "I feel at ease in large crowds.",
            "I like to meet new people."
        ],
        "Agreeableness": [
            "I am considerate of others' feelings.",
            "I sympathize with others' problems.",
            "I enjoy helping others.",
            "I am trusting of others.",
            "I avoid conflict whenever possible.",
            "I am generous with my time and resources.",
            "I am quick to forgive others.",
            "I work well with others in a team.",
            "I value harmony in relationships.",
            "I care deeply about the well-being of others."
        ],
        "Neuroticism": [
            "I often feel anxious or stressed.",
            "I get upset easily.",
            "I am sensitive to criticism.",
            "I frequently worry about things.",
            "I have mood swings.",
            "I find it hard to cope with stress.",
            "I feel insecure in unfamiliar situations.",
            "I am easily discouraged.",
            "I often feel overwhelmed.",
            "I tend to focus on my shortcomings."
        ]
    }

    responses = {}

    # Display questions with sliders grouped by trait
    for trait, trait_questions in questions.items():
        st.subheader(trait)
        responses[trait] = []
        for question in trait_questions:
            response = st.slider(question, 1, 5, 3, key=f"{trait}_{question}")
            responses[trait].append(response)

    if st.button("Submit Test"):
        # Flatten responses for scoring
        flattened_responses = [response for trait_responses in responses.values() for response in trait_responses]
        
        # Step 2: Calculate Scores
        scores = calculate_big_five_scores(flattened_responses)
        st.subheader("Your Personality Profile")
        for trait, score in scores.items():
            st.write(f"{trait}: {score:.2f}")

        # Step 3: Compare with Dataset
        st.header("Step 2: Country Similarities")
        dataset_path = "/workspaces/chatbot/data/Ecology and Culture Cultural Variables.csv"  # Replace with your actual dataset path
        compare_with_dataset(scores, dataset_path)

        # Step 4: Get Negotiation Advice
        st.header("Step 3: Personalized Negotiation Advice")
        if st.button("Generate Advice"):
            advice = get_negotiation_advice(scores)
            st.write(advice)

if __name__ == "__main__":
    main()
